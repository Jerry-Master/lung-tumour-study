import gradio as gr
import os
import numpy as np
import shutil
import cv2
import json
from tumourkit.segmentation import hov_infer
from tumourkit.preprocessing import geojson2pngcsv_main, png2graph_main, hovernet2geojson_main, graph2centroids_main, centroidspng2csv_main, pngcsv2geojson_main
from tumourkit.postprocessing import join_hovprob_graph_main, draw_cells_main
from tumourkit.utils.preprocessing import create_dir
from tumourkit.classification import infer_gnn
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from argparse import Namespace
import logging
from logging import Logger
import argparse
from typing import Tuple, Optional


APP_DIR = os.path.dirname(os.path.abspath(__file__))


def download_file(url: str, filename: str):
    """
    Downloads a file from a specified URL and saves it to the given filename.

    :param url: The URL of the file to be downloaded.
    :type url: str

    :param filename: The name of the file to be saved.
    :type filename: str
    """
    response = requests.get(url, stream=True)
    file_size = int(response.headers.get('Content-Length', 0))
    showname = '...' + filename[-20:] if len(filename) > 20 else filename
    progress = tqdm(response.iter_content(1024), f"Downloading {showname}", total=file_size, unit="B", unit_scale=True, unit_divisor=1024)
    with open(filename, 'wb') as f:
        for data in progress.iterable:
            f.write(data)
            progress.update(len(data))
    progress.close()


def download_folder(url_folder: str, url_files: str, dirname: str):
    """
    Downloads multiple files from a specified folder URL and saves them to the given directory.

    :param url_folder: The URL of the folder containing the files to be downloaded.
    :type url_folder: str

    :param url_files: The files within the folder.
    :type url_files: str

    :param dirname: The name of the directory where the files will be saved.
    :type dirname: str
    """
    response = requests.get(url_folder)
    html_content = response.content
    soup = BeautifulSoup(html_content, 'html.parser')
    file_list = [span.get_text() for span in soup.find_all('span') if span.get_text().startswith('best_')]

    for file in file_list:
        filename = os.path.join(dirname, file)
        url = url_files + file
        response = requests.get(url, stream=True)
        file_size = int(response.headers.get('Content-Length', 0))
        showname = '...' + filename[-20:] if len(filename) > 20 else filename
        progress = tqdm(response.iter_content(1024), f"Downloading {showname}", total=file_size, unit="B", unit_scale=True, unit_divisor=1024)
        with open(filename, 'wb') as f:
            for data in progress.iterable:
                f.write(data)
                progress.update(len(data))
        progress.close()


def download_models_if_needed(hov_dataset: str, hov_model: str, gnn_model: str, weights_dir: str):
    """
    Downloads necessary models and files if they do not exist in the specified directory.

    :param hov_dataset: The name of the Hovernet dataset.
    :type hov_dataset: str

    :param hov_model: The name of the Hovernet model.
    :type hov_model: str

    :param gnn_model: The name of the GNN (Graph Neural Network) model.
    :type gnn_model: str

    :param weights_dir: The directory path where the models and files will be saved.
    :type weights_dir: str
    """
    if not os.path.exists(os.path.join(weights_dir, hov_dataset, hov_model + '.tar')):
        os.makedirs(os.path.join(weights_dir, hov_dataset), exist_ok=True)
        url = f'https://huggingface.co/Jerry-Master/Hovernet-plus-Graphs/resolve/main/{hov_dataset}/hovernet/{hov_model}.tar'
        filename = os.path.join(weights_dir, hov_dataset, hov_model + '.tar')
        download_file(url, filename)
    if not os.path.exists(os.path.join(weights_dir, hov_dataset, 'type_info.json')):
        os.makedirs(os.path.join(weights_dir, hov_dataset), exist_ok=True)
        url = f'https://huggingface.co/Jerry-Master/Hovernet-plus-Graphs/resolve/main/{hov_dataset}/hovernet/type_info.json'
        filename = os.path.join(weights_dir, hov_dataset, 'type_info.json')
        download_file(url, filename)
    if not os.path.exists(os.path.join(weights_dir, hov_dataset, gnn_model)) \
       or len(os.listdir(os.path.join(weights_dir, hov_dataset, gnn_model))) < 3:
        os.makedirs(os.path.join(weights_dir, hov_dataset, gnn_model), exist_ok=True)
        url_folder = f'https://huggingface.co/Jerry-Master/Hovernet-plus-Graphs/tree/main/{hov_dataset}/gnn/{gnn_model}/'
        url_files = f'https://huggingface.co/Jerry-Master/Hovernet-plus-Graphs/resolve/main/{hov_dataset}/gnn/{gnn_model}/'
        dirname = os.path.join(weights_dir, hov_dataset, gnn_model)
        download_folder(url_folder, url_files, dirname)


def create_input_dir(input_image: np.ndarray, delete_prev: bool):
    """
    Creates an input directory and saves the input image in it.

    :param input_image: The input image to be saved.
    :type input_image: np.ndarray

    :param delete_prev: Flag indicating whether to delete the previous input directory if it exists.
    :type delete_prev: bool
    """
    if os.path.exists(os.path.join(APP_DIR, 'tmp')):
        if delete_prev:
            shutil.rmtree(os.path.join(APP_DIR, 'tmp'))
    input_dir = os.path.join(APP_DIR, 'tmp', 'input')
    os.makedirs(input_dir, exist_ok=True)
    cv2.imwrite(os.path.join(input_dir, 'input_image.png'), input_image[:, :, ::-1])


def run_hovernet(hov_dataset: str, hov_model: str, num_classes: int, weights_dir: str):
    """
    Runs HoverNet inference using the specified HoverNet dataset, model, and weights.

    :param hov_dataset: The name of the HoverNet dataset.
    :type hov_dataset: str

    :param hov_model: The name of the HoverNet model.
    :type hov_model: str

    :param num_classes: The number of classes in the dataset.
    :type num_classes: int

    :param weights_dir: The directory path containing the HoverNet weights.
    :type weights_dir: str
    """
    newargs = {
        'nr_types': str(num_classes + 1),
        'type_info_path': os.path.join(weights_dir, hov_dataset, 'type_info.json'),
        'gpu': '0',
        'nr_inference_workers': '0',
        'model_path': os.path.join(weights_dir, hov_dataset, hov_model + '.tar'),
        'batch_size': '10',
        'shape': hov_model[:-2] if 'FT' in hov_model else hov_model,
        'nr_post_proc_workers': '0',
        'model_mode': 'original',
        'help': False
    }
    newsubargs = {
        'input_dir': os.path.join(APP_DIR, 'tmp', 'input'),
        'output_dir': os.path.join(APP_DIR, 'tmp', 'tmp_hov'),
        'draw_dot': False,
        'save_qupath': False,
        'save_raw_map': False,
        'mem_usage': '0.2'
    }
    hov_infer(newargs, newsubargs, 'tile')


def run_posthov(num_classes: int, logger: Logger):
    """
    Runs post-processing steps on the HoverNet predictions.

    :param num_classes: The number of classes in the HoverNet predictions.
    :type num_classes: int

    :param logger: The logger object used for logging messages.
    :type logger: Logger
    """
    newargs = Namespace(
        json_dir=os.path.join(APP_DIR, 'tmp', 'tmp_hov', 'json'),
        gson_dir=os.path.join(APP_DIR, 'tmp', 'gson_hov'),
        num_classes=num_classes
    )
    hovernet2geojson_main(newargs)
    newargs = Namespace(
        gson_dir=os.path.join(APP_DIR, 'tmp', 'gson_hov'),
        png_dir=os.path.join(APP_DIR, 'tmp', 'png_hov'),
        csv_dir=os.path.join(APP_DIR, 'tmp', 'csv_hov'),
        num_classes=num_classes
    )
    geojson2pngcsv_main(newargs)
    create_dir(os.path.join(APP_DIR, 'tmp', 'graphs'))
    newargs = Namespace(
        png_dir=os.path.join(APP_DIR, 'tmp', 'png_hov'),
        orig_dir=os.path.join(APP_DIR, 'tmp', 'input'),
        output_path=os.path.join(APP_DIR, 'tmp', 'graphs', 'raw'),
        num_workers=0
    )
    png2graph_main(newargs)
    newargs = Namespace(
        json_dir=os.path.join(APP_DIR, 'tmp', 'tmp_hov', 'json'),
        graph_dir=os.path.join(APP_DIR, 'tmp', 'graphs', 'raw'),
        output_dir=os.path.join(APP_DIR, 'tmp', 'graphs', 'hovpreds'),
        num_classes=num_classes
    )
    join_hovprob_graph_main(newargs, logger)


def run_graphs(gnn_dataset: str, gnn_model: str, num_classes: int, weights_dir: str):
    """
    Runs the GNN (Graph Neural Network) on the HoverNet graph predictions.

    :param gnn_dataset: The name of the GNN dataset.
    :type gnn_dataset: str

    :param gnn_model: The name of the GNN model.
    :type gnn_model: str

    :param num_classes: The number of classes in the GNN predictions.
    :type num_classes: int

    :param weights_dir: The directory path containing the GNN weights.
    :type weights_dir: str
    """
    disable_prior = 'no-prior' in gnn_model or 'void' in gnn_model
    disable_morph_feats = 'no-morph' in gnn_model or 'void' in gnn_model
    model_name = os.listdir(os.path.join(weights_dir, gnn_dataset, gnn_model))[0]
    model_name, ext = os.path.splitext(model_name)
    newargs = Namespace(
        node_dir=os.path.join(APP_DIR, 'tmp', 'graphs', 'hovpreds'),
        output_dir=os.path.join(APP_DIR, 'tmp', 'gnn_preds'),
        weights=os.path.join(weights_dir, gnn_dataset, gnn_model, model_name + '.pth'),
        conf=os.path.join(weights_dir, gnn_dataset, gnn_model, model_name + '.json'),
        normalizers=os.path.join(weights_dir, gnn_dataset, gnn_model, model_name + '.pkl'),
        num_classes=num_classes,
        disable_prior=disable_prior,
        disable_morph_feats=disable_morph_feats,
    )
    infer_gnn(newargs)


def run_postgraphs(num_classes: int):
    """
    Runs post-processing steps on the GNN predictions.

    :param num_classes: The number of classes in the GNN predictions.
    :type num_classes: int
    """
    newargs = Namespace(
        graph_dir=os.path.join(APP_DIR, 'tmp', 'gnn_preds'),
        centroids_dir=os.path.join(APP_DIR, 'tmp', 'centroids'),
        num_classes=num_classes,
    )
    graph2centroids_main(newargs)
    newargs = Namespace(
        centroids_dir=os.path.join(APP_DIR, 'tmp', 'centroids'),
        png_dir=os.path.join(APP_DIR, 'tmp', 'png_hov'),
        csv_dir=os.path.join(APP_DIR, 'tmp', 'csv_gnn'),
    )
    centroidspng2csv_main(newargs)
    newargs = Namespace(
        png_dir=os.path.join(APP_DIR, 'tmp', 'png_hov'),
        csv_dir=os.path.join(APP_DIR, 'tmp', 'csv_gnn'),
        gson_dir=os.path.join(APP_DIR, 'tmp', 'gson_gnn'),
        num_classes=num_classes
    )
    pngcsv2geojson_main(newargs)


def create_logger() -> Logger:
    """
    Creates a logger object for logging messages.

    :return: The logger object.
    :rtype: Logger
    """
    logger = logging.getLogger('gradio')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def overlay_outputs(hov_dataset: str, use_gnn: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Returns the overlayed images generated from HoverNet and optional GNN predictions.

    :param hov_dataset: The name of the HoverNet dataset.
    :type hov_dataset: str

    :param use_gnn: Flag indicating whether to include GNN predictions in the overlay.
    :type use_gnn: bool

    :return: A tuple containing the HoverNet overlay image and the GNN overlay image (if applicable).
    :rtype: Tuple[np.ndarray, Optional[np.ndarray]]
    """
    newargs = Namespace(
        orig_dir=os.path.join(APP_DIR, 'tmp', 'input'),
        png_dir=os.path.join(APP_DIR, 'tmp', 'png_hov'),
        csv_dir=os.path.join(APP_DIR, 'tmp', 'csv_hov'),
        output_dir=os.path.join(APP_DIR, 'tmp', 'overlay_hov'),
        type_info=os.path.join(APP_DIR, 'weights', hov_dataset, 'type_info.json'),
    )
    draw_cells_main(newargs)
    hov_dir = os.path.join(APP_DIR, 'tmp', 'overlay_hov')
    hov_file = os.listdir(hov_dir)[0]
    hov = cv2.imread(os.path.join(hov_dir, hov_file), -1)[:, :, ::-1]
    if use_gnn:
        newargs = Namespace(
            orig_dir=os.path.join(APP_DIR, 'tmp', 'input'),
            png_dir=os.path.join(APP_DIR, 'tmp', 'png_hov'),
            csv_dir=os.path.join(APP_DIR, 'tmp', 'csv_gnn'),
            output_dir=os.path.join(APP_DIR, 'tmp', 'overlay_gnn'),
            type_info=os.path.join(APP_DIR, 'weights', hov_dataset, 'type_info.json'),
        )
        draw_cells_main(newargs)
        gnn_dir = os.path.join(APP_DIR, 'tmp', 'overlay_gnn')
        gnn_file = os.listdir(gnn_dir)[0]
        gnn = cv2.imread(os.path.join(gnn_dir, gnn_file), -1)[:, :, ::-1]
    else:
        gnn = None
    return hov, gnn


LAST_HOV_MODEL = None
LAST_HOV_DATASET = None
LAST_IMG_HASH = None


def process_image(
        weights_dir: str,
        input_image: np.ndarray,
        hov_dataset: str,
        hov_model: str,
        gnn_model: str
        ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Processes an input image using HoverNet and optionally GNN models.

    :param weights_dir: The directory path containing the model weights. If None, default weights directory will be used.
    :type weights_dir: str

    :param input_image: The input image to be processed.
    :type input_image: np.ndarray

    :param hov_dataset: The name of the HoverNet dataset.
    :type hov_dataset: str

    :param hov_model: The name of the HoverNet model.
    :type hov_model: str

    :param gnn_model: The name of the GNN (Graph Neural Network) model.
    :type gnn_model: str

    :return: A tuple containing the HoverNet output image and the GNN output image (if applicable).
    :rtype: Tuple[np.ndarray, Optional[np.ndarray]]
    """
    # Cache information when possible
    global LAST_HOV_MODEL
    global LAST_IMG_HASH
    global LAST_HOV_DATASET
    input_hash = hash(str(input_image))
    if LAST_IMG_HASH is None:
        LAST_IMG_HASH = input_hash
    if LAST_HOV_MODEL is None or LAST_IMG_HASH is None or LAST_HOV_DATASET is None \
       or LAST_HOV_MODEL != hov_model or LAST_IMG_HASH != input_hash or LAST_HOV_DATASET != hov_dataset:
        LAST_HOV_MODEL = hov_model
        LAST_IMG_HASH = input_hash
        LAST_HOV_DATASET = hov_dataset
        delete_prev = True
    else:
        delete_prev = False

    # Monusac doesn't have graph attention
    if hov_dataset == 'monusac' and gnn_model == 'gat-full':
        gnn_model = 'None'

    if weights_dir is None:
        weights_dir = os.path.join(APP_DIR, 'weights')
    logger = create_logger()
    download_models_if_needed(hov_dataset, hov_model, gnn_model, weights_dir)
    with open(os.path.join(APP_DIR, 'weights', hov_dataset, 'type_info.json'), 'r') as f:
        type_info = json.load(f)
        num_classes = len(type_info.keys()) - 1
    create_input_dir(input_image, delete_prev)
    if delete_prev:
        run_hovernet(hov_dataset, hov_model, num_classes, weights_dir)
        run_posthov(num_classes, logger)
    use_gnn = gnn_model != 'None'
    if use_gnn:
        run_graphs(hov_dataset, gnn_model, num_classes, weights_dir)
        run_postgraphs(num_classes)
    hov, gnn = overlay_outputs(hov_dataset, use_gnn)
    return hov, gnn


def create_ui(weights_dir: str) -> gr.Interface:
    """
    Creates and returns a Gradio interface for the CNN+GNN demo.

    :param weights_dir: The directory path containing the model weights.
    :type weights_dir: str

    :return: A Gradio interface object.
    :rtype: gr.Interface
    """
    image_input = gr.Image(shape=(1024, 1024))
    hov_dataset = gr.Dropdown(choices=[
        'consep', 'monusac', 'breast', 'lung'
    ], label='Select dataset in which models were trained')
    hov_model = gr.Dropdown(choices=[
        '270', '270FT', '518', '518FT'
    ], label="Select Hovernet model")
    gnn_model = gr.Dropdown(choices=[
        'gcn-full', 'gat-full',
        'gcn-no-morph', 'gcn-no-prior', 'gcn-void',
        'None'
        ], label='Select GNN model')
    out1 = gr.Image(label='Hovernet')
    out2 = gr.Image(label='GNN')

    def func(a: np.ndarray, b: str, c: str, d: str):
        process_image(weights_dir, a, b, c, d)
    ui = gr.Interface(
        fn=func,
        inputs=[image_input, hov_dataset, hov_model, gnn_model],
        outputs=[out1, out2],
        title="CNN+GNN Demo",
        description="Upload an image to see the output of the algorithm.",
        examples=[[os.path.join(APP_DIR, 'examples', x), 'breast', '518FT', 'gcn-full'] for x in os.listdir(os.path.join(APP_DIR, 'examples'))]
    )
    return ui


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, default='localhost', help='Default: localhost.')
    parser.add_argument('--port', type=int, default=15000, help='Default: 15000.')
    parser.add_argument('--share', action='store_true', help='Whether to create public link for the gradio demo.')
    parser.add_argument('--weights-dir', type=str, help='Folder to save and load weights from. Leave it blank to use tumourkit internal folders.')
    return parser


def main():
    parser = _create_parser()
    args = parser.parse_args()
    ui = create_ui(args.weights_dir)
    ui.launch(server_name=args.ip, server_port=args.port, share=args.share)


if __name__ == '__main__':
    main()

import gradio as gr
import os
import numpy as np
import shutil
import cv2
import json
from tumourkit.segmentation import hov_infer
from tumourkit.preprocessing import geojson2pngcsv, png2graph, hovernet2geojson, graph2centroids, centroidspng2csv, pngcsv2geojson
from tumourkit.postprocessing import join_hovprob_graph
from tumourkit.utils.preprocessing import create_dir
from tumourkit.classification import infer_gnn
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from argparse import Namespace
import logging
from logging import Logger


APP_DIR = os.path.dirname(os.path.abspath(__file__))


def download_file(url: str, filename: str):
    response = requests.get(url, stream=True)
    file_size = int(response.headers.get('Content-Length', 0))
    progress = tqdm(response.iter_content(1024), f"Downloading {filename}", total=file_size, unit="B", unit_scale=True, unit_divisor=1024)
    with open(filename, 'wb') as f:
        for data in progress.iterable:
            f.write(data)
            progress.update(len(data))
    progress.close()


def download_folder(url_folder: str, url_files: str, dirname: str):
    response = requests.get(url_folder)
    html_content = response.content
    soup = BeautifulSoup(html_content, 'html.parser')
    file_list = [span.get_text() for span in soup.find_all('span') if span.get_text().startswith('best_')]

    for file in file_list:
        filename = os.path.join(dirname, file)
        url = url_files + file
        response = requests.get(url, stream=True)
        file_size = int(response.headers.get('Content-Length', 0))
        progress = tqdm(response.iter_content(1024), f"Downloading {filename}", total=file_size, unit="B", unit_scale=True, unit_divisor=1024)
        with open(filename, 'wb') as f:
            for data in progress.iterable:
                f.write(data)
                progress.update(len(data))
        progress.close()


def download_models_if_needed(hov_dataset: str, hov_model: str, gnn_dataset: str, gnn_model: str):
    if not os.path.exists(os.path.join(APP_DIR, 'weights', hov_dataset, hov_model + '.tar')):
        os.makedirs(os.path.join(APP_DIR, 'weights', hov_dataset), exist_ok=True)
        url = f'https://huggingface.co/Jerry-Master/Hovernet-plus-Graphs/resolve/main/{hov_dataset}/hovernet/{hov_model}.tar'
        filename = os.path.join(APP_DIR, 'weights', hov_dataset, hov_model + '.tar')
        download_file(url, filename)
    if not os.path.exists(os.path.join(APP_DIR, 'weights', hov_dataset, 'type_info.json')):
        os.makedirs(os.path.join(APP_DIR, 'weights', hov_dataset), exist_ok=True)
        url = f'https://huggingface.co/Jerry-Master/Hovernet-plus-Graphs/resolve/main/{hov_dataset}/hovernet/type_info.json'
        filename = os.path.join(APP_DIR, 'weights', hov_dataset, 'type_info.json')
        download_file(url, filename)
    if not os.path.exists(os.path.join(APP_DIR, 'weights', gnn_dataset, gnn_model)) \
        or len(os.listdir(os.path.join(APP_DIR, 'weights', gnn_dataset, gnn_model))) < 3:
        os.makedirs(os.path.join(APP_DIR, 'weights', gnn_dataset, gnn_model), exist_ok=True)
        url_folder = f'https://huggingface.co/Jerry-Master/Hovernet-plus-Graphs/tree/main/{gnn_dataset}/gnn/{gnn_model}/'
        url_files = f'https://huggingface.co/Jerry-Master/Hovernet-plus-Graphs/resolve/main/{gnn_dataset}/gnn/{gnn_model}/'
        dirname = os.path.join(APP_DIR, 'weights', gnn_dataset, gnn_model)
        download_folder(url_folder, url_files, dirname)


def create_input_dir(input_image: np.ndarray):
    input_dir = os.path.join(APP_DIR, 'tmp')
    if os.path.exists(input_dir):
        shutil.rmtree(input_dir)
    os.makedirs(input_dir)
    cv2.imwrite(os.path.join(input_dir, 'input_image.png'), input_image[:, :, ::-1])


def run_hovernet(hov_dataset: str, hov_model: str, num_classes: int):
    newargs = {
        'nr_types': num_classes,
        'type_info_path': os.path.join(APP_DIR, 'weights', hov_dataset, 'type_info.json'),
        'gpu': '0',
        'nr_inference_workers': '0',
        'model_path': os.path.join(APP_DIR, 'weights', hov_dataset, hov_model + '.tar'),
        'batch_size': '10',
        'shape': hov_model[:-2] if 'FT' in hov_model else hov_model,
        'nr_post_proc_workers': '0',
        'model_mode': 'original',
        'help': False
    }
    newsubargs = {
        'input_dir': os.path.join(APP_DIR, 'tmp'),
        'output_dir': os.path.join(APP_DIR, 'tmp_hov'),
        'draw_dot': False,
        'save_qupath': False,
        'save_raw_map': False,
        'mem_usage': '0.2'
    }
    hov_infer(newargs, newsubargs, 'tile')


def run_posthov(num_classes: int, logger: Logger):
    newargs = Namespace(
        json_dir = os.path.join(APP_DIR, 'tmp_hov', 'json'),
        gson_dir = os.path.join(APP_DIR, 'gson_hov'),
        num_classes = num_classes
    )
    hovernet2geojson(newargs)
    newargs = Namespace(
        gson_dir = os.path.join(APP_DIR, 'gson_hov'),
        png_dir = os.path.join(APP_DIR, 'png_hov'),
        csv_dir = os.path.join(APP_DIR, 'csv_hov'),
        num_classes = num_classes
    )
    geojson2pngcsv(newargs)
    create_dir(os.path.join(APP_DIR, 'graphs'))
    newargs = Namespace(
        png_dir = os.path.join(APP_DIR, 'png_hov'),
        orig_dir = os.path.join(APP_DIR, 'tmp'),
        output_path = os.path.join(APP_DIR, 'graphs', 'raw'),
        num_workers = 0
    )
    png2graph(newargs)
    newargs = Namespace(
        json_dir = os.path.join(APP_DIR, 'tmp_hov', 'json'),
        graph_dir = os.path.join(APP_DIR, 'graphs', 'raw'),
        output_dir = os.path.join(APP_DIR, 'graphs', 'hovpreds'),
        num_classes = num_classes
    )
    join_hovprob_graph(newargs, logger)


def run_graphs(gnn_dataset: str, gnn_model: str, num_classes: int):
    disable_prior = 'no-prior' in gnn_model or 'void' in gnn_model
    disable_morph_feats = 'no-morph' in gnn_model or 'void' in gnn_model
    model_name = os.listdir(os.path.join(APP_DIR, 'weights', gnn_dataset, gnn_model))[0][:-4]
    newargs = Namespace(
        node_dir = os.path.join(APP_DIR, 'graphs', 'hovpreds'),
        output_dir = os.path.join(APP_DIR, 'gnn_preds'),
        weights = os.path.join(APP_DIR, 'weights', gnn_dataset, gnn_model, model_name + '.pth'),
        conf = os.path.join(APP_DIR, 'weights', gnn_dataset, gnn_model, model_name + '.json'),
        normalizers = os.path.join(APP_DIR, 'weights', gnn_dataset, gnn_model, model_name + '.pkl'),
        num_classes = num_classes,
        disable_prior = disable_prior,
        disable_morph_feats = disable_morph_feats,
    )
    infer_gnn(newargs)


def run_postgraphs(num_classes: int):
    newargs = Namespace(
        graph_dir = os.path.join(APP_DIR, 'gnn_preds'),
        centroids_dir = os.path.join(APP_DIR, 'centroids'),
        num_classes = num_classes,
    )
    graph2centroids(newargs)
    newargs = Namespace(
        centroids_dir = os.path.join(APP_DIR, 'centroids'),
        png_dir = os.path.join(APP_DIR, 'png_hov'),
        csv_dir = os.path.join(APP_DIR, 'csv_gnn'),
    )
    centroidspng2csv(newargs)
    newargs = Namespace(
        png_dir = os.path.join(APP_DIR, 'png_hov'),
        csv_dir = os.path.join(APP_DIR, 'csv_gnn'),
        gson_dir = os.path.join(APP_DIR, 'gson_gnn'),
        num_classes = num_classes
    )
    pngcsv2geojson(newargs)


def create_logger():
    logger = logging.getLogger('gradio')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def process_image(input_image: np.ndarray, hov_dataset: str, hov_model: str, gnn_dataset: str, gnn_model: str):
    logger = create_logger()
    download_models_if_needed(hov_dataset, hov_model, gnn_dataset, gnn_model)
    with open(os.path.join(APP_DIR, 'weights', hov_dataset, 'type_info.json'), 'r') as f:
        type_info = json.load(f)
        num_classes = len(type_info.keys())
    create_input_dir(input_image)
    run_hovernet(hov_dataset, hov_model, num_classes)
    run_posthov(num_classes, logger)
    run_graphs(gnn_dataset, gnn_model)
    run_postgraphs()
    return input_image


def create_ui():
    image_input = gr.Image(shape=(1024, 1024))
    hov_dataset = gr.Dropdown(choices=[
        'consep', 'monusac', 'breast', 'lung'
    ], label='Select dataset in which hovernet was trained')
    hov_model = gr.Dropdown(choices=[
        '270', '270FT', '518', '518FT'
    ], label="Select Hovernet model.")
    gnn_dataset = gr.Dropdown(choices=[
        'consep', 'monusac', 'breast', 'lung'
    ], label='Select dataset in which gnns were trained')
    gnn_model = gr.Dropdown(choices=[
        'gcn-full', 'gat-full',
        'gcn-no-morph', 'gcn-no-prior', 'gcn-void',
        'None'
        ], label='Select GNN model')
    ui = gr.Interface(
        fn=process_image,
        inputs=[image_input, hov_dataset, hov_model, gnn_dataset, gnn_model],
        outputs='image',
        title="CNN+GNN Demo",
        description="Upload an image to see the output of the algorithm.",
        examples=[[os.path.join(APP_DIR, 'examples', x), 'breast', '518FT', 'breast', 'gcn-full'] for x in os.listdir(os.path.join(APP_DIR, 'examples'))]
    )
    return ui


def main():
    ui = create_ui()
    ui.launch()


if __name__ == '__main__':
    main()
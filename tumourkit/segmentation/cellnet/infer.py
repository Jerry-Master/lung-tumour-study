"""
Script to run merge.

Copyright (C) 2023  Jose Pérez Cano

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Contact information: joseperez2000@hotmail.es
"""
import os
import cv2
import argparse
from parameters import *
from soft_utils import *
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed
import skimage
import torch
import albumentations as albu

import soft_segmentation_models_pytorch as smp
from skimage.segmentation import find_boundaries
from scipy import ndimage as nd

import torch.nn as nn
import sys
PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PKG_DIR)

from utils.preprocessing import *

# Device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

segment_model = None
count_model = None
def load_models(segment_path, count_path):
    global segment_model
    segment_model = torch.load(segment_path, map_location=torch.device('cpu'))
    segment_model.segmentation_head._modules['2'] = nn.Identity()
    global count_model
    count_model = torch.load(count_path, map_location=torch.device('cpu'))

# Parameters
th = 0.15   #note: threshold for the count model, to declare a maximum to be a cell
sigma = 6   #note: sigma for the count model (unused in the current version)
min_prob_threshold = 1

# Compile
ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


###################################################################################################
############################################ FUNCTIONS ############################################
###################################################################################################
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def statistics(cells_mask, centroids_mask):
    """
    Compute statistics of the image: number of cells, number of centers, number of centers per cell...

    Args:
        - class_mask (np.array): classes of each pixel that belongs to a cell. #FF: cells mask!
                                 Size: (512, 512)
        - centroids_mask (np.array): centroids of the cells.
                                     Size: (512, 512)

    Returns:
        - labels_seg (np.array): labels of the connedted components of the segment model.
                                 Size: (512, 512)
        - labels_count (np.array): labels of the centers of the cells predicted by the count model.
                                   Size: (512, 512)
        - centers_in_cell (dict): correspondence between the segment and count labels, relating for
                                  each connected component, the centers situated at each cell.
                                  {cell1: [center1, center2, ...], cell2: [center3, center4, ...], ...}
        - centroids (np.array): centroids of the predicted cell nuclei (binary mask).                     #FF: potser en una versió anterior(?)
                                Size: (512, 512)
    """   

    # Get components from Segmentation
    _, labels_seg = cv2.connectedComponents(cells_mask, 4, ltype=cv2.CV_32S)

    # Get labels and centroids from Count
    _, labels_count = cv2.connectedComponents(centroids_mask, 4, ltype=cv2.CV_32S)
        
    # Correspondence between label seg (cell) and label count (center)
    centers_in_cell = {}                                                                # {cell: [centers]}
    rows, cols = np.where(centroids_mask > 0)
    for i, j in zip(rows, cols):                      # for i, j in zip(rows, cols):
        val_count = labels_count[i,j]
        val_seg = labels_seg[i,j]
        if not val_seg in centers_in_cell:
            centers_in_cell[val_seg] = []
        centers_in_cell[val_seg].append(val_count)
    
    return labels_seg, labels_count, centers_in_cell #last one added

def median_class(pred_seg, labels):
    ''' pred_seg: the prediction of the segmentation model
        labels: the labels of the zones after watershed
    return: the median of the zone at each predicted cell'''
    
    # Median class per label (list with the class for each label (label = position in teh vector))
    new_pred = np.zeros(pred_seg.shape)
    median = nd.median(pred_seg, labels, range(1,labels.max()+1))
    median = np.array([0] + list(median))
    
    # Mask with all the pixels of a connected component with the median class value
    new_pred = median[labels]

    return new_pred.astype(int)    

def get_watershed(cells_mask, prob_mask, centroids_mask, MergeMode):
    """
    Computes the watershed method to split cells with more than one centers.

    Args:
        - class_mask (np.array): classes of each pixel that belongs to a cell.
                                 Size: (512, 512)
        - centroids_mask (np.array): centroids of the cells.
                                     Size: (512, 512)
        - prob_mask (np.array): probabilities of the cells for each class.
                                Size: (C, 512, 512)

    Returns:
        - labels (np.array): Labels of each cell, that is, connected component.
                             Size: (512, 512)
        - cells_prob (np.array): Array with the probabilities of the "final" cells for each class.
                                 Size: (C, 512, 512)
        - seg_labels (np.array): labels of the connedted components of the segment model.
                                 Size: (512, 512)
        - count_labels (np.array): labels of the centers of the cells predicted by the count model.
                                   Size: (512, 512)
        - centers_in_cell (dict): correspondence between the segment and count labels, relating for
                                  each connected component, the centers situated at each cell.
                                  {cell1: [center1, center2, ...], cell2: [center3, center4, ...], ...}
        - centroids (np.array): centroids of the predicted cell nuclei (binary mask).
                                Size: (512, 512)
    """
    
    # Compute statistics --> labels segment and count, centers in each cell and centroids of the count prediction
    seg_labels, count_labels, centers_in_cell = statistics(cells_mask, centroids_mask)

    # print(len(np.unique(count_labels)))

    if MergeMode == 'B': #include centers without cell
        try:
            centers_without_cell = centers_in_cell[0] #centers assigned to cell 0 (no cell)
        except KeyError:
            centers_without_cell = []
        # print('There are: ' + str(len(centers_without_cell)) + ' centers without cell in the image.')
        count_labels_without_cell = np.isin(count_labels,centers_without_cell).astype(int)*255
        cells_mask_without_cell = ((ndi.gaussian_filter(count_labels_without_cell, sigma=2)*255)>64).astype(int)

        cells_mask = cells_mask + cells_mask_without_cell

    elif MergeMode == 'C': #include cells without a center on them
        cells_without_center = [i for i in range(0,len(centers_in_cell.keys())) if i not in centers_in_cell.keys()]
        # print('There are ' + str(len(cells_without_center)) + ' cells without center in the image')

        new_count_label = np.max(count_labels)+1
        for idx in range(len(cells_without_center)):
            tmp_cell_mask = (seg_labels == cells_without_center[idx])
            tmp_cell_mask = cells_mask*tmp_cell_mask
            new_center = ndi.center_of_mass(tmp_cell_mask)
            count_labels[int(new_center[0]),int(new_center[1])] = new_count_label
            new_count_label += 1

    elif MergeMode == 'D':
        try:
            centers_without_cell = centers_in_cell[0] #centers assigned to cell 0 (no cell)
        except KeyError:
            centers_without_cell = []
        # print('There are: ' + str(len(centers_without_cell)) + ' centers without cell in the image.')
        count_labels_without_cell = np.isin(count_labels,centers_without_cell).astype(int)*255
        cells_mask_without_cell = ((ndi.gaussian_filter(count_labels_without_cell, sigma=2)*255)>64).astype(int)
        cells_mask = cells_mask + cells_mask_without_cell

        cells_without_center = [i for i in range(0,len(centers_in_cell.keys())) if i not in centers_in_cell.keys()]
        # print('There are ' + str(len(cells_without_center)) + ' cells without center in the image')
        new_count_label = np.max(count_labels)+1
        for idx in range(len(cells_without_center)):
            tmp_cell_mask = (seg_labels == cells_without_center[idx])
            tmp_cell_mask = cells_mask*tmp_cell_mask
            new_center = ndi.center_of_mass(tmp_cell_mask)
            try:
                count_labels[int(new_center[0]),int(new_center[1])] = new_count_label
            except ValueError:
                continue
            new_count_label += 1

    # Watershed using as markers the centroids
    binary = cells_mask.copy()
    distance = ndi.distance_transform_edt(binary)
    labels = watershed(-distance, markers=count_labels, mask=binary, compactness=1)         # (512, 512)

    # Labels splitted by the outer contours of each connected component
    contour = np.invert(find_boundaries(labels, mode='outer', background=0))
    labels = labels*contour                                                                 # (512, 512)

    # Cells probabilities after watershed
    prob_mask = prob_mask*cells_mask                                                             # (C, 512, 512)
    class_mask = np.uint8((np.argmax(prob_mask, 0) + 1)*cells_mask)
    class_mask = median_class(class_mask, labels)
    
    return seg_labels, count_labels, centers_in_cell, cells_mask, prob_mask, class_mask, labels


def preprocess(image):
    # Preprocess the image
    image = cv2.resize(image,(512,512))
    preprocess = get_preprocessing(preprocessing_fn)
    sample = preprocess(image=image)
    image = sample['image']

    return image

# Block A
def segment(x_tensor):
    """
    Segment and classify the pixels into one of the C classes.

    Args:
        - x_tensor (tensor): image to segment.
                             Size: (3, 512, 512)

    Returns:
        - pred_class (np.array): classes of each pixel that belongs to a cell.
                                 Size: (512, 512)
        - probabilities (np.array): probabilities (obtained from the softmax) of the cells for each class.
                                    Size: (C, 512, 512)
    """
    # Predict (without sigmoid layer)
    pr_maskA = segment_model.to(device).predict(x_tensor)
    pr_maskA = pr_maskA.squeeze().cpu()                                        # (C, 512, 512)

    probabilities_uncurated = pr_maskA.numpy().copy()

    # Cells mask (applying the sigmoid)
    tmp = np.amax(torch.sigmoid(pr_maskA).numpy(), 0).round()                  # (512, 512)

    # Cells probabilities
    pr_maskA = pr_maskA.numpy()
    probabilities = pr_maskA*tmp                                               # (C, 512, 512)

    probabilities = torch.nn.functional.softmax(torch.from_numpy(probabilities), dim=0).numpy()

    return np.uint8(tmp), probabilities, probabilities_uncurated


# Block B
def count(x_tensor):
    """
    Find the center of each individual cell.

    Args:
        - x_tensor (tensor): image to segment.
                             Size: (3, 512, 512)

    Returns:
        - maxims_mask (np.array): centroids of the cells.
                                  Size: (512, 512)
    """

    # Predict
    pr_maskB = count_model.to(device).predict(x_tensor)
    pr_maskB = pr_maskB.squeeze().cpu().numpy()                                                         # (512, 512)

    # Centroids of the predicted gaussians
    maxims_mask, lst_pr = smp.utils.new_metrics.find_local_maxima(pr_maskB, th)                           # (512, 512) i num. de centres
    
    return np.uint8(maxims_mask)

# Merge   
def merge(cells_mask, prob_mask, centroids_mask, init_label, MergeMode):
    """
    Combine the results of both blocks (the segmentation of the cells and the center detection)
    to obtain each cell as a connected component.

    Args:
        - class_mask (np.array): classes of each pixel that belongs to a cell. #FF: cells_mask perhaps?
                                 Size: (512, 512)
        - centroids_mask (np.array): centroids of the cells.
                                     Size: (512, 512)
        - prob_mask (np.array): probabilities of the cells for each class.
                                Size: (C, 512, 512)
        - init_label (int): first label of this tile (used so that, when combining the information
                            of the entire region, the labels are unique).
        
    Returns:
        - 
    """
    
    # Obtain labels after applying watershed
    seg_labels, count_labels, centers_in_cell, cells_mask, prob_mask, class_mask, labels = get_watershed(cells_mask, prob_mask, centroids_mask, MergeMode)
    return cells_mask, class_mask, prob_mask

    
def inference(image, init_label, MergeMode):
    
    x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
    
    # Block A --> segment
    cells_mask, prob_mask, prob_uncur = segment(x_tensor)

    kernel = np.ones((2,2), np.uint8)
    cells_mask = cv2.morphologyEx(cells_mask, cv2.MORPH_OPEN, kernel)

    # Block B --> count
    centroids_mask = count(x_tensor)
    # Merge
    cells_mask, class_mask, prob_mask = merge(cells_mask, prob_mask, centroids_mask, init_label, MergeMode)
    
    return cells_mask, class_mask, prob_mask

def process_image(image, init_label=1,MergeMode = 'A'):

    image = preprocess(image)
    cells_mask, class_mask, prob_mask = inference(image, init_label, MergeMode)

    return cells_mask, class_mask, prob_mask

def save_results(mask, probab, output_path, name, num_cls):
    probab = np.moveaxis(probab, 0, -1)                         # (C, 512, 512) --> (512, 512, C)
    # Connected components
    components, num_cells = skimage.morphology.label(mask, return_num=True, connectivity=1)
    # Class of each cell
    values_cells = np.zeros((num_cells, num_cls))
    for i in range(num_cls):
        values_cells[:,i] = nd.mean(probab[:,:,i], components, range(1,num_cells+1))
    values_cells = torch.nn.functional.softmax(torch.from_numpy(values_cells), dim=1).numpy()
    class_cells = np.argmax(values_cells, axis=1) + 1
    # Compute csv
    k = np.arange(1,num_cells+1)
    cells = np.concatenate((k[:,None], class_cells[:,None]), axis=1)
    cells = pd.DataFrame(cells)
    # Save image and csv with (id, class)
    save_pngcsv(components, cells, output_path+'png/', output_path+'csv/', name)


parser = argparse.ArgumentParser()
parser.add_argument('--img-dir', type=str, required=True,
                    help='Directory of all the images to process.')
parser.add_argument('--output-path', type=str, required=True,
                    help='Directory to save all processed images.')
parser.add_argument('--count-path', type=str, required=True,
                    help='Path to count model.')
parser.add_argument('--segment-path', type=str, required=True,
                    help='Path to segment model.')
#note: execution of the images 
if __name__=='__main__':
    args = parser.parse_args()
    IMG_DIR = parse_path(args.img_dir)
    OUTPUT_PATH = parse_path(args.output_path)
    create_dir(OUTPUT_PATH)
    create_dir(OUTPUT_PATH+'png/')
    create_dir(OUTPUT_PATH+'csv/')
    print('Loading models...')
    COUNT_PATH = args.count_path
    SEGMENT_PATH = args.segment_path
    load_models(SEGMENT_PATH, COUNT_PATH)
    print('Done!')

    names = get_names(IMG_DIR, '.png')
    for k, name in enumerate(names):
        print('Progress: {:2d}/{}'.format(k+1, len(names)), end="\r")
        image = cv2.imread(IMG_DIR + name + '.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cells_mask, class_mask, prob_mask = process_image(image,MergeMode='D')
        save_results(class_mask, prob_mask, OUTPUT_PATH, name, 2)

"""

Evaluating script for cell predictions. 

Input format
------------
Centroids in a csv with columns X,Y and class. Both for prediction and GT.

Output
------
Weighted F1-score between the prediction and the GT at cell-level.

"""
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import argparse
import time
from utils.preprocessing import *

parser = argparse.ArgumentParser()
parser.add_argument('--names', type=str, required=True,
                    help='Path to txt file with names.')
parser.add_argument('--gt_path', type=str, required=True,
                    help='Path to GT files.')
parser.add_argument('--pred_path', type=str, required=True,
                    help='Path to prediction files.')
parser.add_argument('--save_name', type=str, required=True,
                    help='Name to save the result, without file type.')

def read_centroids(name, path):
    """
    Format of the csv should be columns: X, Y, class
    """
    centroid_csv = pd.read_csv(path + name + '.centroids.csv')
    centroid_csv = centroid_csv.drop(centroid_csv[centroid_csv['class']==-1].index)
    return centroid_csv.to_numpy(dtype=int)

def get_centroid_by_id(img, idx):
    """
    img contains a different id value per component at each pixel
    """
    X, Y = np.where(img == idx)
    if len(X) == 0 or len(Y) == 0:
        return -1, -1
    return X.mean(), Y.mean()

def extract_centroids(img, csv):
    """
    Output format: list of (x,y,class) tuples
    """
    centroids = []
    for i, row in csv.iterrows():
        x, y = get_centroid_by_id(img, row.id)
        if x == -1:
            continue
        centroids.append((x,y,row.label))
    return centroids

def generate_tree(centroids):
    """
    Input format: list of (x,y,class) tuples
    """
    centroids_ = np.array(list(map(lambda x: (x[0], x[1]), centroids)))
    return KDTree(centroids_)

def find_nearest(a, B):
    """
    a: (x,y,class) tuple
    B: KDTree to search for nearest point
    """
    x, y = a[0], a[1]
    dist, idx = B.query([x,y], k=1)
    return idx

def get_confusion_matrix(gt_centroids, pred_centroids):
    """
    Each centroid is represented by a 3-tuple with (X, Y, class).
    Class is 0=non-tumour, 1=tumour.
    """
    t0 = time.time()
    if len(gt_centroids) == 0:
        return None

    gt_tree = generate_tree(gt_centroids[:,:2])
    pred_tree = generate_tree(pred_centroids[:,:2])
    t1 = time.time()
    # print('Time generating KDTree(secs):', t1-t0)
    t0 = time.time()
    M = np.zeros((2,2)) 
    for point_id, point in enumerate(gt_centroids):
        closest_id = find_nearest(point[:2], pred_tree)
        closest = pred_centroids[closest_id]
        if closest[2] != -1 and point[2] != -1 and point_id == find_nearest(closest[:2], gt_tree):
            M[int(point[2]-1)][int(closest[2]-1)] += 1
    t1 = time.time()
    # print('Time querying KDTree(secs):', t1-t0)
    t0 = time.time()
    return M

def get_weighted_F1_score(M):
    """
    Computes weighted F1 score from confusion matrix
    """
    if M is None:
        return -1, -1
    eps=1e-7

    TP = np.diag(M)
    M_ = M - np.diag(TP)
    FP = np.sum(M_, axis=0)
    FN = np.sum(M_, axis=1)
    support = np.sum(M, axis=1)

    prec = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * prec * recall / (prec + recall)

    total = support[1:].sum()
    wf1 = np.sum(f1[1:] * support[1:] / total)

    return f1, wf1

def compute_percentage(arr):
    """
    arr is an array of integers representing classes
    It returns the the percentage of class 2: #2 / (#1 + #2)
    """
    n_one = np.sum(arr==1)
    n_two = np.sum(arr==2)
    return n_two / (n_one + n_two)

def get_percentages(gt_centroids, pred_centroids):
    gt_labels = gt_centroids[:,2]
    pred_labels = pred_centroids[:,2]
    gt_per = compute_percentage(gt_labels)
    pred_per = compute_percentage(pred_labels)
    return gt_per, pred_per

def save_score(scores, name, percentages, save_path):
    """
    Appends the F1 score and weighted f1 score in scores in the file results.txt.
    The name of the labels file should be given.
    """
    f1, wf1 = scores
    gt_per, pred_per = percentages
    with open(save_path + '.txt', 'a') as f:
        print(name, file=f)
        print('    F1 score: {}\n    Weighted F1 score: {:.3f}\n'.format(f1, wf1), file=f, end='')
        print('    GT percentage: {:.3f}\n    Pred percentage: {:.3f}\n'.format(gt_per, pred_per), file=f, end='')
        print('    Error: {:.3f}\n'.format(abs(gt_per - pred_per)), file=f)

def save_csv(metrics, save_path):
    metrics_df = pd.DataFrame(metrics, columns=['name', 'F1_1', 'F1_2', 'WF1'])
    metrics_df.to_csv(save_path + '.csv', index=False)

if __name__ == '__main__':
    args = parser.parse_args()
    names = read_names(args.names)
    metrics = []
    for k, name in enumerate(names):
        print('Progress: {:2d}/{}'.format(k+1, len(names)), end="\r")
        gt_centroids = read_centroids(name, args.gt_path)
        pred_centroids = read_centroids(name, args.pred_path)
        confusion_matrix = get_confusion_matrix(gt_centroids, pred_centroids)
        scores = get_weighted_F1_score(confusion_matrix)
        percentages = get_percentages(gt_centroids, pred_centroids)
        save_score(scores, name, percentages, args.save_name)
        metrics.append([name, *scores[0], scores[1]])
    save_csv(metrics, args.save_name)


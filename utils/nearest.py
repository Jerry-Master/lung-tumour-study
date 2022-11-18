from scipy.spatial import KDTree
import numpy as np
from typing import List, Tuple

def generate_tree(centroids: List[Tuple[int,int,int]]) -> KDTree:
    """
    Input format: list of (x,y,class) tuples.
    """
    centroids_ = np.array(list(map(lambda x: (x[0], x[1]), centroids)))
    return KDTree(centroids_)

def find_nearest(a: Tuple[int,int,int], B: KDTree) -> int:
    """
    a: (x,y,class) tuple.
    B: KDTree to search for nearest point.
    """
    x, y = a[0], a[1]
    dist, idx = B.query([x,y], k=1)
    return idx

def find_nearest_dist_idx(a: Tuple[int,int,int], B: KDTree) -> Tuple[float, int]:
    """
    a: (x,y,class) tuple.
    B: KDTree to search for nearest point.
    """
    x, y = a[0], a[1]
    dist, idx = B.query([x,y], k=1)
    return dist, idx
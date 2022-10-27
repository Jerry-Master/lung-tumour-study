from typing import Callable
from scipy.spatial import KDTree
import queue

Point = tuple[float,float]
Contour = list[Point]
Cell = tuple[int, int, Contour] # id, class

"""NOT DOCUMENTED"""
def get_N_closest_pairs_dists(a: Contour, b: Contour, N: int) -> list[float]:
    if len(a) > len(b):
        tree = KDTree(a)
        query_set = b
    else:
        tree = KDTree(b)
        query_set = a
    cpairs = queue.PriorityQueue(maxsize=N)
    for point in query_set:
        dist, idx = tree.query(point, k=N)
        for i in range(N):
            if cpairs.full():
                top_pair = cpairs.get()
                if dist[i] >= -top_pair: # Negative distances to get the highest one
                    cpairs.put(top_pair)
                else:
                    cpairs.put(-dist[i])
            else:
                cpairs.put(-dist[i])
    cpairs_list = []
    while not cpairs.empty():
        cpairs_list.append(-cpairs.get()) # Undo negative distances
    return cpairs_list

"""NOT DOCUMENTED"""
def create_comparator(threshold: float, num_frontier: int) -> Callable[[Contour,Contour], bool]:
    def is_equal(a: Contour, b: Contour) -> bool:
        cpairs_dists = get_N_closest_pairs_dists(a, b, num_frontier)
        for dist in cpairs_dists:
            if dist > threshold:
                return False
        return True
    return is_equal

"""NOT IMPLEMENTED"""
def merge_cells(a: Cell, b: Cell) -> Cell:
    return a
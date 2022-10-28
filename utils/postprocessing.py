from typing import Callable
from scipy.spatial import KDTree

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
    cpairs = []
    for point in query_set:
        dist, idx = tree.query(point, k=N)
        cpairs.extend(dist)
        cpairs.sort()
        cpairs = cpairs[:N]
    return cpairs

"""NOT DOCUMENTED"""
def create_comparator(threshold: float, num_frontier: int) -> Callable[[Contour,Contour], bool]:
    def is_equal(a: Contour, b: Contour) -> bool:
        cpairs_dists = get_N_closest_pairs_dists(a, b, num_frontier)
        for dist in cpairs_dists:
            if dist > threshold:
                return False
        return True
    return is_equal

"""NOT DOCUMENTED"""
def get_N_closest_pairs_idx(a: Contour, b: Contour, N: int) -> tuple[list[int],list[int]]:
    is_a_tree = len(a) > len(b)
    if is_a_tree:
        tree = KDTree(a)
        query_set = b
    else:
        tree = KDTree(b)
        query_set = a
    cpairs = []
    for idx2, point in enumerate(query_set):
        dist, idx = tree.query(point, k=N)
        cpairs.extend(zip(dist, idx, [idx2 for _ in range(len(idx))]))
        cpairs.sort()
        cpairs = cpairs[:N]
    dists, idxa, idxb = list(zip(*cpairs))
    if not is_a_tree:
        idxa, idxb = idxb, idxa
    return list(idxa), list(idxb)

"""NOT DOCUMENTED"""
def get_greatest_connected_component(idx: list[int], max_idx: int) -> tuple[int,int]:
    idx.sort()
    l_final, l_aux, r_final, r_aux = 0, 0, 0, 0
    found_comp = False
    for k in range(1,len(idx)):
        if idx[k-1]+1 != idx[k]:
            found_comp = True
            # End of component
            r_aux = k-1
            if  r_aux - l_aux > r_final - l_final:
                # Save greatest component
                r_final, l_final = r_aux, l_aux
            # Restart left auxiliary index
            l_aux = k
    if not found_comp:
        return idx[0], idx[-1]
    if idx[0] == 0 and idx[-1] == max_idx:
        # Circular case
        r_aux = 1
        while r_aux < len(idx) and idx[r_aux-1] + 1 == idx[r_aux]:
            r_aux += 1
        r_aux -= 1
        l_aux = len(idx)-1
        while l_aux >= 0 and idx[l_aux-1] + 1 == idx[l_aux]:
            l_aux -= 1
        if (r_aux-0) + (len(idx)-l_aux) > r_final - l_final:
            r_final, l_final = r_aux, l_aux
    return idx[l_final], idx[r_final]

"""NOT DOCUMENTED"""
def remove_idx(a: Cell, a_idx: list[int]) -> Cell:
    left, right = get_greatest_connected_component(a_idx, len(a[2])-1)
    res = []
    if left > right: # Contour is in third position
        res.extend(a[2][right+1:left])
    elif left <= right:
        res.extend(a[2][right+1:])
        res.extend(a[2][0:left]) 
    return (a[0], a[1], res)

"""NOT IMPLEMENTED"""
def check_order(a: Cell, b: Cell) -> Cell:
    # Checks if joining a and b creates a cross in the middle.
    return True

"""NOT DOCUMENTED"""
def join(a: Cell, b: Cell) -> Cell:
    good_order = check_order(a, b)
    if not good_order:
        b[2].reverse()
    a[2].extend(b[2])
    return a

"""NOT DOCUMENTED"""
def merge_cells(a: Cell, b: Cell) -> Cell:
    a_idx, b_idx = get_N_closest_pairs_idx(a[2], b[2], 30)
    a = remove_idx(a, a_idx)
    b = remove_idx(b, b_idx)
    c = join(a, b)
    return c
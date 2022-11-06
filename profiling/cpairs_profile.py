import sys
import os
import numpy as np
import cProfile
import io
import pstats

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TEST_DIR)
sys.path.append(PKG_DIR)

from utils.postprocessing import get_N_closest_pairs_dists


THRESHOLD = 0.001
tests = [
    (
        np.hstack((np.linspace(0,30,100).reshape(-1,1), np.ones((100,1)))),
        np.hstack((np.linspace(0,30,100).reshape(-1,1), np.ones((100,1))*2)), 
        np.ones((10,))
    ),
    (
        np.hstack((np.linspace(0,300,101).reshape(-1,1), np.ones((101,1))))+np.array([[1,0]]),
        np.hstack((np.linspace(0,300,101).reshape(-1,1), np.ones((101,1))*2)), 
        np.ones((20,)) * np.sqrt(2)
    ),
    (
        np.hstack((np.linspace(0,30,11).reshape(-1,1)**2, np.ones((11,1)))),
        np.hstack((np.linspace(0,30,11).reshape(-1,1), np.ones((11,1))*2)), 
        np.array([1, 1, np.sqrt(3**2+1), np.sqrt(3**2+1), np.sqrt(3**2+1), np.sqrt(6**2+1), np.sqrt(6**2+1)])
    ),
    (
        np.hstack((np.linspace(0,300000,100001).reshape(-1,1), np.ones((100001,1))))+np.array([[1,0]]),
        np.hstack((np.linspace(0,300,101).reshape(-1,1), np.ones((101,1))*2)), 
        np.ones((20,)) * np.sqrt(2)
    )
]
def exec_conversion(A,B, result):
    pred = get_N_closest_pairs_dists(A, B, len(result))
    pred = sorted(pred)
    if len(pred) != len(result):
        return False
    for i in range(len(pred)):
        if abs(pred[i] - result[i]) > THRESHOLD:
            return False
    return True

# cProfile.runctx('exec_conversion(*tests[0])', None, locals(), filename='prof1.txt')
# cProfile.runctx('exec_conversion(*tests[1])', None, locals(), filename='prof2.txt')
# cProfile.runctx('exec_conversion(*tests[2])', None, locals(), filename='prof3.txt')

for i in range(len(tests)):
    pr = cProfile.Profile()
    pr.enable()

    my_result = exec_conversion(*tests[i])

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats()

    with open('cpairs/cpairs_profile' + str(i+1) + '.txt', 'w+') as f:
        f.write(s.getvalue())
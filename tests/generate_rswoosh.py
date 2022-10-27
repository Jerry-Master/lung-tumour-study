import numpy as np
import os
import json
import random

TEST_DIR = os.path.dirname(os.path.abspath(__file__))

def save(A, B, name):
    with open(TEST_DIR + '/rswoosh/' + name + '.input.json', 'w') as f:
        json.dump(A, f)
    with open(TEST_DIR + '/rswoosh/' + name + '.output.json', 'w') as f:
        json.dump(B, f)

def create_line(p, q):
    x = np.linspace(p[0], q[0], 100)[1:-1]
    y = np.linspace(p[1], q[1], 100)[1:-1]
    return np.hstack((x.reshape(-1,1), y.reshape(-1,1)))

def split_circle(radius, theta):
    tol = 1e-5
    left_angles = np.linspace(theta + tol, theta + np.pi, 100)
    left_points = np.hstack((np.cos(left_angles).reshape(-1,1), np.sin(left_angles).reshape(-1,1))) * radius
    right_angles = np.linspace(theta + np.pi + tol, theta + 2*np.pi, 100)
    right_points = np.hstack((np.cos(right_angles).reshape(-1,1), np.sin(right_angles).reshape(-1,1))) * radius
    merged = np.vstack((left_points, right_points)).copy()
    left_points = np.vstack((left_points, create_line(left_points[-1], left_points[0])))
    right_points = np.vstack((right_points, create_line(right_points[-1], right_points[0])))
    return left_points, right_points, merged

def translate(points, x, y):
    points = points.copy()
    points[:,0] += x
    points[:,1] += y
    return points.tolist()

def generate_cell_tuple(points, x, y):
    idx = 1e5 * x + y
    cls = 0
    return tuple((int(idx),int(cls),points))

if __name__=='__main__':
    """
    First test:
    Circles split in two with a line in between.
    """
    counter = 0
    for r in [10,20,30]:
        for theta in [0, np.pi / 8, np.pi / 4]:
            counter += 1
            left, right, merged = split_circle(r, theta)
            A, B = [], []
            for x in range(0,16*r,4*r):
                for y in range(0,16*r,4*r):
                    rand_number = random.randint(0,5)
                    if rand_number == 3:
                        A.append(generate_cell_tuple(translate(left, x, y), x / r, y / r))
                        A.append(generate_cell_tuple(translate(right, x, y), x / r, y / r + 1))
                        B.append(generate_cell_tuple(translate(merged, x, y), x / r, y / r))
                    elif rand_number == 1:
                        A.append(generate_cell_tuple(translate(left, x, y), x / r, y / r))
                        B.append(generate_cell_tuple(translate(left, x, y), x / r, y / r))
                    else:
                        A.append(generate_cell_tuple(translate(merged, x, y), x / r, y / r))
                        B.append(generate_cell_tuple(translate(merged, x, y), x / r, y / r))
            save(A,B, 'circles'+str(counter))
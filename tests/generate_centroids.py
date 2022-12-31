"""
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
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix

TEST_DIR = os.path.dirname(os.path.abspath(__file__))

def save(A, B, confusion, name):
    pd.DataFrame(A, columns=['X','Y','class']).to_csv(TEST_DIR + '/centroids/'+ name +'.A.csv', index=False)
    pd.DataFrame(B, columns=['X','Y','class']).to_csv(TEST_DIR + '/centroids/'+ name +'.B.csv', index=False)
    pd.DataFrame(confusion).to_csv(TEST_DIR + '/centroids/'+ name +'.result.csv', index=False, header=False)

def polygon(n, theta=0.01, radius=1):
    paramA = np.linspace(0, 2*np.pi, n+1)
    paramB = paramA + theta
    X, Y = radius*np.cos(paramA), radius*np.sin(paramA)
    A = np.vstack([X, Y, np.ones(n+1)]).transpose()
    X_, Y_ = radius*np.cos(paramB), radius*np.sin(paramB)
    B = np.vstack([X_, Y_, np.ones(n+1)]).transpose()
    return A[:-1], B[:-1]


if __name__=='__main__':
    A = np.random.uniform(low=0, high=1024, size=(100,2))
    A = np.hstack([A, np.ones((100,1))])
    B = A.copy()
    save(A, B, np.array([[100,0],[0,0]]), 'random1')
    B[:,2] = 2
    save(A, B, np.array([[0,100],[0,0]]), 'random2')
    B[:30,2] = 1
    save(A, B, np.array([[30,70],[0,0]]), 'random3')
    A[30:50,2] = 2
    save(A, B, np.array([[30,50],[0,20]]), 'random4')

    A, B = polygon(8)
    save(A, B, np.array([[8,0],[0,0]]), 'circle1')
    A, B = polygon(9)
    save(A, B, np.array([[9,0],[0,0]]), 'circle2')
    A, B = polygon(20)
    save(A, B, np.array([[20,0],[0,0]]), 'circle3')
    A[:10,2] = 2
    B[:10,2] = 2
    save(A, B, np.array([[10,0],[0,10]]), 'circle4')
    A[:5,2] = 1
    save(A, B, np.array([[10,5],[0,5]]), 'circle5')
    B[15:,2] = 2
    save(A, B, np.array([[5,10],[0,5]]), 'circle6')

    A, B = polygon(100, theta=0)
    A[:,:2] *= 8
    B[:,:2] *= 10
    save(A, B, np.array([[100,0],[0,0]]), 'circle7')
    A, B = polygon(50, theta=0)
    A[:,:2] *= 2
    B[:,:2] *= 5
    save(A, B, np.array([[50,0],[0,0]]), 'circle8')
    B[:,2] = 2
    save(A, B, np.array([[0,50],[0,0]]), 'circle9')

    for k in range(10):
        A[:,2] = np.random.randint(1,3,50)
        B[:,2] = np.random.randint(1,3,50)
        conf = confusion_matrix(A[:,2], B[:,2])
        save(A, B, conf, 'random_labels'+str(k+1))

    A, B = polygon(179, theta=2*np.pi/360, radius=20)
    for k in range(10):
        A[:,2] = np.random.randint(1,3,179)
        B[:,2] = np.random.randint(1,3,179)
        conf = confusion_matrix(A[:,2], B[:,2])
        save(A, B, conf, 'random_labels_rotated'+str(k+1))

    N = 3000
    A, B = polygon(N-1, theta=np.pi/(N), radius=200)
    for k in range(30):
        A[:,2] = np.random.randint(1,3,N-1)
        B[:,2] = np.random.randint(1,3,N-1)
        conf = confusion_matrix(A[:,2], B[:,2])
        save(A, B, conf, 'random_labels_rotated_vis'+str(k+1))

    N = 3000
    A, B = polygon(N-1, theta=np.pi/(N), radius=200)
    M = 300
    B = np.vstack([B, np.random.randint(-5, 5, size=(M,3))])
    for k in range(30):
        A[:,2] = np.random.randint(1,3,N-1)
        B[:N-1,2] = np.random.randint(1,3,N-1)
        B[N-1:,2] = np.random.randint(1,3,M)
        conf = confusion_matrix(A[:,2], B[:N-1,2])
        save(A, B, conf, 'extra_points'+str(k+1))

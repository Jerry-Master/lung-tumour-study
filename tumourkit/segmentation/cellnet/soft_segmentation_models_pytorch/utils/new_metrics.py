from os import listdir
from os.path import isfile, join
import json
from matplotlib import image
import scipy.misc
from matplotlib import pyplot as plt
import numpy as np
import math
from scipy.ndimage import gaussian_filter

import cv2
from skimage import color
from skimage.morphology import extrema
from skimage import exposure


def mask(file):
	i = 0
	success = 0
	with open(file) as json_file:
		data = json.load(json_file)
		jstr = json.dumps(data, indent=4)
	while i < len(file[:-5]):
		if file[i:i+2] == "w=":
			i = i + 2
			w = ''
			while file[i] != ',' and file[i] != '.':
				w = w + file[i]
				i = i + 1
			success = success + 0.5
		if file[i:i+2] == "h=":
			i = i + 2
			h = ''
			while file[i] != ')' and file[i] != '.':
				h = h + file[i]
				i = i + 1
			success = success + 0.5
		i = i + 1
	if success == 1:
		w = int(w)
		h = int(h)
	else:
		try:
			img = image.imread(file[:-4] + 'png')
			w = img.shape[1]
			h = img.shape[0]
			success = 1
		except:
			print("Image not found: there has to be an image with the same name as the .json file. Here '" + file[:-4] + "png'")
	
	if success == 1:
				
		#CREATE THE EMPTY BINARY MASK
		binary = np.zeros((h, w))
		
		# GET REGIONS LIST
		regions = data['_via_img_metadata'][list(data['_via_img_metadata'].keys())[0]]['regions']
		

		[Y, X] = [h, w]
		for region in regions:
			ty = region['region_attributes']["type"]
			pix = 250
			if ty=="blue":
				pix = 300
			elif ty=="brown1":
				pix = 50
			elif ty=="brown2":
				pix = 100
			elif ty=="brown3":
				pix = 150
			elif ty=="brown":
				pix = 200
			elif ty=="other":
				pix = 250
			cx = region['shape_attributes']['cx']
			cy = region['shape_attributes']['cy']
			rx = region['shape_attributes']['rx']
			ry = region['shape_attributes']['ry']
			theta = region['shape_attributes']['theta']
			y2x = dict()
			for k in range(1000):
				phi = k/500*np.pi
				x = int(round(cx + rx*np.cos(phi)*np.cos(theta) - ry*np.sin(phi)*np.sin(theta)))
				if x<0:
					x = 0
				elif x>=X:
					x = X-1
				y = int(round(cy + rx*np.cos(phi)*np.sin(theta) + ry*np.sin(phi)*np.cos(theta)))
				if y<0:
					y = 0
				elif y>=Y:
					y = Y-1
				
				binary[y, x] = pix
				if y in y2x and x != y2x[y]:
					for xi in range(min(x, y2x[y]), max(x, y2x[y])):
						binary[y, xi] = pix
				else:
					y2x[y] = x


		return binary



def regions(file):
	with open(file) as json_file:
		data = json.load(json_file)
		jstr = json.dumps(data, indent=4)
	regions = data['_via_img_metadata'][list(data['_via_img_metadata'].keys())[0]]['regions']
	return regions

def find_nearest_point_binary(lst, lst1, i):
    # Find the element in lst that is nearest to the ith element of lst1, where both are lists of points (xc, yc)
    
    # Find the point from lst nearest to the i'th point in lst1
    cx = lst1[i][0]
    cy = lst1[i][1]
    mindist = 1500**2
    k = None
    for j in range(len(lst)):
        d = (cx - lst[j][0])**2 + (cy - lst[j][1])**2
        if d < mindist:
            k = j
            mindist = d
            if d==0:
                break
    K = k
    return K, mindist

def compare_binary(r1, r2):
    # Compare the list of points r1 with the list r2 which is considered to be the GT
    
    # Build the correspondences
    g1 = [(None, None)]*len(r1)
    d1 = [None]*len(r1)
    for i in range(len(r1)):
        g1[i], d1[i] = find_nearest_point_binary(r2, r1, i)
    g2 = [(None, None)]*len(r2)
    d2 = [None]*len(r2)
    for i in range(len(r2)):
        g2[i], d2[i] = find_nearest_point_binary(r1, r2, i)
    
    # Delete all the correspondences that are too far
    g11 = g1*1
    g22 = g2*1
    for i in range(len(r1)):
        if d1[i] > 25:
            g11[i] = -1
    for i in range(len(r2)):
        if d2[i] > 25:
            g22[i] = -1
    
    # Delete all the correspondences that are not 1 to 1
    b1 = [-1]*len(g2)
    b2 = [-1]*len(g1)
    for i in range(len(r1)):
        if g2[g1[i]] != i:
            g11[i] = -1
        else:
            if b1[g1[i]] == -1:
                b1[g1[i]] = i
            else:
                g11[i] = -1
                g11[b1[g1[i]]] = -1
    
    for i in range(len(r2)):
        if g1[g2[i]] != i:
            g22[i] = -1
        else:
            if b2[g2[i]] == -1:
                b2[g2[i]] = i
            else:
                g22[i] = -1
                g22[b2[g2[i]]] = -1
    
    # Count the true/false positives/negatives
    tp = 0
    fp = 0
    for i in range(len(r1)):
        if g11[i] == -1:
            fp += 1
        else:
            tp += 1
    fn = 0
    for i in range(len(r2)):
        if g22[i] == -1:
            fn += 1      
    # Compute precission, recall and F-score
    precission = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    if precission+recall <= 0:
        return (0,0,0)
        
    F = 2 * precission * recall / (precission + recall)
    return precission, recall, F

def find_nearest_point_class(lst, lst1, i):
    # Find the element in lst that is nearest to the ith element of lst1, where both are lists of labeled points ((xc, yc), class)
    
    # Find the point from lst nearest to the i'th point in lst1
    cx = lst1[i][0][0]
    cy = lst1[i][0][1]
    mindist = 1500**2
    k = None
    for j in range(len(lst)):
        d = (cx - lst[j][0][0])**2 + (cy - lst[j][0][1])**2
        if d < mindist:
            k = j
            mindist = d
            if d==0:
                break
    K = k
    return K, mindist

def compare_class(r1, r2):
    # Compare the list of points r1 with the list r2 which is considered to be the GT, where both are lists of labeled points ((xc, yc), class)
    
    # Build the correspondences
    g1 = [(None, None)]*len(r1)
    d1 = [None]*len(r1)
    for i in range(len(r1)):
        g1[i], d1[i] = find_nearest_point_class(r2, r1, i)
    g2 = [(None, None)]*len(r2)
    d2 = [None]*len(r2)
    for i in range(len(r2)):
        g2[i], d2[i] = find_nearest_point_class(r1, r2, i)
    
    # Delete the correspondences that are too far
    g11 = g1*1
    g22 = g2*1
    for i in range(len(r1)):
        if d1[i] > 25:
            g11[i] = -1
    for i in range(len(r2)):
        if d2[i] > 25:
            g22[i] = -1
    
    # Delete all the correspondences that are not 1 to 1
    b1 = [-1]*len(g2)
    b2 = [-1]*len(g1)
    for i in range(len(r1)):
        if g2[g1[i]] != i:
            g11[i] = -1
        else:
            if b1[g1[i]] == -1:
                b1[g1[i]] = i
            else:
                g11[i] = -1
                g11[b1[g1[i]]] = -1
    
    for i in range(len(r2)):
        if g1[g2[i]] != i:
            g22[i] = -1
        else:
            if b2[g2[i]] == -1:
                b2[g2[i]] = i
            else:
                g22[i] = -1
                g22[b2[g2[i]]] = -1
    
    # Build the confussion matrix
    c = 0
    classes = {}
    for i in range(len(r1)):
        clas = r1[i][1]
        if clas in classes:
            continue
        else:
            classes[clas] = c
            c += 1
    for i in range(len(r2)):
        clas = r2[i][1]
        if clas in classes:
            continue
        else:
            classes[clas] = c
            c += 1
    
    M = np.zeros((c+1, c+1))
    for i in range(len(r1)):
        clas = r1[i][1]
        if g11[i] == -1:
            M[classes[clas], c] += 1
        else:
            trueclas = r2[g11[i]][1]
            M[classes[clas], classes[trueclas]] += 1
    for i in range(len(r2)):
        trueclas = r2[i][1]
        if g22[i] == -1:
            M[c, classes[trueclas]] += 1
            
    # Confusion matrix with the predicted class in the rows and the actual class in the columns
    
    return M, [*classes.keys(), 'not detected']

def get_lst_from_json(json, labeled=True):
    reg = regions(json)
    if labeled:
        lst = [((0, 0), '')]*len(reg)
    else:
        lst = [(0, 0)]*len(reg)
    for i in range(len(reg)):
        if labeled:
            lst[i]=((reg[i]['shape_attributes']['cx'], reg[i]['shape_attributes']['cy']), reg[i]['region_attributes']['type'])
        else:
            lst[i]=(reg[i]['shape_attributes']['cx'], reg[i]['shape_attributes']['cy'])
    return lst


##############



def find_local_maxima(pred, h, centers=False):
    if not centers: #if gaussians
        #pred = color.rgb2gray(pred)
        pred = exposure.rescale_intensity(pred)
        h_maxima = extrema.h_maxima(pred, h)
    else: #if centroides
        h_maxima = pred
        
    connectivity = 4  
    output = cv2.connectedComponentsWithStats(h_maxima, connectivity, ltype=cv2.CV_32S)
    num_labels = output[0]
    centroids = output[3] #0 es background
    
    centr = []
    for i in range(num_labels):
        if i!=0: #background
            centr.append((int(centroids[i,1]), int(centroids[i,0])))
            
    centroides = np.zeros((h_maxima.shape))
    for i in centr:
        centroides[i[0], i[1]]=255


    return centroides, centr
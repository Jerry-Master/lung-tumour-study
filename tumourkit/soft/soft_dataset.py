"""
Module containing PyTorch dataset for both count and segment of the soft.

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
import cv2
from scipy.ndimage import gaussian_filter
import pandas as pd
from  torch.utils.data import Dataset as BaseDataset


class softDataset(BaseDataset):
    """Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_fps (list): List containing file names without extension.
        images_dir (str): Path to folder containing images.
        masks_dir (str): path to segmentation masks folder.
        sigma (float): value of the variance of the gaussians (count only).
        classes (list): values of class names to extract from segmentation mask (segment only).
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.).
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.).
        mode (str): Either segment or count.
        class_path (str): Path to folder containing class csv (segment only).
    """
    
    CLASSES = None
    
    def __init__(
            self, 
            images_fps, 
            images_dir,
            masks_dir, 
            sigma=None,
            augmentation=None, 
            preprocessing=None,
            mode=None,
            class_path=None
    ):
        super().__init__()
        assert mode is not None, 'Must specify mode (segment or count).'
        assert mode == 'count' or mode == 'segment',  'mode must be either count of segment.'
        self.mode = mode
        if self.mode == 'segment':
            self.class_path = class_path
        self.images_fps = images_fps
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        # Specific for count
        if self.mode == 'count':
            assert sigma is not None, 'Must provide variance for mode count.'
            self.sigma = sigma 
    
    
    def __getitem__(self, i):
        name = self.images_fps[i]
        # Input image
        self.image = cv2.imread(self.images_dir + name + '.png')
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image = cv2.resize(self.image, (512,512))

        # Different outputs images
        if self.mode == 'count':
            self.mask = cv2.imread(self.masks_dir + name + '.points.png', 0)
            self.mask = self.mask/255
            # Put a gaussian in each centroid
            self.mask = gaussian_filter(self.mask, sigma=self.sigma)
            self.mask = self.mask/self.mask.max()
            self.mask = cv2.resize(self.mask, (512,512), interpolation=cv2.INTER_NEAREST) 
            self.mask = self.mask.reshape((*self.mask.shape, 1)).astype('float')
            self.mask = self.mask.reshape((512,512,1)).astype('float')
        elif self.mode == 'segment':
            self.mask = cv2.imread(self.masks_dir + name + '.GT_cells.png',-1)
            self.mask = cv2.resize(self.mask, (512,512), interpolation=cv2.INTER_NEAREST)
            # get the classes of each cell
            self.mapping = pd.read_csv(self.class_path + name + '.class.csv', header=None)
            self.mapping = dict((int(idx),int(label)) for _, (idx, label) in self.mapping.iterrows())
            self.mapid2class()

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=self.image, mask=self.mask)
            self.image, self.mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=self.image, mask=self.mask)
            self.image, self.mask = sample['image'], sample['mask']
            
        return self.image, self.mask

    def mapid2class(self):
        all_classes = np.zeros(self.mask.shape)
        for i in self.mapping:
            all_classes[self.mask==i] = self.mapping[i]
        # labels used in the masks for each class, in the same order as CLASSES         
        num_classes = 2 # Change to more general parameter
        self.mask = self.separate_channels(all_classes, num_classes)

    def separate_channels(self, all_classes, num_classes):
        # Separate into one channel per class
        if num_classes < 2 :
            classes = [(all_classes > 0)]
        else:
            classes = [(all_classes == v) for v in range(1+num_classes)]
        return np.stack(classes, axis=-1).astype('float')
        
    def __len__(self):
        return len(self.images_fps)
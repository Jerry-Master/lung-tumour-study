"""
Script to train segment and count.

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
import albumentations as albu
import soft_segmentation_models_pytorch as smp
from soft_dataset import softDataset
from torch.utils.data import DataLoader
import itertools
import torch
import argparse
import numpy as np
import sys
import os

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PKG_DIR)

from utils.preprocessing import *

parser = argparse.ArgumentParser()
parser.add_argument('--img-dir', type=str, required=True,
                    help='Path to original image folder.')
parser.add_argument('--mask-dir-count-train', type=str, required=True,
                    help='Path to count masks folder of training.')
parser.add_argument('--mask-dir-count-val', type=str, required=True,
                    help='Path to count masks folder of validation.')
parser.add_argument('--mask-dir-segment-train', type=str, required=True,
                    help='Path to segment masks folder of training.')
parser.add_argument('--mask-dir-segment-val', type=str, required=True,
                    help='Path to segment masks folder of validation.')
parser.add_argument('--cls-dir-train', type=str, required=True,
                    help='Path to class folder of training.')
parser.add_argument('--cls-dir-val', type=str, required=True,
                    help='Path to class folder of validation.')
parser.add_argument('--train-path', type=str, required=True,
                    help='Path to train_names.txt.')
parser.add_argument('--val-path', type=str, required=True,
                    help='Path to validation_names.txt.')
parser.add_argument('--output-path', type=str, required=True,
                    help='Path to save models.')

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda'
ACTIVATION = 'sigmoid'

def get_training_augmentation(DAparams):
    train_transform = []
    if DAparams['flip']:
        flip = albu.HorizontalFlip(p=0.5)
        train_transform.append(flip)
    if DAparams['shift']:
        shift = albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=10, shift_limit=0.1, p=1, border_mode=0)
        train_transform.append(shift)
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = []
    return albu.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    def to_tensor(x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def train(params, params_names, num, metrics_, model, train_dataset, valid_dataset, output_path, mode):
    print('\n------------'+str(num)+'----------')
    parameters_line = 'mode:' + mode + 'Thresh='+str(params['thresh']) + ', sigma='+ str(params['sigma']) + ', epochs='+ str(params['epochs']) + \
    ', loss=' + str(params['loss']).split('(')[0] + ', lr=' + str(params['lr']) + ', opt=' \
    + str(params['optimizer']).split(' ')[0] + ', bs=' + str(params['batch_size'])
    print(parameters_line)

    #Final Logs
    train_logs_all = {'loss':[],'fscore':[], 'recall':[], 'precision':[]} 
    valid_logs_all = {'loss':[],'fscore':[], 'recall':[], 'precision':[]} 

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0) 

    # create epoch runners 
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
      model, loss=params['loss'], metrics=metrics_, optimizer=params['optimizer'], device=DEVICE, verbose=True)

    valid_epoch = smp.utils.train.ValidEpoch(
      model, loss=params['loss'], metrics=metrics_, device=DEVICE, verbose=True)

    # train model for N epochs
    import re
    loss_name = str(params['loss']).split('(')[0]
    loss_name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', loss_name)
    loss_name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', loss_name).lower()

    max_score = 0

    n = params['epochs']
    for i in range(0, n):
        print('\nEpoch: {}'.format(i))
        
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        if mode == 'count':
            train_logs_all['loss'].append(train_logs[loss_name])
            valid_logs_all['loss'].append(valid_logs[loss_name])
            train_logs_all['fscore'].append(train_logs['custom_fscore'])
            valid_logs_all['fscore'].append(valid_logs['custom_fscore'])
            train_logs_all['recall'].append(train_logs['custom_recall'])
            valid_logs_all['recall'].append(valid_logs['custom_recall'])
            train_logs_all['precision'].append(train_logs['custom_precision'])
            valid_logs_all['precision'].append(valid_logs['custom_precision'])
        else:
            train_logs_all['loss'].append(train_logs[loss_name])
            valid_logs_all['loss'].append(valid_logs[loss_name])
            train_logs_all['fscore'].append(train_logs['fscore'])
            valid_logs_all['fscore'].append(valid_logs['fscore'])
            train_logs_all['recall'].append(train_logs['recall'])
            valid_logs_all['recall'].append(valid_logs['recall'])
            train_logs_all['precision'].append(train_logs['precision'])
            valid_logs_all['precision'].append(valid_logs['precision'])

        # do something (save model, change lr, etc.)
        if mode == 'count':
            if max_score < valid_logs['custom_fscore']:
                max_score = valid_logs['custom_fscore']
                torch.save(model, output_path+str(num)+'.best_model.pth')
                print('Model saved!')
        else:
            if max_score < valid_logs['fscore']:
                max_score = valid_logs['fscore']
                torch.save(model, output_path+str(num)+'.best_model.pth')
                print('Model saved!')

    #Save results
    params['optimizer'] = ''
    params['loss'] = ''
    data = {'params':params_names, 'training':train_logs_all, 'validation':valid_logs_all}

    #change the number at each test
    create_dir(output_path +'logs/')
    with open(output_path +'logs/'+str(num)+'.'+parameters_line+'.txt', 'w') as outfile:
        json.dump(data, outfile)
    print('Model saved as'+ str(num)+'.'+parameters_line+'.txt')


def hyperparameter_optimization(num, thresholds, sigmes, losses, learn_rates, optimizers, epochs, batch_sizes, train_dataset, valid_dataset, output_path, num_classes, mode):
    ''' grid search for all the hyperparameters in each list of values: trains every combination of values and returns the metrics and images'''
    hyperparameters = [thresholds, sigmes, losses, learn_rates, optimizers, epochs, batch_sizes]
    for th, s, loss, lr, opt, N, b in itertools.product(*hyperparameters):
        # restart weights
        model = smp.Unet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, classes=num_classes, activation=ACTIVATION,)
        metrics_ = [smp.utils.metrics.Fscore(threshold=0.5), smp.utils.metrics.Precision(threshold=0.5), 
                    smp.utils.metrics.Recall(threshold=0.5)]

        if loss == 'dice':
            l = smp.utils.losses.DiceLoss()
        elif loss == 'mse':
            l = smp.utils.losses.MSELoss()
            metrics_ = [smp.utils.metrics.CustomAll(threshold=th)]
        elif loss == 'focal':
            l = smp.losses.focal.FocalLoss(mode='multilabel')
        else:
            assert(False)
        
        if opt == 'adam':
            opt_ = torch.optim.Adam([dict(params=model.parameters(), lr=lr),])
        else:
            opt_ = torch.optim.SGD([dict(params=model.parameters(), lr=lr),])
        

        params = {'thresh': th, 'sigma': s, 'batch_size':b, 'optimizer': opt_, 'lr':lr, 'loss':l, 'epochs': N }
        params_names = {'thresh': th, 'sigma': s, 'batch_size':b, 'optimizer': opt, 'lr':lr, 'loss':loss, 'epochs': N}
        train(params, params_names, str(num), metrics_, model, train_dataset, valid_dataset, output_path, mode)        
        print(loss,lr,opt,N,b)
        num=num+1
                                

if __name__=='__main__':
    args = parser.parse_args()
    IMG_DIR = parse_path(args.img_dir)
    MASK_DIR_COUNT_TRAIN = parse_path(args.mask_dir_count_train)
    MASK_DIR_SEGMENT_TRAIN = parse_path(args.mask_dir_segment_train)
    CLS_DIR_TRAIN = parse_path(args.cls_dir_train)
    MASK_DIR_COUNT_VAL = parse_path(args.mask_dir_count_val)
    MASK_DIR_SEGMENT_VAL = parse_path(args.mask_dir_segment_val)
    CLS_DIR_VAL = parse_path(args.cls_dir_val)
    OUTPUT_PATH = parse_path(args.output_path)
    create_dir(OUTPUT_PATH)
    OUT_SEGMENT = OUTPUT_PATH + 'segment/'
    create_dir(OUT_SEGMENT)
    OUT_COUNT = OUTPUT_PATH + 'count/'
    create_dir(OUT_COUNT)
    train_names = read_names(args.train_path)
    val_names = read_names(args.val_path)

    # Loss
    losses = ['focal']
    # Hyperparameters
    learn_rates = [0.0005,0.00005,0.000005]
    optimizers = ['adam']
    epochs=[200]
    batch_sizes = [2,4,6,8]
    thresholds = [0.15]
    sigmes = [6] 

    num_model = 0
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    DA_params = {'flip':True, 'shift':True}
    # Segment
    train_dataset = softDataset(
        images_fps=train_names, images_dir=IMG_DIR, masks_dir=MASK_DIR_SEGMENT_TRAIN, class_path=CLS_DIR_TRAIN, mode='segment',
        augmentation=get_training_augmentation(DA_params), preprocessing=get_preprocessing(preprocessing_fn)
    )
    valid_dataset = softDataset(
        images_fps=val_names, images_dir=IMG_DIR, masks_dir=MASK_DIR_SEGMENT_VAL, class_path=CLS_DIR_VAL, mode='segment',
        augmentation=get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn)
    )
    hyperparameter_optimization(num_model, thresholds, sigmes, losses, learn_rates, optimizers, epochs, batch_sizes, train_dataset, valid_dataset, OUT_SEGMENT, 3, mode='segment')

    # Count
    # Loss
    losses = ['mse']
    train_dataset = softDataset(
        images_fps=train_names, images_dir=IMG_DIR, masks_dir=MASK_DIR_COUNT_TRAIN, mode='count', sigma=sigmes[0],
        augmentation=get_training_augmentation(DA_params), preprocessing=get_preprocessing(preprocessing_fn)
    )
    valid_dataset = softDataset(
        images_fps=val_names, images_dir=IMG_DIR, masks_dir=MASK_DIR_COUNT_VAL, mode='count', sigma=sigmes[0],
        augmentation=get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn)
    )
    hyperparameter_optimization(num_model, thresholds, sigmes, losses, learn_rates, optimizers, epochs, batch_sizes, train_dataset, valid_dataset, OUT_COUNT, 1, mode='count')

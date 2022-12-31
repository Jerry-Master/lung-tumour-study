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
import os
import numpy as np

def split_dataset(ori_path, gt_path, sufix):
    allFileNames = os.listdir(ori_path)

    np.random.seed(97)
    np.random.shuffle(allFileNames)

    train_FileNames, val_FileNames= np.split(np.array(allFileNames),[int(len(allFileNames)*0.8)])

    train_FileNames = [ name for name in train_FileNames.tolist()]
    val_FileNames = [ name for name in val_FileNames.tolist()]

    x_train = [ori_path + name for name in train_FileNames]
    y_train = [gt_path + name[:-4] + sufix + '.png' for name in train_FileNames]
    x_valid = [ori_path + name for name in val_FileNames]
    y_valid = [gt_path + name[:-4] + sufix + '.png' for name in val_FileNames]

    return x_train, y_train, x_valid, y_valid

def split_dataset_per_patient(ori_path, gt_path, sufix, removeStroma):
    allFileNames = os.listdir(ori_path)
    train_FileNames = []
    val_FileNames = []
    stroma=[]
    if removeStroma:
        stroma.append('372 KI67 (x=13022.0, y=146089.0, w=1500.0, h=1500.0).png')
        stroma.append('400 KI67 (x=26108.0, y=101121.0, w=1500.0, h=1500.0).png')

    for name in allFileNames:
        if name not in stroma:
            patient = int(name.split(' ')[0])
            if patient <= 1921:
                train_FileNames.append(name)
            else:
                val_FileNames.append(name)
    
    x_train = [ori_path + name for name in train_FileNames]
    y_train = [gt_path + name[:-4] + sufix + '.png' for name in train_FileNames]
    x_valid = [ori_path + name for name in val_FileNames]
    y_valid = [gt_path + name[:-4] + sufix + '.png' for name in val_FileNames]

    return x_train, y_train, x_valid, y_valid


################################# TO PLOT MODEL LEARNING CURVES
import glob
import matplotlib.pyplot as plt
import json
def plot(train, valid, doc, epochs, max_value, max_epoch):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    fig.suptitle(doc.name.split('/')[-1])
    #train
    data = train
    ax1.plot( epochs, data['loss'], marker='', color='indianred', linewidth=1, label='loss')
    ax1.plot( epochs, data['fscore'], marker='', color='olivedrab', linewidth=1, label='fscore')
    ax1.plot( epochs, data['recall'], marker='', color='cornflowerblue', linewidth=1,  label='recall')
    ax1.plot( epochs, data['precision'], marker='', color='darkturquoise', linewidth=1, label='precision')

    ax1.legend()
    for line in ax1.legend().get_lines():
        line.set_linewidth(2.0)
  
    ax1.set(xlabel='epochs', title='Training', ylim=(0,1.1))

    #valid
    data = valid
    ax2.plot( epochs, data['loss'], marker='', color='indianred', linewidth=1, label='loss')
    ax2.plot( epochs, data['fscore'], marker='', color='olivedrab', linewidth=1, label='fscore')
    ax2.plot( epochs, data['recall'], marker='', color='cornflowerblue', linewidth=1,  label='recall')
    ax2.plot( epochs, data['precision'], marker='', color='darkturquoise', linewidth=1, label='precision')

    ax2.legend()
    for line in ax2.legend().get_lines():
        line.set_linewidth(2.0)
    
    ax2.axvline(x=max_epoch, color='gray', linestyle='dotted', linewidth=0.75)
    ax2.plot(max_epoch, max_value, '.', color='gray', markersize=5)
    ax2.set(xlabel='epochs', title='Validation', ylim=(0,1.1))
    
    
def logs_graphic(path, model):
    filePaths = glob.glob(os.path.join(path,'{0}*.txt'.format(str(model))))

    # Just open first ocurrence, if any
    if filePaths:
        print("Found: ", filePaths[0])
        doc = open(filePaths[0], 'r')
        data = json.load(doc)

        valid_logs_all = data['validation']
        train_logs_all = data['training']
        params = data['params']
        epochs = [i for i in range(0,params['epochs'])]

        max_fscore_value = max(valid_logs_all['fscore'])
        max_fscore_epoch = valid_logs_all['fscore'].index(max_fscore_value)
        last_fscore_value = valid_logs_all['fscore'][params['epochs']-1]
        
        plot(train_logs_all, valid_logs_all, doc, epochs, max_fscore_value, max_fscore_epoch)



        print('The maximum Fscore is '+ str(max_fscore_value) +' in epoch number ' + str(max_fscore_epoch))
        print('In this epoch the precision is ' +str(valid_logs_all['precision'][max_fscore_epoch]) + ' and recall is '+ str(valid_logs_all['recall'][max_fscore_epoch]))
        print('The last Fscore value is ' + str(last_fscore_value))

    else:
        print("not found")
"""
Script to train and save several GNN configurations.

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
import math
from typing import Optional, Tuple, Dict, List, Any
from sklearn.preprocessing import Normalizer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
import dgl
from dgl.dataloading import GraphDataLoader
from read_graph import GraphDataset
from models.gcn import GCN
from models.hgao import HardGAT
from models.gat import GAT
import argparse
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')
import sys
import os
import json
import pickle

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PKG_DIR)

from utils.preprocessing import parse_path, create_dir
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--node-dir', type=str, required=True,
                     help='Path to folder containing train and validation folder with .nodes.csv files.')
parser.add_argument('--log-dir', type=str, required=True,
                     help='Path to save tensorboard logs.')
parser.add_argument('--early-stopping-rounds', type=int, required=True,
                     help='Number of epochs needed to consider convergence when worsening.')
parser.add_argument('--batch-size', type=int, required=True,
                     help='Batch size. No default.')
parser.add_argument('--model-name', type=str, required=True, choices=['GCN', 'ATT', 'HATT', 'SAGE', 'BOOST'],
                     help='Which model to use. Options: GCN, ATT, HATT, SAGE, BOOST')
parser.add_argument('--save-file', type=str, required=True,
                     help='Name to file where to save the results. Must not contain extension.')
parser.add_argument('--num-confs', type=int, default=50,
                     help='Upper bound on the number of configurations to try.')
parser.add_argument('--save-dir', type=str,
                     help='Folder to save models weights and confs.')                     

def evaluate(
    loader: GraphDataLoader,
    model: nn.Module,
    device: str,
    writer: Optional[SummaryWriter] = None,
    epoch: Optional[int] = None,
    log_suffix: Optional[str] = None
    ) -> Tuple[float, float, float]:
    """
    Evaluates model in loader.
    Logs to tensorboard with suffix log_suffix.
    Returns the model in evaluation mode.
    """
    model.eval()
    preds, labels, probs = np.array([]).reshape(0,1), np.array([]).reshape(0,1), np.array([]).reshape(0,1)
    for g in loader:
        # self-loops
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        # data
        features = g.ndata['X'].to(device)
        # Forward
        logits = model(g, features)
        pred = logits.argmax(1).detach().numpy().reshape(-1,1)
        preds = np.vstack((preds, pred))
        prob = F.softmax(logits, dim=1).detach().numpy()[:,1].reshape(-1,1)
        probs = np.vstack((probs, prob))
        label = g.ndata['y'].detach().numpy().reshape(-1,1)
        labels = np.vstack((labels, label))
    # Compute metrics on validation  
    f1 = f1_score(labels, preds)  
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs)
    # Tensorboard
    if writer is not None:
        assert(log_suffix is not None and epoch is not None)
        writer.add_scalar('Accuracy/' + log_suffix, acc, epoch)
        writer.add_scalar('F1/' + log_suffix, f1, epoch)
        writer.add_scalar('ROC_AUC/' + log_suffix, auc, epoch)
    return f1, acc, auc

def train_one_iter(
    tr_loader: GraphDataLoader,
    model: nn.Module,
    device: str,
    optimizer: Optimizer,
    epoch: int,
    writer: SummaryWriter
) -> None:
    """
    Trains for one iteration, as the name says.
    """
    model.train()
    for step, tr_g in enumerate(tr_loader):
        # self-loops
        tr_g = dgl.remove_self_loop(tr_g)
        tr_g = dgl.add_self_loop(tr_g)
        # data
        features = tr_g.ndata['X'].to(device)
        labels = tr_g.ndata['y'].to(device)
        # Forward
        logits = model(tr_g, features)
        loss = F.cross_entropy(logits, labels)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Compute metrics on training
        pred = logits.argmax(1).detach().numpy()
        labels = labels.detach().numpy()
        train_acc = accuracy_score(labels, pred)
        train_f1 = f1_score(labels, pred)
        probs = F.softmax(logits, dim=1).detach().numpy()[:,1]
        train_auc = roc_auc_score(labels, probs)
        # Tensorboard
        writer.add_scalar('Accuracy/train', train_acc, step+len(tr_loader)*epoch)
        writer.add_scalar('F1/train', train_f1, step+len(tr_loader)*epoch)
        writer.add_scalar('ROC_AUC/train', train_auc, step+len(tr_loader)*epoch)

def train(
    tr_loader: GraphDataLoader, 
    val_loader: GraphDataLoader, 
    model: nn.Module,
    optimizer: Optimizer,
    writer: SummaryWriter,
    n_early: int,
    device: Optional[str] = 'cpu'
    ) -> None:
    """
    Train the model with early stopping on F1 score or until 1000 iterations.
    """
    model.to(device)
    n_epochs = 1000
    best_val_f1 = 0
    early_stop_rounds = 0
    for epoch in range(n_epochs):
        train_one_iter(tr_loader, model, device, optimizer, epoch, writer)
        val_f1, val_acc, val_auc = evaluate(val_loader, model, device, writer, epoch, 'validation')
        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            early_stop_rounds = 0
        elif early_stop_rounds < n_early:
            early_stop_rounds += 1
        else:
            return
        

def load_dataset(node_dir: str, bsize: int) -> Tuple[GraphDataLoader, GraphDataLoader, GraphDataLoader]:
    """
    Creates Torch dataloaders for training. 
    Folder structure:
    node_dir:
     - train
       - file1.nodes.csv
       ...
     - validation
       - file1.nodes.csv
       ...
     - test
       - file1.nodes.csv
       ...
    """
    train_dataset = GraphDataset(
        node_dir=os.path.join(node_dir,'train'), 
        max_dist=200, max_degree=10, column_normalize=True)
    train_dataloader = GraphDataLoader(train_dataset, batch_size=bsize, shuffle=True)
    val_dataset = GraphDataset(
        node_dir=os.path.join(node_dir, 'validation'), 
        max_dist=200, max_degree=10, normalizers=train_dataset.get_normalizers())
    val_dataloader = GraphDataLoader(val_dataset, batch_size=1, shuffle=False)
    test_dataset = GraphDataset(
        node_dir=os.path.join(node_dir, 'test'), 
        max_dist=200, max_degree=10, normalizers=train_dataset.get_normalizers())
    test_dataloader = GraphDataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_dataloader, val_dataloader, test_dataloader

def generate_configurations(max_confs: int, model_name: str) -> List[Dict[str, int]]:
    """
    Generates a grid in the search space with no more than max_confs configurations.
    Parameters changed: NUM_LAYERS, DROPOUT, NORM_TYPE
    """
    num_layers_confs = int(math.sqrt(max_confs / 2))
    num_dropout_confs = int(max_confs // (2 * num_layers_confs))
    assert(2 * num_layers_confs * num_dropout_confs <= max_confs)
    assert num_layers_confs <= 15, 'Too many layers'
    confs = []
    for num_layers in np.linspace(1,15, num_layers_confs):
        num_layers = int(num_layers)
        for dropout in np.linspace(0,0.9, num_dropout_confs):
            conf = {}
            conf['MODEL_NAME'] = model_name
            conf['NUM_LAYERS'] = num_layers
            conf['DROPOUT'] = dropout
            conf['NORM_TYPE'] = 'bn'
            confs.append(conf)

            conf = {}
            conf['MODEL_NAME'] = model_name
            conf['NUM_LAYERS'] = num_layers
            conf['DROPOUT'] = dropout
            conf['NORM_TYPE'] = 'gn'
            confs.append(conf)
    return confs

def load_model(conf: Dict[str,Any]) -> nn.Module:
    """
    Available models: GCN, ATT, HATT, SAGE, BOOST
    Configuration space: NUM_LAYERS, DROPOUT, NORM_TYPE
    """
    num_feats = 18
    num_classes = 2
    hidden_feats = 100
    if conf['MODEL_NAME'] == 'GCN':
        return GCN(num_feats, hidden_feats, num_classes, conf['NUM_LAYERS'], conf['DROPOUT'], conf['NORM_TYPE'])
    if conf['MODEL_NAME'] == 'ATT' or conf['MODEL_NAME'] == 'HATT':
        num_heads = 8
        num_out_heads = 1
        heads = ([num_heads] * conf['NUM_LAYERS']) + [num_out_heads]
        if conf['MODEL_NAME'] == 'ATT':
            return GAT(num_feats, hidden_feats, num_classes, heads, conf['NUM_LAYERS'], conf['DROPOUT'], conf['NORM_TYPE'])
        return HardGAT(num_feats, hidden_feats, num_classes, heads, conf['NUM_LAYERS'], conf['DROPOUT'], conf['NORM_TYPE'])
    assert False, 'Model not implemented.'

def create_results_file(filename: str) -> None:
    """
    Creates header of .csv result file to append results.
    filename must not contain extension.
    """
    with open(filename + '.csv', 'w') as f:
        print('F1 Score,Accuracy,ROC AUC,NUM_LAYERS,DROPOUT,NORM_TYPE', file=f)

def append_results(
    filename: str, f1: float, acc: float, auc: float, num_layers: int, dropout: float, bn_type: str
) -> None:
    """
    Appends result to given filename.
    filename must not contain extension.
    """
    with open(filename + '.csv', 'a') as f:
        print(f1, acc, auc, num_layers, dropout, bn_type, file=f, sep=',')

def name_from_conf(conf: Dict[str, Any]) -> str:
    """
    Generates a name from the configuration object.
    """
    return conf['MODEL_NAME'] + '_' + str(conf['NUM_LAYERS']) + '_' \
        + str(conf['DROPOUT']) + '_' + str(conf['NORM_TYPE']) 

def save_model(model: nn.Module, conf: Dict[str, Any], normalizers: Tuple[Normalizer]) -> None:
    """
    Save model weights and configuration file to SAVE_DIR
    """
    name = name_from_conf(conf)
    state_dict = model.state_dict()
    torch.save(state_dict, SAVE_DIR + 'weights/' + name + '.pth')
    with open(SAVE_DIR + 'confs/' + name + '.json', 'w') as f:
        json.dump(conf, f)
    with open(SAVE_DIR + 'normalizers/' + name + '.pkl', 'wb') as f:
        pickle.dump(normalizers, f)

def main(args):
    # Tensorboard logs
    writer = SummaryWriter(log_dir=LOG_DIR)
    # Datasets
    train_dataloader, val_dataloader, test_dataloader = load_dataset(args.node_dir, args.batch_size)
    # Configurations
    confs = generate_configurations(args.num_confs, args.model_name)
    create_results_file(args.save_file)
    for conf in confs:
        # Model
        model = load_model(conf)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # Train
        train(train_dataloader, val_dataloader, model, optimizer, writer, args.early_stopping_rounds)
        test_f1, test_acc, test_auc = evaluate(test_dataloader, model, 'cpu')
        append_results(args.save_file, test_f1, test_acc, test_auc, conf['NUM_LAYERS'], conf['DROPOUT'], conf['NORM_TYPE'])
        if SAVE_WEIGHTS:
            save_model(model, conf, train_dataloader.dataset.get_normalizers())


if __name__=='__main__':   
    args = parser.parse_args()
    LOG_DIR = parse_path(args.log_dir)
    create_dir(LOG_DIR)
    SAVE_WEIGHTS = False
    if args.save_dir is not None:
        SAVE_WEIGHTS = True
        SAVE_DIR = parse_path(args.save_dir)
        create_dir(SAVE_DIR)
        create_dir(SAVE_DIR + 'weights')
        create_dir(SAVE_DIR + 'confs')
        create_dir(SAVE_DIR + 'normalizers')
    main(args)
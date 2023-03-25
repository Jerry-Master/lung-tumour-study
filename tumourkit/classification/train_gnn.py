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
from .read_graph import GraphDataset
from .models.gcn import GCN
from .models.hgao import HardGAT
from .models.gat import GAT
import argparse
from argparse import Namespace
from torch.utils.tensorboard import SummaryWriter
import warnings
import os
from concurrent.futures import ThreadPoolExecutor
import json
import pickle
from ..utils.preprocessing import parse_path, create_dir
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
warnings.filterwarnings('ignore')

def evaluate(
        loader: GraphDataLoader,
        model: nn.Module,
        device: str,
        writer: Optional[SummaryWriter] = None,
        epoch: Optional[int] = None,
        log_suffix: Optional[str] = None,
        num_classes: Optional[str] = 2,
        ) -> Tuple[float, float, float]:
    """
    Evaluates model in loader.
    Logs to tensorboard with suffix log_suffix.
    Returns the model in evaluation mode.
    """
    model.eval()
    preds, labels, probs = np.array([]).reshape(0,1), np.array([]).reshape(0,1), np.array([]).reshape(0,1)
    for g in loader:
        g = g.to(device)
        # self-loops
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        # data
        features = g.ndata['X']
        # Forward
        logits = model(g, features)
        pred = logits.argmax(1).detach().cpu().numpy().reshape(-1,1)
        preds = np.vstack((preds, pred))
        if num_classes == 2:
            prob = F.softmax(logits, dim=1).detach().cpu().numpy()[:,1].reshape(-1,1)
            probs = np.vstack((probs, prob))
        label = g.ndata['y'].detach().cpu().numpy().reshape(-1,1)
        labels = np.vstack((labels, label))
    # Compute metrics on validation  
    if num_classes == 2:
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
    else:
        micro = f1_score(labels, preds, average='micro')
        macro = f1_score(labels, preds, average='macro')
        weighted = f1_score(labels, preds, average='weighted')
        # Tensorboard
        if writer is not None:
            writer.add_scalar('Accuracy/' + log_suffix, micro, epoch)
            writer.add_scalar('Macro F1/' + log_suffix, macro, epoch)
            writer.add_scalar('Weighted F1/' + log_suffix, weighted, epoch)
        return micro, macro, weighted
    

def train_one_iter(
        tr_loader: GraphDataLoader,
        model: nn.Module,
        device: str,
        optimizer: Optimizer,
        epoch: int,
        writer: SummaryWriter,
        num_classes: int,
        ) -> None:
    """
    Trains for one iteration, as the name says.
    """
    model.train()
    for step, tr_g in enumerate(tr_loader):
        tr_g = tr_g.to(device)
        # self-loops
        tr_g = dgl.remove_self_loop(tr_g)
        tr_g = dgl.add_self_loop(tr_g)
        # data
        features = tr_g.ndata['X']
        labels = tr_g.ndata['y']
        # Forward
        logits = model(tr_g, features)
        loss = F.cross_entropy(logits, labels)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Compute metrics on training
        pred = logits.argmax(1).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        if num_classes == 2:
            train_acc = accuracy_score(labels, pred)
            train_f1 = f1_score(labels, pred)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()[:,1]
            train_auc = None
            if len(np.unique(labels)) > 1:
                train_auc = roc_auc_score(labels, probs)
            
            # Tensorboard
            writer.add_scalar('Accuracy/train', train_acc, step+len(tr_loader)*epoch)
            writer.add_scalar('F1/train', train_f1, step+len(tr_loader)*epoch)
            if train_auc is not None:
                writer.add_scalar('ROC_AUC/train', train_auc, step+len(tr_loader)*epoch)
        else:
            train_micro = f1_score(labels, pred, average='micro')
            train_macro = f1_score(labels, pred, average='macro')
            train_weighted = f1_score(labels, pred, average='weighted')
            # Tensorboard
            writer.add_scalar('Accuracy/train', train_micro, step+len(tr_loader)*epoch)
            writer.add_scalar('Macro F1/train', train_macro, step+len(tr_loader)*epoch)
            writer.add_scalar('Weighted F1/train', train_weighted, step+len(tr_loader)*epoch)

def train(
        save_dir: str,
        save_weights: bool,
        tr_loader: GraphDataLoader, 
        val_loader: GraphDataLoader, 
        model: nn.Module,
        optimizer: Optimizer,
        writer: SummaryWriter,
        n_early: int,
        device: Optional[str] = 'cpu',
        check_iters: Optional[int] = -1,
        conf: Optional[Dict[str,Any]] = None,
        normalizers: Optional[Tuple[Normalizer]] = None,
        num_classes: Optional[int] = 2,
        ) -> None:
    """
    Train the model with early stopping on F1 score (weighted) or until 1000 iterations.
    """
    model = model.to(device)
    n_epochs = 1000
    best_val_f1 = 0
    early_stop_rounds = 0
    for epoch in range(n_epochs):
        train_one_iter(tr_loader, model, device, optimizer, epoch, writer, num_classes)
        val_metrics = evaluate(val_loader, model, device, writer, epoch, 'validation', num_classes=num_classes)
        if num_classes == 2:
            val_f1, val_acc, val_auc = val_metrics
        else:
            val_micro, val_macro, val_f1 = val_metrics
        # Save checkpoint
        if save_weights and check_iters != -1 and epoch % check_iters == 0:
            save_model(save_dir, model, conf, normalizers, prefix='last_')
        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            early_stop_rounds = 0
            if save_weights:
                save_model(save_dir, model, conf, normalizers, prefix='best_')
        elif early_stop_rounds < n_early:
            early_stop_rounds += 1
        else:
            return
        

def load_dataset(
        train_node_dir: str,
        val_node_dir: str,
        test_node_dir: str,
        bsize: int,
        ) -> Tuple[GraphDataLoader, GraphDataLoader, GraphDataLoader]:
    """
    Creates Torch dataloaders for training. 
    Folder structure:
    node_dir:
     - train
      - graphs
       - file1.nodes.csv
       ...
     - validation
      - graphs
       - file1.nodes.csv
       ...
     - test
      - graphs
       - file1.nodes.csv
       ...
    """
    train_dataset = GraphDataset(
        node_dir=train_node_dir,
        max_dist=200, max_degree=10, column_normalize=True)
    train_dataloader = GraphDataLoader(train_dataset, batch_size=bsize, shuffle=True)
    val_dataset = GraphDataset(
        node_dir=val_node_dir,
        max_dist=200, max_degree=10, normalizers=train_dataset.get_normalizers())
    val_dataloader = GraphDataLoader(val_dataset, batch_size=1, shuffle=False)
    test_dataset = GraphDataset(
        node_dir=test_node_dir,
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
            conf['NORM_TYPE'] = None
            confs.append(conf)
    return confs

def load_model(conf: Dict[str,Any], num_classes: int) -> nn.Module:
    """
    Available models: GCN, ATT, HATT, SAGE, BOOST
    Configuration space: NUM_LAYERS, DROPOUT, NORM_TYPE
    """
    num_feats = 18 + (1 if num_classes == 2 else num_classes)
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

def create_results_file(filename: str, num_classes: int) -> None:
    """
    Creates header of .csv result file to append results.
    filename must not contain extension.
    """
    if num_classes == 2:
        with open(filename + '.csv', 'w') as f:
            print('F1 Score,Accuracy,ROC AUC,NUM_LAYERS,DROPOUT,NORM_TYPE', file=f)
    else:
        with open(filename + '.csv', 'w') as f:
            print('Micro F1,Macro F1,Weighted F1,NUM_LAYERS,DROPOUT,NORM_TYPE', file=f)

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

def save_model(
        save_dir: str,
        model: nn.Module,
        conf: Dict[str, Any],
        normalizers: Tuple[Normalizer],
        prefix: Optional[str] = ''
        ) -> None:
    """
    Save model weights and configuration file to SAVE_DIR
    """
    name = prefix + name_from_conf(conf)
    state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(save_dir, 'weights', name + '.pth'))
    with open(os.path.join(save_dir, 'confs', name + '.json'), 'w') as f:
        json.dump(conf, f)
    with open(os.path.join(save_dir, 'normalizers', name + '.pkl'), 'wb') as f:
        pickle.dump(normalizers, f)

def train_one_conf(
        args: Namespace,
        conf: Dict[str, Any],
        train_dataloader: GraphDataLoader,
        val_dataloader: GraphDataLoader,
        test_dataloader: GraphDataLoader,
        log_dir: str,
        save_weights: bool,
        save_dir: str,
        num_classes: int,
        ) -> Tuple[float, float, float, nn.Module, Dict[str, Any]]:
    # Tensorboard logs
    writer = SummaryWriter(log_dir=os.path.join(log_dir, name_from_conf(conf)))
    # Model
    model = load_model(conf, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Train
    train(
        save_dir, save_weights, train_dataloader, val_dataloader,
        model, optimizer, writer, args.early_stopping_rounds,
        args.device, args.checkpoint_iters, conf, train_dataloader.dataset.get_normalizers(),
        num_classes=num_classes
    )
    test_metrics = evaluate(test_dataloader, model, args.device, num_classes=num_classes)
    model = model.cpu()
    if num_classes == 2:
        test_f1, test_acc, test_auc = test_metrics
        return test_f1, test_acc, test_auc, model, conf
    else:
        test_micro, test_macro, test_weighted = test_metrics
        return test_micro, test_macro, test_weighted, model, conf


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-node-dir', type=str, required=True,
                        help='Path to folder containing train folder with .nodes.csv files.')
    parser.add_argument('--validation-node-dir', type=str, required=True,
                        help='Path to folder containing validation folder with .nodes.csv files.')
    parser.add_argument('--test-node-dir', type=str, required=True,
                        help='Path to folder containing test folder with .nodes.csv files.')
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
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu',
                        help='Device to execute. Either cpu or cuda. Default: cpu.')
    parser.add_argument('--num-workers', type=int, default=1, 
                        help='Number of processors to use. Default: 1.')
    parser.add_argument('--checkpoint-iters', type=int, default=-1, 
                        help='Number of iterations at which to save model periodically while training. Set to -1 for no checkpointing. Default: -1.')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes to consider for classification (background not included).')
    return parser


def main_with_args(args: Namespace):
    log_dir = parse_path(args.log_dir)
    create_dir(log_dir)
    save_weights = False
    if args.save_dir is not None:
        save_weights = True
        save_dir = parse_path(args.save_dir)
        create_dir(save_dir)
        create_dir(save_dir + 'weights')
        create_dir(save_dir + 'confs')
        create_dir(save_dir + 'normalizers')
    # Datasets
    train_dataloader, val_dataloader, test_dataloader = load_dataset(
        train_node_dir=args.train_node_dir,
        val_node_dir=args.validation_node_dir,
        test_node_dir=args.test_node_dir,
        bsize=args.batch_size,
    )
    # Configurations
    confs = generate_configurations(args.num_confs, args.model_name)
    create_results_file(args.save_file, args.num_classes)
    if args.num_workers > 0:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []
            for conf in confs:
                future = executor.submit(
                    train_one_conf,
                    args, conf, train_dataloader, val_dataloader, test_dataloader, log_dir, save_weights, save_dir, args.num_classes
                )
                futures.append(future)
            for future in futures:
                if args.num_classes == 2:
                    test_f1, test_acc, test_auc, model, conf = future.result()
                    append_results(args.save_file, test_f1, test_acc, test_auc, conf['NUM_LAYERS'], conf['DROPOUT'], conf['NORM_TYPE'])
                else:
                    test_micro, test_macro, test_weighted, model, conf = future.result()
                    append_results(args.save_file, test_micro, test_macro, test_weighted, conf['NUM_LAYERS'], conf['DROPOUT'], conf['NORM_TYPE'])
                if save_weights:
                    save_model(save_dir, model, conf, train_dataloader.dataset.get_normalizers(), 'last_')
    else:
        for conf in confs:
            if args.num_classes == 2:
                test_f1, test_acc, test_auc, model, conf = train_one_conf(
                    args, conf, train_dataloader, val_dataloader, test_dataloader, log_dir, save_weights, save_dir, args.num_classes
                )
                append_results(args.save_file, test_f1, test_acc, test_auc, conf['NUM_LAYERS'], conf['DROPOUT'], conf['NORM_TYPE'])
            else:
                test_micro, test_macro, test_weighted, model, conf = train_one_conf(
                    args, conf, train_dataloader, val_dataloader, test_dataloader, log_dir, save_weights, save_dir, args.num_classes
                )
                append_results(args.save_file, test_micro, test_macro, test_weighted, conf['NUM_LAYERS'], conf['DROPOUT'], conf['NORM_TYPE'])
            if save_weights:
                save_model(save_dir, model, conf, train_dataloader.dataset.get_normalizers(), 'last_')


def main():
    parser = _create_parser()
    args = parser.parse_args()
    main(args)


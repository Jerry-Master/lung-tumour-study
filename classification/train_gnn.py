from typing import Optional
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

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PKG_DIR)

from utils.preprocessing import *
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

def train(
    tr_loader: GraphDataset, 
    val_loader: GraphDataset, 
    model: nn.Module,
    optimizer: Optimizer,
    writer: SummaryWriter,
    n_epochs: int,
    device: Optional[str] = 'cpu'
    ) -> None:
    model.to(device)
    for epoch in range(n_epochs):
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
            writer.add_scalar('Accuracy/train', train_acc, step+len(tr_loader)*epoch)
            train_f1 = f1_score(labels, pred)
            writer.add_scalar('F1/train', train_f1, step+len(tr_loader)*epoch)
            probs = F.softmax(logits, dim=1).detach().numpy()[:,1]
            train_auc = roc_auc_score(labels, probs)
            writer.add_scalar('ROC_AUC/train', train_auc, step+len(tr_loader)*epoch)
            
        model.eval()
        preds, labels, probs = np.array([]).reshape(0,1), np.array([]).reshape(0,1), np.array([]).reshape(0,1)
        for val_g in val_loader:
            # self-loops
            val_g = dgl.remove_self_loop(val_g)
            val_g = dgl.add_self_loop(val_g)
            # data
            features = val_g.ndata['X'].to(device)
            # Forward
            logits = model(val_g, features)
            pred = logits.argmax(1).detach().numpy().reshape(-1,1)
            preds = np.vstack((preds, pred))
            prob = F.softmax(logits, dim=1).detach().numpy()[:,1].reshape(-1,1)
            probs = np.vstack((probs, prob))
            label = val_g.ndata['y'].detach().numpy().reshape(-1,1)
            labels = np.vstack((labels, label))
        # Compute metrics on validation    
        val_acc = accuracy_score(labels, preds)
        writer.add_scalar('Accuracy/validation', val_acc, step+len(val_loader)*epoch)
        val_f1 = f1_score(labels, preds)
        writer.add_scalar('F1/validation', val_f1, step+len(val_loader)*epoch)
        val_auc = roc_auc_score(labels, probs)
        writer.add_scalar('ROC_AUC/validation', val_auc, step+len(val_loader)*epoch)

parser = argparse.ArgumentParser()
parser.add_argument('--node-dir', type=str, required=True,
                     help="Path to .nodes.csv files.")
parser.add_argument('--log-dir', type=str, required=True,
                     help="Path to save tensorboard logs.")
parser.add_argument('--epochs', type=int, required=True,
                     help="Number of epochs to train for.")

if __name__=='__main__':   
    args = parser.parse_args()
    NUM_CLASSES = 2
    NUM_FEATS = 18
    HIDDEN_FEATS = 100
    NUM_LAYERS = 5
    LOG_DIR = parse_path(args.log_dir)
    create_dir(LOG_DIR)

    writer = SummaryWriter(log_dir=LOG_DIR)
    train_dataset = GraphDataset(node_dir=os.path.join(args.node_dir,'train'), max_dist=200, max_degree=10)
    train_dataloader = GraphDataLoader(train_dataset, batch_size=20, shuffle=True)
    val_dataset = GraphDataset(node_dir=os.path.join(args.node_dir, 'validation'), max_dist=200, max_degree=10)
    val_dataloader = GraphDataLoader(val_dataset, batch_size=1, shuffle=True)
    model = GCN(NUM_FEATS, HIDDEN_FEATS, NUM_CLASSES, NUM_LAYERS)
    NUM_HEADS = 8
    NUM_OUT_HEADS = 1
    heads = ([NUM_HEADS] * NUM_LAYERS) + [NUM_OUT_HEADS]
    # model = GAT(NUM_FEATS, HIDDEN_FEATS, NUM_CLASSES, heads, NUM_LAYERS)
    # model = HardGAT(NUM_LAYERS, NUM_FEATS, HIDDEN_FEATS, NUM_CLASSES, heads, F.elu)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train(train_dataloader, val_dataloader, model, optimizer, writer, args.epochs)
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from dgl.dataloading import GraphDataLoader
from dgl.nn import GraphConv
from read_graph import GraphDataset
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

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats, allow_zero_in_degree=True)
        self.conv2 = GraphConv(h_feats, h_feats, allow_zero_in_degree=True)
        self.conv3 = GraphConv(h_feats, h_feats, allow_zero_in_degree=True)
        self.conv4 = GraphConv(h_feats, h_feats, allow_zero_in_degree=True)
        self.conv5 = GraphConv(h_feats, num_classes, allow_zero_in_degree=True)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = self.conv3(g, h)
        h = F.relu(h)
        h = self.conv4(g, h)
        h = F.relu(h)
        h = self.conv5(g, h)
        return h

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
        writer.add_scalar('Accuracy/validation', val_acc, step+len(tr_loader)*epoch)
        val_f1 = f1_score(labels, preds)
        writer.add_scalar('F1/validation', val_f1, step+len(tr_loader)*epoch)
        val_auc = roc_auc_score(labels, probs)
        writer.add_scalar('ROC_AUC/validation', val_auc, step+len(tr_loader)*epoch)

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
    LOG_DIR = parse_path(args.log_dir)
    create_dir(LOG_DIR)

    writer = SummaryWriter(log_dir=LOG_DIR)
    train_dataset = GraphDataset(node_dir=os.path.join(args.node_dir,'train'), max_dist=200, max_degree=10)
    train_dataloader = GraphDataLoader(train_dataset, batch_size=20, shuffle=True)
    val_dataset = GraphDataset(node_dir=os.path.join(args.node_dir, 'validation'), max_dist=200, max_degree=10)
    val_dataloader = GraphDataLoader(val_dataset, batch_size=1, shuffle=True)
    model = GCN(NUM_FEATS, HIDDEN_FEATS, NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train(train_dataloader, val_dataloader, model, optimizer, writer, args.epochs)
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
from dgl.nn import GraphConv
from read_graph import GraphDataset
import argparse

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats, allow_zero_in_degree=True)
        self.conv2 = GraphConv(h_feats, num_classes, allow_zero_in_degree=True)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # best_val_acc = 0
    # best_test_acc = 0

    features = g.ndata['X']
    labels = g.ndata['y']
    for e in range(50):
        # Forward
        logits = model(g, features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = F.cross_entropy(logits, labels)

        # Compute accuracy on training/validation/test
        train_acc = (pred == labels).float().mean()
        """val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc"""

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print('In epoch {}, loss: {:.3f}, train accuracy {:.3f}'.format(
                e, loss, train_acc))

parser = argparse.ArgumentParser()
parser.add_argument('--node_dir', type=str, required=True,
                     help="Path to .nodes.csv files.")

if __name__=='__main__':   
    args = parser.parse_args()
    NUM_CLASSES = 2
    NUM_FEATS = 18
    HIDDEN_FEATS = 16
    graph_dataset = GraphDataset(node_dir=args.node_dir, max_dist=200, max_degree=10)
    train_dataloader = GraphDataLoader(graph_dataset, batch_size=2, shuffle=True)
    model = GCN(NUM_FEATS, HIDDEN_FEATS, NUM_CLASSES)
    for g in train_dataloader:
        train(g, model)
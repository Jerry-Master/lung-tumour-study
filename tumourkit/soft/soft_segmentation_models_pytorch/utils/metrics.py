from . import base
from . import functional as F
from ..base.modules import Activation
import cv2
from . import new_metrics as func
import numpy as np
import torch
from matplotlib import pyplot as plt

class IoU(base.Metric):
    __name__ = 'iou_score'

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.iou(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Fscore(base.Metric):

    def __init__(self, beta=1, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.f_score(
            y_pr, y_gt,
            eps=self.eps,
            beta=self.beta,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Accuracy(base.Metric):

    def __init__(self, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.accuracy(
            y_pr, y_gt,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Recall(base.Metric):

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.recall(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Precision(base.Metric):

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.precision(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


def CustomMetrics(y_pr,y_gt, threshold, activation):
    y_gt_array = y_gt.squeeze().cpu().detach().numpy()
    
#     y_pr = activation(y_pr)
    y_pr_array = y_pr.squeeze().cpu().detach().numpy()
    
    maxima_gt, lst_gt = func.find_local_maxima(y_gt_array, threshold)
    maxima_pr, lst_pr = func.find_local_maxima(y_pr_array, threshold)
    if lst_pr==[]:
        return [0,0,0]
    
    prec,rec, fscore = func.compare_binary(lst_pr, lst_gt)
    return [prec,rec,fscore]


class CustomAll(base.Metric):

    def __init__(self, eps=1e-7, threshold=0.15, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        sum_precs = 0
        sum_recs = 0
        sum_fscores = 0
        
        dims = y_pr.size()[0]
        for d in range(dims):
            prec, rec, fscore = CustomMetrics(y_pr[d],y_gt[d],self.threshold,self.activation)
            sum_precs = sum_precs + prec
            sum_recs = sum_recs + rec
            sum_fscores = sum_fscores + fscore
        mean_precs = sum_precs/dims
        mean_recs = sum_recs/dims
        mean_fscores = sum_fscores/dims

        return torch.tensor([mean_precs, mean_recs, mean_fscores], device='cuda:0')


class CustomFscore(base.Metric):

    def __init__(self, eps=1e-7, threshold=0.15, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        sum_fscores = 0
        dims = y_pr.size()[0]
        for d in range(dims):
            fscore = CustomMetrics(y_pr[d],y_gt[d],self.threshold,self.activation)[2]
            sum_fscores = sum_fscores + fscore
        mean_fscores = sum_fscores/dims

        return torch.tensor(mean_fscores, device='cuda:0')
        
class CustomPrecision(base.Metric):

    def __init__(self, eps=1e-7, threshold=0.15, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        sum_precs = 0
        dims = y_pr.size()[0]
        for d in range(dims):
            prec = CustomMetrics(y_pr[d],y_gt[d],self.threshold,self.activation)[0]
            sum_precs = sum_precs + prec
        mean_precs = sum_precs/dims
        
        return torch.tensor(mean_precs, device='cuda:0')

class CustomRecall(base.Metric):

    def __init__(self, eps=1e-7, threshold=0.15, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):        
        sum_recs = 0
        dims = y_pr.size()[0]
        for d in range(dims):
            rec = CustomMetrics(y_pr[d],y_gt[d],self.threshold,self.activation)[1]
            sum_recs = sum_recs + rec
        mean_recs = sum_recs/dims
        
        return torch.tensor(mean_recs, device='cuda:0')
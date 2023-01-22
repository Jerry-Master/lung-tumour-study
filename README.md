# lung-tumour-study
[![Python application](https://github.com/Jerry-Master/lung-tumour-study/actions/workflows/python-app.yml/badge.svg)](https://github.com/Jerry-Master/lung-tumour-study/actions/workflows/python-app.yml) ![pyversion](logos/py_version.svg) ![torchversion](logos/torch_version.svg) ![dglversion](logos/dgl_version.svg)

Github repository for my Bachelor's thesis. It is dedicated to study lung tumour through WSI.

## Dataset (v1)

The dataset consists of a set of labelled WSI tiles of lung cancer. The labels are done at x40 magnification and are done pixel-wise. It is therefore a segmentation and nuclei classification dataset. Right now it has 44 images manually labelled by me, with the help of Hovernet.

## Baseline (Soft & Hovernet)

The initial baseline is set by Hovernet's model trained with the first version of the dataset. I also tried the soft algorithm developed by Digipatics with far worse results.

## Improvements (Doing)

Current line of research right now is about applying Machine Learning with Graphs. It has yet to be decided the way to construct the graph and the design space of the models to use.

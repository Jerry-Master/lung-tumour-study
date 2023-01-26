# Graphs-based tumoural cell classification

In order to improve Hovernet results, the usage of graph neural networks is proposed. This module contains scripts to train GNNs on top of previous predictions from computer vision methods. Apart from that, here is the code for several interesting experiments and visualizations.

## Node-only classification

To see if using graph neural networks is useful, we need something to compare it too. The comparison with computer vision methods is quite unfair since they operate on distinct realms. For that reason, GNN are compared to methods that only use the node attributes without any kind of edge information. By achieving greater results than those methods the expressivity of GNNs are thus proved.

### Description

The methods used are xgboost and automl (bayesian optimization). The xgboost model is trained under different configurations using cross-validation to obtain the metrics for the best configuration. The automl is left for five hours to find the best model.

### Usage

To obtain the results for node-only methods, run `train_xgboost.py` and `train_automl.py`. The input format is the one described in preprocessing module for `.nodes.csv` files. Output format is csv for the results and pickle for the model of the automl script. The xgboost doesn't save any model.

## Heterogeneity

To measure how heterogeneus the cell attributes are, the automl method is trained under two different split, one where nodes from one image belong to the same split, and another where nodes from the same image can appear in train and test. This way, if the model trained under the second split gives much better results, it implies that the cell attributes are not homogeneous.

## Graph Neural Networks

In order to achieve the bests results, different models are tried, constituting what I call a graph zoo. Right now the zoo contains graph convolutional neural network, graph attention and graph hard attention. I am planning to also include GraphSAGE and Boost the Convolve. Apart from different architectures there are also different normalization schemes, including batch normalization and instance normalization. However, the implementation of the latter seems wrong based on empirical results. Finally, different levels of dropout are tried too. 

### Usage

The script `train_gnn.py` performs several training in parallel for a given architecture varying the number of layers, the quantity of dropout and the normalization scheme.

The script `infer.py` appends the column `prob1` to the `.nodes.csv` which correspond to the prediction of the given model.

The script `evaluate.py` computes the metrics from the output of the `infer.py` script and the ground truth. It also creates reliability diagrams to visualize how calibrated the model is. For the latter use the `--draw` flag.

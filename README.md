Workload Performance Prediction
====

We propose a performance prediction system for concurrent queries using a graph embedding based model. We first propose a graph model to encode query features, where each vertex is a node in the query plan of a query and each edge between two vertices denotes the correlations between them, e.g., sharing the same table/index or competing resources. We then propose a prediction model, in which we use a graph embedding network to encode the graph features and adopt a prediction network to predict query performance using deep learning.

## Before you run

1. Install dependencies.

```
pip install -r requirements.txt
```

2. Download the dataset from https://cloud.tsinghua.edu.cn/f/b6f4e92ba387445cb825/  (pmodel_data.zip), and put the unzip directory to the current main path (./pmodel_data)

## Usage

1. Run ``main.py''

**Note**: Turn on the hyperparameter (no_upd=True) if you want to test the graph-update performance.

2. Compact large graphs using ``graph-generation-merged.py''


## Cite

Please cite our paper if you find this work interesting:

```
@article{DBLP:journals/pvldb/ZhouSLF20,
  author    = {Xuanhe Zhou and
               Ji Sun and
               Guoliang Li and
               Jianhua Feng},
  title     = {Query Performance Prediction for Concurrent Queries using Graph Embedding},
  journal   = {Proc. {VLDB} Endow.},
  pages     = {1416--1428},
  year      = {2020},
}
```

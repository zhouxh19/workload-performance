Workload Performance Prediction
====

We propose a performance prediction system for concurrent queries using a graph embedding based model. We first propose a graph model to encode query features, where each vertex is a node in the query plan of a query and each edge between two vertices denotes the correlations between them, e.g., sharing the same table/index or competing resources. We then propose a prediction model, in which we use a graph embedding network to encode the graph features and adopt a prediction network to predict query performance using deep learning.

## Before you run

Install dependencies.

```
pip install -r requirements.txt
```

## Usage

1. Unzip pmodel_data.zip

2. Open up performance-graphembedding-checkpoint.ipynb, and reproduce by following the steps

**Note**: You can jump over the workload and graph generation procedures (which need to connect to your own database), and direct run the prediction part with the prepared data (in ./pmodel_data).

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
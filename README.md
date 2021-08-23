Workload Performance Prediction
====

We propose a performance prediction system for concurrent queries using a graph embedding based model. We first propose a graph model to encode query features, where each vertex is a node in the query plan of a query and each edge between two vertices denotes the correlations between them, e.g., sharing the same table/index or competing resources. We then propose a prediction model, in which we use a graph embedding network to encode the graph features and adopt a prediction network to predict query performance using deep learning.

## Before you run

Install dependencies.

```
pip install -r requirements.txt
```

## Usage

```python3 train.py --epochs=500 --lr=0.01```

## Results

![train](/Users/xuanhe/Documents/our-paper/workload-performance/code.log/workload-performance/photo/train.png)



![test](/Users/xuanhe/Documents/our-paper/workload-performance/code.log/workload-performance/photo/test.png)

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

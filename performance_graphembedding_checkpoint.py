#!/usr/bin/env python
# coding: utf-8

# In[1]:
from __future__ import division
from __future__ import print_function

import json
import numpy as np
import os

import configparser
import psycopg2
import pymysql
import pymysql.cursors as pycursor
import numpy as np

import time
import glob
from constants import NODE_DIM, args

# # 1. Generate Workload Dataset

# In[2]:


# cur_path = os.path.abspath('.')
# data_path = cur_path + '/pmodel_data/job/'

# edge_dim = 100000  # upper bound of edges
# node_dim = 1000  # upper bound of nodes

'''
class DataType(IntEnum):
    Aggregate = 0
    NestedLoop = 1
    IndexScan = 2
'''
# mp_optype = {'Aggregate': 0, 'Nested Loop': 1, 'Index Scan': 2, 'Hash Join': 3, 'Seq Scan': 4, 'Hash': 5,
#              'Update': 6}  # operator types in the queries

# oid = 0  # operator number
# min_timestamp = -1  # minimum timestamp of a graph

'''
argus = { "mysql": {
    "host": "166.111.121.62",
    "password": "db10204",
    "port": 3306,
    "user": "feng"},
    "postgresql": {
            "host": "166.111.121.62",
            "password": "db10204",
            "port": 5433,
            "user": "postgres"}}
argus["postgresql"]["host"]
'''


# In[3]:


# obtain and normalize configuration knobs
from dbconnnection import *

# db = Database("mysql")
# print(db.fetch_knob())


# In[4]:


# actual runtime:  actuall executed (training data) / estimated by our model
# operators in the same plan can have data conflicts (parallel)
from nodeutils import *


# import merge


'''
def generate_graph(wid, path=data_path, mp_optype=None):
    if mp_optype is None:
        mp_optype = {'Aggregate': 0, 'Nested Loop': 1, 'Index Scan': 2, 'Hash Join': 3, 'Seq Scan': 4, 'Hash': 5,
                     'Update': 6}
    # global oid, min_timestamp # write only.
    # fuction
    # return
    # todo: timestamp

    vmatrix = []
    ematrix = []
    mergematrix = []
    conflict_operators = {}

    oid = 0
    min_timestamp = -1
    with open(path + "sample-plan-" + str(wid) + ".txt", "r") as f:
        # vertex: operators
        # edge: child-parent relations
        for sample in f.readlines():
            sample = json.loads(sample)

            # Step 1: read (operators, parent-child edges) in separate plans
            start_time, node_matrix, edge_matrix, conflict_operators, node_merge_matrix, mp_optype, oid, min_timestamp  = \
                extract_plan(sample, conflict_operators, mp_optype, oid, min_timestamp) # warning : may cause probs.
            mergematrix = mergematrix + node_merge_matrix
            vmatrix = vmatrix + node_matrix
            ematrix = ematrix + edge_matrix

        # ZXN TEMP Modified BEGIN
        # Step 2: read related knobs
        db = Database("mysql")
        knobs = db.fetch_knob()

        # Step 3: add relations across queries
        ematrix = add_across_plan_relations(conflict_operators, knobs, ematrix)

        # edge: data relations based on (access tables, related knob values)
        # vmatrix, ematrix = merge.mergegraph_main(mergematrix, ematrix, vmatrix)
    ### ZXN TEMP Modified ENDED
    return vmatrix, ematrix, mergematrix

'''

'''
graphs = glob.glob("./pmodel_data/job/graph/sample-plan-*")
num_graphs = int(len(graphs)/2)
print("[Generated Graph]", num_graphs)
'''

# # Graph Embedding Algorithm

from graphembedding import *

import torch.nn.functional as F
# adj, features, labels, idx_train, idx_val, idx_test =
# load_data(path = r"C:\Users\Filene\Downloads\workload-performance-main\workload-performance-main\pmodel_data\job\graph\sample-plan-", dataset = "0")
import random

# In[10]:


'''
x = np.asarray([[1, 2], [3, 4]])
X = torch.Tensor(x)
print(X.shape)
pad_dims = (1, 3)
X = F.pad(X, pad_dims, "constant")
print(X)
print(X.shape[0])
'''

# ## GCN Model
from GCN import *

# In[15]:
from train import run_train_no_upd, run_test_no_upd, run_train_upd, run_test_upd

if __name__ == "__main__":
    no_upd = False
    if no_upd:
        iteration_num, num_graphs, model = run_train_no_upd(demo=True)
        run_test_no_upd(iteration_num, num_graphs, model)
    # iteration_num, num_graphs, model = run_train_no_upd(demo=False)
    else:
        num_graphs, come_num, model, adj, vmatrix, ematrix, mp_optype, oid, min_timestamp = run_train_upd(demo=True)
        run_test_upd(num_graphs, come_num, model, adj, vmatrix, ematrix, mp_optype, oid, min_timestamp)


# In[16]:

# assume num_graphs >> come_num
'''
num_graphs = 4
come_num = 1

graphs = glob.glob("./pmodel_data/job/sample-plan-*")
# num_graphs = len(graphs)

# train model on a big graph composed of graph_num samples
vmatrix = []
ematrix = []
feature_num = 3
conflict_operators = {}

for wid in range(num_graphs):
    with open(DATAPATH + "/sample-plan-" + str(wid) + ".txt", "r") as f:

        for sample in f.readlines():
            sample = json.loads(sample)

            start_time, node_matrix, edge_matrix, conflict_operators, _, mp_optype, oid, min_timestamp = \
                extract_plan(sample, conflict_operators, mp_optype, oid, min_timestamp)

            vmatrix = vmatrix + node_matrix
            ematrix = ematrix + edge_matrix

    db = Database("mysql")
    knobs = db.fetch_knob()
    ematrix = add_across_plan_relations(conflict_operators, knobs, ematrix)


# TODO more features, more complicated model
model = get_model(feature_num=feature_num, hidden=args.hidden,nclass=NODE_DIM,dropout=args.dropout)
optimizer = get_optimizer(model=model,lr=args.lr,weight_decay=args.weight_decay)
adj, features, labels, idx_train, idx_val, idx_test = load_data_from_matrix(np.array(vmatrix, dtype=np.float32),
                                                                            np.array(ematrix, dtype=np.float32))

ok_times = 0
for epoch in range(args.epochs):
    # print(features.shape, adj.shape)
    loss_train = train(epoch, labels, features, adj, idx_train, idx_val, model=model, optimizer=optimizer)
    if loss_train < 0.002:
        ok_times += 1
    if ok_times >= 20:
        break

test(labels, idx_test, features, adj, model)
'''

'''
def predict(labels, features, adj, dh):
    model.eval()
    output = model(features, adj, dh)
    loss_test = F.mse_loss(output, labels)
    acc_test = accuracy(output, labels)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()))


import bisect

# new queries( come_num samples ) come
new_e = []
conflict_operators = {}
phi = []
for wid in range(num_graphs, num_graphs + come_num):

    with open(data_path + "sample-plan-" + str(wid) + ".txt", "r") as f:

        # new query come
        for sample in f.readlines():

            # updategraph-add
            sample = json.loads(sample)

            start_time, node_matrix, edge_matrix, conflict_operators, _, mp_optype, oid, min_timestamp = \
                extract_plan(sample, conflict_operators, mp_optype, oid, min_timestamp)

            vmatrix = vmatrix + node_matrix
            new_e = new_e + edge_matrix

            db = Database("mysql")
            knobs = db.fetch_knob()

            new_e = add_across_plan_relations(conflict_operators, knobs, new_e)

            # incremental prediction
            dadj, dfeatures, dlabels, _, _, _ = load_data_from_matrix(np.array(vmatrix, dtype=np.float32),
                                                                      np.array(new_e, dtype=np.float32))

            model.eval()
            dh = model(dfeatures, dadj, None, True)

            predict(dlabels, dfeatures, adj, dh)

            for node in node_matrix:
                bisect.insort(phi, [node[-2] + node[-1], node[0]])

            # updategraph-remove
            num = bisect.bisect(phi, [start_time, -1])
            if num > 20: # ZXN: k = 20, num > k.
                rmv_phi = [e[1] for e in phi[:num]]
                phi = phi[num:]
                vmatrix = [v for v in vmatrix if v[0] not in rmv_phi]
                new_e = [e for e in new_e if e[0] not in rmv_phi and e[1] not in rmv_phi]
                for table in conflict_operators:
                    conflict_operators[table] = [v for v in conflict_operators[table] if v[0] not in rmv_phi]
'''

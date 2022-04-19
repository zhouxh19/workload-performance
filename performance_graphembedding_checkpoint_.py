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

# # 1. Generate Workload Dataset

# In[2]:


cur_path = os.path.abspath('.')
data_path = os.path.join(cur_path, "pmodel_data", "job")

edge_dim = 100000  # upper bound of edges
node_dim = 1000  # upper bound of nodes

'''
class DataType(IntEnum):
    Aggregate = 0
    NestedLoop = 1
    IndexScan = 2
'''
mp_optype = {'Aggregate': 0, 'Nested Loop': 1, 'Index Scan': 2, 'Hash Join': 3, 'Seq Scan': 4, 'Hash': 5,
             'Update': 6}  # operator types in the queries

oid = 0  # operator number
min_timestamp = -1  # minimum timestamp of a graph

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

class DictParser(configparser.ConfigParser):
    def read_dict(self):
        d = dict(self._sections)
        for k in d:
            d[k] = dict(d[k])
        return d


cf = DictParser()
cf.read("config.ini", encoding="utf-8")
config_dict = cf.read_dict()


def parse_knob_config():
    _knob_config = config_dict["knob_config"]
    for key in _knob_config:
        _knob_config[key] = json.loads(str(_knob_config[key]).replace("\'", "\""))
    return _knob_config


class Database:
    def __init__(self, server_name='postgresql'):

        knob_config = parse_knob_config()
        self.knob_names = [knob for knob in knob_config]
        self.knob_config = knob_config
        self.server_name = server_name

        # print("knob_names:", self.knob_names)

        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            sql = "SELECT count FROM INFORMATION_SCHEMA.INNODB_METRICS where status='enabled'"
            cursor.execute(sql)
            result = cursor.fetchall()

            self.internal_metric_num = len(result)
            cursor.close()
            conn.close()
        except Exception as err:
            print("execute sql error:", err)

    def _get_conn(self):
        if self.server_name == 'mysql':
            sucess = 0
            conn = -1
            count = 0
            while not sucess and count < 3:
                try:
                    conn = pymysql.connect(host="166.111.121.62",
                                           port=3306,
                                           user="feng",
                                           password="db10204",
                                           db='INFORMATION_SCHEMA',
                                           connect_timeout=36000,
                                           cursorclass=pycursor.DictCursor)

                    sucess = 1
                except Exception as result:
                    count += 1
                    time.sleep(10)
            if conn == -1:
                raise Exception

            return conn

        elif self.server_name == 'postgresql':
            sucess = 0
            conn = -1
            count = 0
            while not sucess and count < 3:
                try:
                    db_name = "INFORMATION_SCHEMA"  # zxn Modified.
                    conn = psycopg2.connect(database="INFORMATION_SCHEMA", user="lixizhang", password="xi10261026zhang",
                                            host="166.111.5.177", port="5433")
                    sucess = 1
                except Exception as result:
                    count += 1
                    time.sleep(10)
            if conn == -1:
                raise Exception
            return conn

        else:
            print('数据库连接不上...')
            return

    def fetch_knob(self):
        state_list = np.append([], [])
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            sql = "select"
            for i, knob in enumerate(self.knob_names):
                sql = sql + ' @@' + knob

                if i < len(self.knob_names) - 1:
                    sql = sql + ', '

            # state metrics
            cursor.execute(sql)
            result = cursor.fetchall()

            for i in range(len(self.knob_names)):
                value = result[0]["@@%s" % self.knob_names[i]] if result[0]["@@%s" % self.knob_names[i]] != 0 else \
                self.knob_config[self.knob_names[i]]["max_value"]  # not limit if value equals 0

                # print(value, self.knob_config[self.knob_names[i]]["max_value"], self.knob_config[self.knob_names[i]]["min_value"])
                state_list = np.append(state_list, value / (
                            self.knob_config[self.knob_names[i]]["max_value"] - self.knob_config[self.knob_names[i]][
                        "min_value"]))
            cursor.close()
            conn.close()
        except Exception as error:
            print("fetch_knob Error:", error)

        return state_list


# db = Database("mysql")
# print(db.fetch_knob())


# In[4]:


# actual runtime:  actuall executed (training data) / estimated by our model
# operators in the same plan can have data conflicts (parallel)

def compute_cost(node):
    return (float(node["Total Cost"]) - float(node["Startup Cost"])) / 1e6


def compute_time(node):
    # return float(node["Actual Total Time"]) - float(node["Actual Startup Time"])
    return float(node["Actual Total Time"])  # mechanism within pg


def get_used_tables(node):
    tables = []

    stack = [node]
    while stack != []:
        parent = stack.pop(0)

        if "Relation Name" in parent:
            tables.append(parent["Relation Name"])

        if "Plans" in parent:
            for n in parent["Plans"]:
                stack.append(n)

    return tables


def extract_plan(sample, conflict_operators):
    global mp_optype, oid, min_timestamp
    if min_timestamp < 0:
        min_timestamp = float(sample["start_time"])
        start_time = 0
    else:
        start_time = float(sample["start_time"]) - min_timestamp
    # function: extract SQL feature
    # return: start_time, node feature, edge feature

    plan = sample["plan"]
    while isinstance(plan, list):
        plan = plan[0]
    # Features: print(plan.keys())
    # start time = plan["start_time"]
    # node feature = [Node Type, Total Cost:: Actual Total Time]
    # node label = [Actual Startup Time, Actual Total Time]

    plan = plan["Plan"]  # root node
    node_matrix = []
    edge_matrix = []
    node_merge_matrix = []

    # add oid for each operator
    stack = [plan]
    while stack != []:
        parent = stack.pop(0)
        parent["oid"] = oid
        oid = oid + 1

        if "Plans" in parent:
            for node in parent["Plans"]:
                stack.append(node)

    stack = [plan]
    while stack != []:
        parent = stack.pop(0)
        run_cost = compute_cost(parent)
        run_time = compute_time(parent)
        # print(parent["Actual Total Time"], parent["Actual Startup Time"], run_time)

        if parent["Node Type"] not in mp_optype:
            mp_optype[parent["Node Type"]] = len(mp_optype)

        tables = get_used_tables(parent)
        # print("[tables]", tables)

        operator_info = [parent["oid"], start_time + parent["Startup Cost"] / 1e6,
                         start_time + parent["Total Cost"] / 1e6]

        for table in tables:
            if table not in conflict_operators:
                conflict_operators[table] = [operator_info]
            else:
                conflict_operators[table].append(operator_info)

        node_feature = [parent["oid"], mp_optype[parent["Node Type"]], run_cost,
                        start_time + float(parent["Actual Startup Time"]), run_time]

        node_matrix = [node_feature] + node_matrix

        node_merge_feature = [parent["oid"], start_time + parent["Startup Cost"] / 1e6,
                              start_time + parent["Total Cost"] / 1e6, mp_optype[parent["Node Type"]], run_cost,
                              start_time + float(parent["Actual Startup Time"]), run_time]
        node_merge_matrix = [node_merge_feature] + node_merge_matrix
        # [id?, l, r, ....]

        if "Plans" in parent:
            for node in parent["Plans"]:
                stack.append(node)
                edge_matrix = [[node["oid"], parent["oid"], 1]] + edge_matrix

    # node: 18 * featuers
    # edge: 18 * 18

    return start_time, node_matrix, edge_matrix, conflict_operators, node_merge_matrix


def overlap(node_i, node_j):
    if (node_j[1] < node_i[2] and node_i[2] < node_j[2]):

        return (node_i[2] - node_j[1]) / (node_j[2] - min(node_i[1], node_j[1]))

    elif (node_i[1] < node_j[2] and node_j[2] < node_i[2]):

        return (node_j[2] - node_i[1]) / (node_i[2] - min(node_i[1], node_j[1]))

    else:
        return 0


def add_across_plan_relations(conflict_operators, knobs, ematrix):
    # TODO better implementation
    data_weight = 0.1
    for knob in knobs:
        data_weight *= knob
    # print(conflict_operators)

    # add relations [rw/ww, rr, config]
    for table in conflict_operators:
        for i in range(len(conflict_operators[table])):
            for j in range(i + 1, len(conflict_operators[table])):

                node_i = conflict_operators[table][i]
                node_j = conflict_operators[table][j]

                time_overlap = overlap(node_i, node_j)
                if time_overlap:
                    ematrix = ematrix + [[node_i[0], node_j[0], -data_weight * time_overlap]]
                    ematrix = ematrix + [[node_j[0], node_i[0], -data_weight * time_overlap]]

                '''
                if overlap(i, j) and ("rw" or "ww"):
                    ematrix = ematrix + [[conflict_operators[table][i], conflict_operators[table][j], data_weight * time_overlap]]
                    ematrix = ematrix + [[conflict_operators[table][j], conflict_operators[table][i], data_weight * time_overlap]]
                '''

    return ematrix


import merge


def generate_graph(wid, path=data_path):
    global oid, min_timestamp
    # fuction
    # return
    # todo: timestamp

    vmatrix = []
    ematrix = []
    mergematrix = []
    conflict_operators = {}

    oid = 0
    min_timestamp = -1
    with open( os.path.join(path, "sample-plan-" + str(wid) + ".txt"), "r") as f:
        # vertex: operators
        # edge: child-parent relations
        for sample in f.readlines():
            sample = json.loads(sample)

            # Step 1: read (operators, parent-child edges) in separate plans
            start_time, node_matrix, edge_matrix, conflict_operators, node_merge_matrix = extract_plan(sample,
                                                                                                       conflict_operators)

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


# In[5]:


# Step-0: split the workloads into multiple concurrent queries at different time ("sample-plan-x")

workloads = glob.glob("./pmodel_data/job/sample-plan-*")

start_time = time.time()
num_graphs = 3000
#for wid in range(num_graphs):
#    st = time.time()

#    vmatrix, ematrix, mergematrix = generate_graph(wid)
    # vmatrix, ematrix = merge.mergegraph_main(mergematrix, ematrix, vmatrix)
#    print("[graph {}]".format(wid), "time:{}; #-vertex:{}, #-edge:{}".format(time.time() - st, len(vmatrix), len(ematrix)))

### ZXN TEMP Modified BEGIN
#    with open(data_path + "graph/" + "sample-plan-" + str(wid) + ".content", "w") as wf:
#       for v in vmatrix:
#           wf.write(str(v[0]) + "\t" + str(v[1]) + "\t" + str(v[2]) + "\t" + str(v[3]) + "\t" + str(v[4]) + "\n")
#    with open(data_path + "graph/" + "sample-plan-" + str(wid) + ".cites", "w") as wf:
#       for e in ematrix:
#           wf.write(str(e[0]) + "\t" + str(e[1]) + "\t" + str(e[2]) + "\n")
### ZXN TEMP Modified ENDED

end_time = time.time()

print("Total Time:{}".format(end_time - start_time))

# In[6]:



graphs = glob.glob("./pmodel_data/job/graph/sample-plan-*")
num_graphs = int(len(graphs)/2)
print("[Generated Graph]", num_graphs)


# # Graph Embedding Algorithm

# In[7]:


import numpy as np
import scipy.sparse as sp
import torch


# ## Load Data

# In[8]:


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# In[9]:


import torch.nn.functional as F


def load_data(dataset, path=data_path):
    print('Loading {} dataset...'.format(dataset))

    vmatrix = np.genfromtxt("{}.content".format( os.path.join(path, dataset)),
                            dtype=np.dtype(str))

    ematrix = np.genfromtxt("{}.cites".format( os.path.join(path, dataset)),
                            dtype=np.float32)

    return load_data_from_matrix(vmatrix, ematrix)


# adj, features, labels, idx_train, idx_val, idx_test =
# load_data(path = r"C:\Users\Filene\Downloads\workload-performance-main\workload-performance-main\pmodel_data\job\graph\sample-plan-", dataset = "0")
import random


def load_data_from_matrix(vmatrix, ematrix):
    idx_features_labels = vmatrix

    # encode vertices
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)

    # encode labels
    # labels = encode_onehot(idx_features_labels[:, -2])
    labels = idx_features_labels[:, -1].astype(float)

    # encode edges
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)

    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = ematrix[:, :-1]

    # print(list(map(idx_map.get, edges_unordered.flatten())))
    # print(edges_unordered.flatten())

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    # edges (weights are computed in gcn)

    # modified begin.
    edges_value = ematrix[:, -1:]
    # modified end.

    # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(node_dim, node_dim),dtype=np.float32)
    # print("old_adj = ", adj)
    adj = sp.coo_matrix((edges_value[:, 0], (edges[:, 0], edges[:, 1])), shape=(node_dim, node_dim), dtype=np.float32)
    # print("new_adj = ", adj)
    # print(adj.shape)

    # build symmetric adjacency matrix
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    operator_num = adj.shape[0]
    idx_train = range(int(0.8 * operator_num))
    # print("idx_train", idx_train)
    idx_val = range(int(0.8 * operator_num), int(0.9 * operator_num))
    idx_test = range(int(0.9 * operator_num), int(operator_num))

    features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # padding to the same size
    # print(features.shape)
    # print(node_dim - features.shape[0])
    dim = (0, 0, 0, node_dim - features.shape[0])
    features = F.pad(features, dim, "constant", value=0)

    labels = labels.astype(np.float32)
    labels = torch.from_numpy(labels)
    # print(labels[idx_train].dtype)
    labels.unsqueeze(1)
    labels = labels * 10000
    labels = F.pad(labels, [0, node_dim - labels.shape[0]], "constant", value=0)

    # print("features", features.shape)
    return adj, features, labels, idx_train, idx_val, idx_test


# In[10]:


import torch.nn.functional as F

x = np.asarray([[1, 2], [3, 4]])
X = torch.Tensor(x)
print(X.shape)
pad_dims = (1, 3)
X = F.pad(X, pad_dims, "constant")
print(X)
print(X.shape[0])


# ## GCN Model

# In[11]:


class arguments():
    def __init__(self):
        self.cuda = True
        self.fastmode = False
        self.seed = 42
        self.epochs = 200
        self.lr = 0.01
        self.weight_decay = 5e-4
        #self.hidden = 16
        self.hidden = 12
        self.dropout = 0.5


args = arguments()

# In[12]:


from pathlib import Path

print(Path().resolve())

# In[13]:


import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# In[14]:


import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.fc = nn.Linear(nclass, 1)
        self.dropout = dropout

    def forward(self, x, adj, dh=None, embed=False):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        if embed:
            return x
        if dh is not None:
            x = x + dh
        x = self.fc(x)

        #        return F.log_softmax(x, dim=1)
        return x


# In[15]:


import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim


def train(epoch, labels, features, adj, idx_train):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    # print(output[idx_train])
    # print("output = !",output,"labels = !", labels)
    loss_train = F.mse_loss(output[idx_train], labels[idx_train])

    # loss_train = nn.CrossEntropyLoss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()


    if epoch == 0:
        print("---------------------")
        for name, paras in model.named_parameters():
            TUP = (name, paras.grad)
            print("name:",name, "parameters_value:",paras)
            print("-->grad_requirs:", paras.requires_grad, "-->grad_value", paras.grad)
        print("______________________")
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)
        # transfer output to ms
        # output = output * 1000
    # https://www.cnblogs.com/52dxer/p/13793911.html
    loss_val = F.mse_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return round(loss_train.item(), 4)


def test(labels, idx_test):
    model.eval()
    output = model(features, adj)
    # transfer output to ms
    # output = output * 1000
    loss_test = F.mse_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()))


# Step-3:
feature_num = 3
# num_graphs = 10
graphs = glob.glob("./pmodel_data/job/sample-plan-*")
num_graphs = len(graphs)
iteration_num = int(round(0.8 * num_graphs, 0))
print("[training samples]:{}".format(iteration_num))

model = GCN(nfeat=feature_num,
            nhid=args.hidden,
            nclass=node_dim,
            dropout=args.dropout)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

for wid in range(iteration_num):
    print("[graph {}]".format(wid))
    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = load_data(path=os.path.join(data_path,"graph"),
                                                                    dataset="sample-plan-" + str(wid))
    # print(adj.shape)

    # Model Training
    ok_times = 0
    t_total = time.time()
    for epoch in range(args.epochs):
        # print(features.shape, adj.shape)
        loss_train = train(epoch, labels, features, adj, idx_train)
        if loss_train < 0.002:
            ok_times += 1
        if ok_times >= 20:
            break

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Model Validation
    test(labels, idx_test)

for wid in range(iteration_num, num_graphs):
    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = load_data(path=os.path.join(data_path,"graph"),
                                                                    dataset="sample-plan-" + str(wid))

    # Model Testing
    t_total = time.time()
    test(labels, idx_test)
    print("Testing Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# In[16]:

# assume graph_num >> come_num
graph_num = 4
come_num = 1

# train model on a big graph composed of graph_num samples
min_timestamp = -1
vmatrix = []
ematrix = []
conflict_operators = {}

for wid in range(graph_num):

    with open( os.path.join(data_path,"sample-plan-" + str(wid) + ".txt"), "r") as f:

        for sample in f.readlines():
            sample = json.loads(sample)

            start_time, node_matrix, edge_matrix, conflict_operators, _ = extract_plan(sample, conflict_operators)

            vmatrix = vmatrix + node_matrix
            ematrix = ematrix + edge_matrix

db = Database("mysql")
knobs = db.fetch_knob()
ematrix = add_across_plan_relations(conflict_operators, knobs, ematrix)

# TODO more features, more complicated model
model = GCN(nfeat=feature_num,
            nhid=args.hidden,
            nclass=node_dim,
            dropout=args.dropout)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

adj, features, labels, idx_train, idx_val, idx_test = load_data_from_matrix(np.array(vmatrix, dtype=np.float32),
                                                                            np.array(ematrix, dtype=np.float32))

ok_times = 0
for epoch in range(args.epochs):
    # print(features.shape, adj.shape)
    loss_train = train(epoch, labels, features, adj, idx_train)
    if loss_train < 0.002:
        ok_times += 1
    if ok_times >= 20:
        break

test(labels, idx_test)


def predict(labels, features, adj, dh):
    model.eval()
    output = model(features, adj, dh)
    loss_test = F.mse_loss(output, labels)
    acc_test = accuracy(output, labels)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()))


import bisect

# new queries( come_num samples ) come
k = 20
new_e = []
conflict_operators = {}
phi = []
for wid in range(graph_num, graph_num + come_num):

    with open( os.path.join(data_path,"sample-plan-" + str(wid) + ".txt"), "r") as f:

        # new query come
        for sample in f.readlines():

            # updategraph-add
            sample = json.loads(sample)

            start_time, node_matrix, edge_matrix, conflict_operators, _ = extract_plan(sample, conflict_operators)

            vmatrix = vmatrix + node_matrix
            new_e = new_e + edge_matrix

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
            if num > k:
                rmv_phi = [e[1] for e in phi[:num]]
                phi = phi[num:]
                vmatrix = [v for v in vmatrix if v[0] not in rmv_phi]
                new_e = [e for e in new_e if e[0] not in rmv_phi and e[1] not in rmv_phi]
                for table in conflict_operators:
                    conflict_operators[table] = [v for v in conflict_operators[table] if v[0] not in rmv_phi]

# In[ ]:





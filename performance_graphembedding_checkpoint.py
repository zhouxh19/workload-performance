#!/usr/bin/env python
# coding: utf-8

from __future__ import division
from __future__ import print_function

import json
import os
import configparser
import psycopg2
import pymysql
import pymysql.cursors as pycursor

import time
import glob


cur_path = os.path.abspath('.')
data_path = os.path.join(cur_path,"pmodel_data","job")

edge_dim = 100000 # upper bound of edges
node_dim = 1000 # upper bound of nodes

'''
class DataType(IntEnum):
    Aggregate = 0
    NestedLoop = 1
    IndexScan = 2
'''
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

# obtain and normalize configuration knobs
class DictParser(configparser.ConfigParser):
    def read_dict(self):
        d = dict(self._sections)
        for k in d:
            d[k] = dict(d[k])
        return d

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

oid = 0 # operator number
min_timestamp = -1 # minimum timestamp of a graph
from extract_and_generate import extract_plan
from extract_and_generate import generate_graph
from extract_and_generate import add_across_plan_relations

cf = DictParser()
cf.read("config.ini", encoding="utf-8")
config_dict = cf.read_dict()

# db = Database("mysql")
# print(db.fetch_knob())

# Step-0: split the workloads into multiple concurrent queries at different time ("sample-plan-x")

workloads = glob.glob("./pmodel_data/job/sample-plan-*")

start_time = time.time()
# num_graphs = 3000
# zxn modified
# notation: oid may be unuseful.
num_graphs = 2
for wid in range(num_graphs):
    st = time.time()
    vmatrix, ematrix, mergematrix, oid, min_timestamp = generate_graph(wid, data_path)
    # optional: merge
    # vmatrix, ematrix = merge.mergegraph_main(mergematrix, ematrix, vmatrix)
    print("[graph {}]".format(wid), "time:{}; #-vertex:{}, #-edge:{}".format(time.time() - st, len(vmatrix), len(ematrix)))

    with open( os.path.join(data_path,"graph", "sample-plan-" + str(wid) + ".content"), "w") as wf:
       for v in vmatrix:
           wf.write(str(v[0]) + "\t" + str(v[1]) + "\t" + str(v[2]) + "\t" + str(v[3]) + "\t" + str(v[4]) + "\n")
    with open( os.path.join(data_path, "graph" , "sample-plan-" + str(wid) + ".cites"), "w") as wf:
       for e in ematrix:
           wf.write(str(e[0]) + "\t" + str(e[1]) + "\t" + str(e[2]) + "\n")

end_time = time.time()
print("Total Time:{}".format(end_time - start_time))

graphs = glob.glob("./pmodel_data/job/graph/sample-plan-*")
num_graphs = int(len(graphs)/2)
print("[Generated Graph]", num_graphs)


# # Graph Embedding Algorithm
import numpy as np

import torch
import torch.nn.functional as F

x=np.asarray([[1,2], [3, 4]])
X=torch.Tensor(x)
print(X.shape)
pad_dims = (1, 3)
X=F.pad(X,pad_dims,"constant")
print(X)
print(X.shape[0])


# GCN Model
class arguments():
    def __init__(self):
        self.cuda = True
        self.fastmode = False
        self.seed = 42
        self.epochs = 200
        self.lr = 0.01
        self.weight_decay = 5e-4
        self.hidden = 16
        self.dropout = 0.5
        
args = arguments()

from pathlib import Path
print(Path().resolve())

import math
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
        return self.__class__.__name__ + ' ('                + str(self.in_features) + ' -> '                + str(self.out_features) + ')'


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

import time
import argparse
import numpy as np
from dataloder import accuracy
from dataloder import load_data
from dataloder import load_data_from_matrix

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
    print('Epoch: {:04d}'.format(epoch+1),
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
num_graphs = 10
# graphs = glob.glob("./pmodel_data/job/sample-plan-*")
# num_graphs = len(graphs)
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
    adj, features, labels, idx_train, idx_val, idx_test = load_data(path = os.path.join(data_path,"graph"), dataset = "sample-plan-" + str(wid))
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
    adj, features, labels, idx_train, idx_val, idx_test = load_data(path = os.path.join(data_path, "graph/"), dataset = "sample-plan-" + str(wid))
    
    # Model Testing
    t_total = time.time()
    test(labels, idx_test)
    print("Testing Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


# assume graph_num >> come_num
graph_num = 4
come_num = 1

# train model on a big graph composed of graph_num samples
min_timestamp = -1
vmatrix = []
ematrix = [] 
conflict_operators = {}

for wid in range(graph_num):
    
    with open(data_path + "sample-plan-" + str(wid) + ".txt", "r") as f:    

        for sample in f.readlines():
            sample = json.loads(sample)
            
            start_time, node_matrix, edge_matrix, conflict_operators, _ , min_timestamp = extract_plan(sample, conflict_operators)
            
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

adj, features, labels, idx_train, idx_val, idx_test = load_data_from_matrix(np.array(vmatrix, dtype=np.float32), np.array(ematrix, dtype=np.float32))

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

    with open(data_path + "sample-plan-" + str(wid) + ".txt", "r") as f:
        
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
            dadj, dfeatures, dlabels, _, _, _ = load_data_from_matrix(np.array(vmatrix, dtype=np.float32), np.array(new_e, dtype=np.float32))

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


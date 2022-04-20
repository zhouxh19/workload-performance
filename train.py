import glob
import json
import os
import time
import bisect

import numpy as np
import torch.nn.functional as F
import torch.optim as optim

from GCN import GCN, get_model, get_optimizer
from constants import args, NODE_DIM, DATAPATH
from dbconnnection import Database
from graphembedding import load_data, accuracy, load_data_from_matrix
from graphgen import generate_graph
from nodeutils import extract_plan, add_across_plan_relations



def train(epoch, labels, features, adj, idx_train, idx_val, model, optimizer):
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
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return round(loss_train.item(), 4)


def test(labels, idx_test, features, adj, model):
    model.eval()
    output = model(features, adj)
    # transfer output to ms
    # output = output * 1000
    loss_test = F.mse_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()))


def run_train_no_upd(demo=False):
    # Step-3:
    feature_num = 3
    num_graphs = -1
    if demo:
        num_graphs = 10
    else:
        graphs = glob.glob("./pmodel_data/job/sample-plan-*")
        num_graphs = len(graphs)
    iteration_num = int(round(0.8 * num_graphs, 0))
    print("[training samples]:{}".format(iteration_num))

    for wid in range(iteration_num):
        print("[graph {}]".format(wid))

        model = get_model(feature_num=feature_num, hidden=args.hidden, nclass=NODE_DIM, dropout=args.dropout)
        optimizer = get_optimizer(model=model, lr=args.lr, weight_decay=args.weight_decay)
        # Load data
        adj, features, labels, idx_train, idx_val, idx_test = load_data(path=DATAPATH + "/graph/",
                                                                        dataset="sample-plan-" + str(wid))
        # print(adj.shape)
        # Model Training
        ok_times = 0
        t_total = time.time()
        for epoch in range(args.epochs):
            # print(features.shape, adj.shape)
            loss_train = train(epoch, labels, features, adj, idx_train, idx_val, model, optimizer)
            if loss_train < 0.002:
                ok_times += 1
            if ok_times >= 20:
                break

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        # model validate.
        test(labels, idx_test, features=features, adj=adj, model=model)
    return iteration_num, num_graphs, model


def run_test_no_upd(iteration_num, num_graphs, model):
    for wid in range(iteration_num, num_graphs):
        # Load data
        adj, features, labels, idx_train, idx_val, idx_test = load_data(path=DATAPATH + "/graph/",
                                                                        dataset="sample-plan-" + str(wid))
        # Model Testing
        t_total = time.time()
        test(labels, idx_test, features=features, adj=adj, model=model)
        print("Testing Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


def run_train_upd(demo=True, come_num=0):
    mp_optype = {'Aggregate': 0, 'Nested Loop': 1, 'Index Scan': 2, 'Hash Join': 3, 'Seq Scan': 4, 'Hash': 5,
                 'Update': 6}  # operator types in the queries
    oid = 0
    min_timestamp = -1
    if demo:
        num_graphs = 4
        come_num = 1
    else:
        graphs = glob.glob("./pmodel_data/job/sample-plan-*")
        num_graphs = len(graphs)
        assert come_num == 0
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
        print(wid)
        with open(DATAPATH + "/sample-plan-" + str(wid) + ".txt", "r") as f:

            for sample in f.readlines():
                sample = json.loads(sample)

                start_time, node_matrix, edge_matrix, conflict_operators, _, mp_optype, oid, min_timestamp = \
                    extract_plan(sample, conflict_operators, mp_optype, oid, min_timestamp)
                # print( "OID:" + str(oid))
                vmatrix = vmatrix + node_matrix
                ematrix = ematrix + edge_matrix

        db = Database("mysql")
        knobs = db.fetch_knob()
        ematrix = add_across_plan_relations(conflict_operators, knobs, ematrix)

    # TODO more features, more complicated model
    model = get_model(feature_num=feature_num, hidden=args.hidden, nclass=NODE_DIM, dropout=args.dropout)
    optimizer = get_optimizer(model=model, lr=args.lr, weight_decay=args.weight_decay)
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
    test(labels, idx_test, features, adj, model) # TODO: change name to validate.

    return num_graphs, come_num, model, adj, vmatrix, ematrix, mp_optype, oid, min_timestamp

def run_test_upd(num_graphs, come_num, model, adj, vmatrix, ematrix, mp_optype, oid, min_timestamp):
    def predict(labels, features, adj, dh):
        model.eval()
        output = model(features, adj, dh)
        loss_test = F.mse_loss(output, labels)
        acc_test = accuracy(output, labels)
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()))

#    mp_optype = {'Aggregate': 0, 'Nested Loop': 1, 'Index Scan': 2, 'Hash Join': 3, 'Seq Scan': 4, 'Hash': 5,
#                 'Update': 6}  # operator types in the queries
#    oid = 0
#    min_timestamp = -1

#    oid = 0

    # new queries( come_num samples ) come
    # modify: new_e = []
    # change new_e -> ematrix
    conflict_operators = {}
    phi = []
    for wid in range(num_graphs, num_graphs + come_num):
        print(oid, min_timestamp)
        with open(DATAPATH+"/sample-plan-" + str(wid) + ".txt", "r") as f:

            # new query come
            for sample in f.readlines():

                # updategraph-add
                sample = json.loads(sample)

                start_time, node_matrix, edge_matrix, conflict_operators, _, mp_optype, oid, min_timestamp = \
                    extract_plan(sample, conflict_operators, mp_optype, oid, min_timestamp)

                vmatrix = vmatrix + node_matrix
                ematrix = ematrix + edge_matrix

                db = Database("mysql")
                knobs = db.fetch_knob()

                ematrix = add_across_plan_relations(conflict_operators, knobs, ematrix)

                # incremental prediction
                dadj, dfeatures, dlabels, _, _, _ = load_data_from_matrix(np.array(vmatrix, dtype=np.float32),
                                                                          np.array(ematrix, dtype=np.float32))

                model.eval()
                dh = model(dfeatures, dadj, None, True)

                predict(dlabels, dfeatures, adj, dh)

                for node in node_matrix:
                    bisect.insort(phi, [node[-2] + node[-1], node[0]])

                # updategraph-remove
                num = bisect.bisect(phi, [start_time, -1])
                if num > 20:  # ZXN: k = 20, num > k.
                    rmv_phi = [e[1] for e in phi[:num]]
                    phi = phi[num:]
                    vmatrix = [v for v in vmatrix if v[0] not in rmv_phi]
                    new_e = [e for e in new_e if e[0] not in rmv_phi and e[1] not in rmv_phi]
                    for table in conflict_operators:
                        conflict_operators[table] = [v for v in conflict_operators[table] if v[0] not in rmv_phi]


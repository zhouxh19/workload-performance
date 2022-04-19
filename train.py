import glob
import time
import torch.nn.functional as F
import torch.optim as optim

from GCN import GCN, get_model, get_optimizer
from constants import args, NODE_DIM, DATAPATH
from graphembedding import load_data, accuracy


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

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

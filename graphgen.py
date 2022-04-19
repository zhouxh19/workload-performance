import glob

from dbconnnection import Database
from nodeutils import *
from constants import DATAPATH
import os
import time
import json

def generate_graph(wid, path=DATAPATH, mp_optype=None):
    if mp_optype is None:
        mp_optype = {'Aggregate': 0, 'Nested Loop': 1, 'Index Scan': 2, 'Hash Join': 3, 'Seq Scan': 4, 'Hash': 5,
                     'Update': 6}
    # global oid, min_timestamp # write only.
    # todo: timestamp

    vmatrix = []
    ematrix = []
    mergematrix = []
    conflict_operators = {}

    oid = 0
    min_timestamp = -1
    with open(os.path.join(path, "sample-plan-" + str(wid) + ".txt"), "r") as f:
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

        # Step 2: read related knobs
        db = Database("mysql")
        knobs = db.fetch_knob()

        # Step 3: add relations across queries
        ematrix = add_across_plan_relations(conflict_operators, knobs, ematrix)

        # edge: data relations based on (access tables, related knob values)
        # vmatrix, ematrix = merge.mergegraph_main(mergematrix, ematrix, vmatrix)
    ### ZXN TEMP Modified ENDED
    return vmatrix, ematrix, mergematrix

def graphGen(data_path = os.path.join(os.path.abspath('.'), 'pmodel_data','job')):
    # Step-0: split the workloads into multiple concurrent queries at different time ("sample-plan-x")

    workloads = glob.glob("./pmodel_data/job/sample-plan-*")
    start_time = time.time()
    print("Generating Graph...")
    num_graphs = 3000
    for wid in range(num_graphs):
        st = time.time()
        vmatrix, ematrix, mergematrix = generate_graph(wid)
        # vmatrix, ematrix = merge.mergegraph_main(mergematrix, ematrix, vmatrix)
        print("[graph {}]".format(wid), "time:{}; #-vertex:{}, #-edge:{}".format(time.time() - st, len(vmatrix), len(ematrix)))

    ### ZXN TEMP Modified BEGIN
        with open(data_path + "graph/" + "sample-plan-" + str(wid) + ".content", "w") as wf:
           for v in vmatrix:
               wf.write(str(v[0]) + "\t" + str(v[1]) + "\t" + str(v[2]) + "\t" + str(v[3]) + "\t" + str(v[4]) + "\n")
        with open(data_path + "graph/" + "sample-plan-" + str(wid) + ".cites", "w") as wf:
           for e in ematrix:
               wf.write(str(e[0]) + "\t" + str(e[1]) + "\t" + str(e[2]) + "\n")
    ### ZXN TEMP Modified ENDED

    end_time = time.time()

    print("Total Time:{}".format(end_time - start_time))


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

def overlap(node_i, node_j):
    if (node_j[1] < node_i[2] and node_i[2] < node_j[2]):

        return (node_i[2] - node_j[1]) / (node_j[2] - min(node_i[1], node_j[1]))

    elif (node_i[1] < node_j[2] and node_j[2] < node_i[2]):

        return (node_j[2] - node_i[1]) / (node_i[2] - min(node_i[1], node_j[1]))

    else:
        return 0


def extract_plan(sample, conflict_operators, mp_optype, oid, min_timestamp):
    # global mp_optype, oid, min_timestamp
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

    return start_time, node_matrix, edge_matrix, conflict_operators, node_merge_matrix, mp_optype, oid, min_timestamp


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

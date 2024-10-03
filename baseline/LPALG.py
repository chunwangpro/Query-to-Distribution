# This is the implementation of LPALG(PGM) algorithm for query optimization.
# "Data generation using declarative constraints"
import warnings

warnings.filterwarnings("ignore")
import argparse
import itertools
import sys
import time
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy import optimize

import datasets
import estimators as estimators_lib

# from scipy.sparse import csr_matrix


def Oracle(table, query):
    cols, idxs, ops, vals = query
    oracle_est = estimators_lib.Oracle(table)
    return oracle_est.Query(cols, ops, vals)


def cal_true_card(query, table):
    cols, idxs, ops, vals = query
    ops = np.array(ops)
    probs = Oracle(table, (cols, idxs, ops, vals))
    return probs


def GenerateQuery(table, min_num_filters, max_num_filters, rng, dataset):
    """Generate a random query."""
    num_filters = rng.randint(max_num_filters - 1, max_num_filters)
    cols, idxs, ops, vals = SampleTupleThenRandom(table, num_filters, rng, dataset)
    sel = cal_true_card((cols, idxs, ops, vals), table) / float(table.cardinality)
    return cols, idxs, ops, vals, sel


def SampleTupleThenRandom(table, num_filters, rng, dataset):
    vals = []
    new_table = table.data
    s = new_table.iloc[rng.randint(0, new_table.shape[0])]
    vals = s.values
    if dataset in ["dmv", "dmv-tiny", "order_line"]:
        vals[6] = vals[6].to_datetime64()
    elif dataset in ["orders1", "orders"]:
        vals[4] = vals[4].to_datetime64()
    elif dataset == "lineitem":
        vals[10] = vals[10].to_datetime64()
        vals[11] = vals[11].to_datetime64()
        vals[12] = vals[12].to_datetime64()
    idxs = rng.choice(len(table.columns), replace=False, size=num_filters)
    cols = np.take(table.columns, idxs)
    # If dom size >= 10, okay to place a range filter.
    # Otherwise, low domain size columns should be queried with equality.
    ops = rng.choice([">", "<=", ">=", "="], size=num_filters)
    ops_all_eqs = ["="] * num_filters
    sensible_to_do_range = [c.DistributionSize() >= 10 for c in cols]
    ops = np.where(sensible_to_do_range, ops, ops_all_eqs)
    # if num_filters == len(table.columns):
    #     return table.columns,np.arange(len(table.columns)), ops, vals
    vals = vals[idxs]
    op_a = []
    val_a = []
    for i in range(len(vals)):
        val_a.append([vals[i]])
        op_a.append([ops[i]])
    return cols, idxs, pd.DataFrame(op_a).values, pd.DataFrame(val_a).values


# LPALG (PGM) algorithm
def dictionary_column_interval(table_size, query_set):
    # Traverse all queries to apply the intervalization skill for each column
    n_column = table_size[1]
    column_interval = {}
    for i in range(n_column):
        column_interval[i] = set([sys.maxsize])  # use set([0, sys.maxsize]) to adapt '>' and '<'.
    for query in query_set:
        col_idxs = query[1]
        vals = query[3]
        for i in range(len(col_idxs)):
            column_interval[col_idxs[i]].add(vals[i][0])
    for k, v in column_interval.items():
        if not v:
            column_interval[k] = [0]
        else:
            column_interval[k] = sorted(list(v))
    return column_interval


def dictionary_column_variable(column_to_interval):
    # Assign a sequential index to each interval in each column
    column_to_variable = {}
    total_intervals = 0
    column_variable_number = []
    for k, v in column_to_interval.items():
        count = len(v)
        column_to_variable[k] = [total_intervals + i for i in range(count)]
        total_intervals += count
        column_variable_number.append(count)
    return total_intervals, column_variable_number, column_to_variable


def dictionary_variable_interval(column_to_interval, column_to_variable):
    # Map each interval index to the left endpoint of its corresponding interval
    variable_to_interval = {}
    for column, variable in column_to_variable.items():
        for i in range(len(variable)):
            variable_to_interval[variable[i]] = column_to_interval[column][i]
    return variable_to_interval


def op_to_variables(column_to_variable, column, index, op):
    # Find all matching intervals' index in the column based on the operator and input interval index
    column_interval_idx = np.array(column_to_variable[column])
    if op == ">":
        return list(column_interval_idx[index + 1 :])
    elif op == ">=":
        return list(column_interval_idx[index:])
    elif op == "=":
        return [column_interval_idx[index]]
    elif op == "<":
        return list(column_interval_idx[:index])
    elif op == "<=":
        return list(column_interval_idx[: index + 1])


def dictionary_query_to_interval(query_set, column_to_interval, column_to_variable):
    # Traverse all queries to find their corresponding interval index
    query_to_interval = {}
    for i in range(len(query_set)):
        col_idxs = query_set[i][1]
        ops = query_set[i][2]
        vals = query_set[i][3]
        query_to_interval[i] = []
        for j in range(len(col_idxs)):
            column = col_idxs[j]
            index = column_to_interval[column].index(vals[j][0])
            query_to_interval[i].append(
                op_to_variables(column_to_variable, column, index, ops[j][0])
            )
    return query_to_interval


def dictionary_query_to_x_index(query_set, query_to_interval, column_to_variable):
    # Traverse all queries to find their corresponding x index
    query_to_x_index = {}
    for i in range(len(query_set)):
        col_idxs = query_set[i][1]
        query_to_x_index[i] = [[] for i in range(n_column)]
        for j in range(len(col_idxs)):
            column = col_idxs[j]
            variable = query_to_interval[i][j]
            for k in variable:
                x_index = column_to_variable[column].index(k)
                query_to_x_index[i][column].append(x_index)
    return query_to_x_index


def transfer_x_index(query_to_x_index, column_variable_number):
    # Transfer all empty x-index-list to all-indexes-list corresponding to the column
    x_index = {}
    for k, v in query_to_x_index.items():
        x_index[k] = []
        for i in range(len(v)):
            if v[i] == []:
                x_index[k].append([j for j in range(column_variable_number[i])])
            else:
                x_index[k].append(v[i])
    return x_index


def x0():
    # use average value to initialize the x: n_row / total_x
    return np.ones(total_x) * n_row / total_x


def bounds():
    # each x should be in [0, n_row]
    return np.array([[0, n_row]] * total_x)


def constraints():
    # constraints is always limited to zero when using SLSQP
    # sum of all x should always equal to n_row
    return [{"type": "eq", "fun": lambda x: n_row - x.sum()}]


def query_constraints(query_set, x_index, column_variable_number):
    query_constraints_list = []
    # Find the corresponding x (then sum) for each query
    # because x may be a array with multiple dimensions(such as 10), build and store such big matrix is not feasible
    # so we use a 1D array to represent the x, and use the following method to find the corresponding x index
    # To be simple: 5D array [2, 3, 4, 5, 6] -> 1D array [x_ind @ find]
    find = np.array(
        [np.product(column_variable_number[i:]) for i in range(1, len(column_variable_number))]
        + [1]
    )
    for key, values in x_index.items():
        sel = query_set[key][4]
        x_ind = np.array([x for x in itertools.product(*values)])  # , dtype=np.uint16)
        result = x_ind @ find

        def value_constraints(x, sel=sel, value=result):
            # the cardinality-error of a query
            return x[value].sum() - sel * n_row

        # sparse matrix csr_matrix
        #         row = np.zeros(len(result))
        #         col = result
        #         data = np.ones(len(result))
        #         matrix = csr_matrix((data, (row, col)), shape=(1, total_x))#, dtype=np.uint16)
        #         def value_constraints(x, sel=sel, matrix=matrix):
        #             return matrix @ x - sel * n_row

        query_constraints_list.append(value_constraints)
    return query_constraints_list


def fun():
    def error(x):
        # reduce the total cardinality-error of a query to zeros
        # divided by n_row or (n_row)**2 to limit the error to a small range
        return sum([constraint(x) ** 2 for constraint in query_constraints_list]) / (n_row)  # **2

    return error


def randomized_rouding(x):
    # with probability to shift to interval left or right points
    int_x = deepcopy(x)
    for i in range(len(x)):
        xi = x[i]
        floor = np.floor(xi)
        ceil = np.ceil(xi)
        if not floor == ceil:
            int_x[i] = np.random.choice([floor, ceil], p=[xi - floor, ceil - xi])
    return int_x


def generate_table_data(column_to_interval, int_x, column_variable_number):
    df = pd.DataFrame(
        columns=[f"col_{i}" for i in range(n_column)], index=[i for i in range(int_x.sum())]
    )

    column_to_x = []
    for i in column_variable_number:
        column_to_x.append([j for j in range(i)])
    all_x = np.array([x for x in itertools.product(*column_to_x)], dtype=np.uint16)

    count = 0
    for i in range(total_x):  # Here: total_x == len(int_x), n_column == all_x.shape[1]
        if int_x[i] != 0:
            df.iloc[count : count + int_x[i], :] = [
                column_to_interval[j][all_x[i][j]] for j in range(n_column)
            ]
            count += int_x[i]
    return df


def execute_query(dataNew, query_set):
    # calculate the Q-error of each query
    diff = []
    for query in query_set:
        sentence = ""
        for i in range(len(query[0])):
            if i != 0:
                sentence += " and "
            sentence += f"col_{query[1][i]}"
            if query[2][i][0] == "=":
                sentence += "=="
            else:
                sentence += query[2][i][0]
            sentence += f"{query[3][i][0]}"
        sel = dataNew.query(sentence).shape[0] / dataNew.shape[0]
        sel2 = query[4]
        if sel == 0:
            sel += 1 / dataNew.shape[0]
        if sel2 == 0:
            sel2 += 1 / n_row
        if sel < sel2:
            diff.append(sel2 / sel)
        else:
            diff.append(sel / sel2)
    return diff


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wine", help="Dataset.")
    parser.add_argument("--query-size", type=int, default=5, help="query size")
    parser.add_argument("--num-conditions", type=int, default=2, help="num of conditions")

    try:
        args = parser.parse_args()  # for python
    except:
        args, unknown = parser.parse_known_args()  # for jupyter notebook

    print("\nBegin Generating Queries ...")
    rng = np.random.RandomState(42)
    table = datasets.LoadDataset(f"{args.dataset}.csv", args.dataset, type_casts={})
    query_set = [
        GenerateQuery(table, 2, args.num_conditions + 1, rng, args.dataset)
        for _ in range(args.query_size)
    ]
    print("Done.\n")

    print("\nBegin Building LPALG (PGM) Model ...")
    tic = time.time()
    table_size = table.data.shape
    n_row, n_column = table_size

    column_to_interval = dictionary_column_interval(table_size, query_set)

    total_intervals, column_variable_number, column_to_variable = dictionary_column_variable(
        column_to_interval
    )
    variable_to_interval = dictionary_variable_interval(column_to_interval, column_to_variable)

    total_x = np.product(column_variable_number)

    query_to_interval = dictionary_query_to_interval(
        query_set, column_to_interval, column_to_variable
    )
    query_to_x_index = dictionary_query_to_x_index(query_set, query_to_interval, column_to_variable)

    x_index = transfer_x_index(query_to_x_index, column_variable_number)

    query_constraints_list = query_constraints(query_set, x_index, column_variable_number)
    print("Done.\n")

    print(f"Begin Solving LP problem with total param = {total_x} ...")
    res = optimize.minimize(
        fun(),
        x0(),
        method="SLSQP",
        constraints=constraints(),
        bounds=bounds(),
        # tol=1e-323,
        # options={'maxiter': 1e10},
        # options={'maxiter': 1}, # only for checking the algorithm works
    )
    print("\n Optimize.minimize Solver Status: \n", res)

    print("\nBegin Generating Data ...")
    int_x = randomized_rouding(res.x).astype(int)
    print(f"\n Integer X: ( length = {len(int_x)} )\n", int_x)

    dataNew = generate_table_data(column_to_interval, int_x, column_variable_number)
    # print(dataNew)
    print("Done.\n")

    print("\nBegin Calculate Query Error ...")
    diff = execute_query(dataNew, query_set)
    print("Done.\n")

    print(f"\n\n Q-error of LPALG (PGM) algorithm:")
    print(
        f"  (dataset={args.dataset}, query_size={args.query_size}, condition={args.num_conditions}, total_param={total_x}):\n"
    )
    print(f"min:    {np.min(diff)}")
    print(f"10:     {np.percentile(diff, 10)}")
    print(f"20:     {np.percentile(diff, 20)}")
    print(f"30:     {np.percentile(diff, 30)}")
    print(f"40:     {np.percentile(diff, 40)}")
    print(f"median: {np.median(diff)}")
    print(f"60:     {np.percentile(diff, 60)}")
    print(f"70:     {np.percentile(diff, 70)}")
    print(f"80:     {np.percentile(diff, 80)}")
    print(f"mean:   {np.mean(diff)}")
    print(f"90:     {np.percentile(diff, 90)}")
    print(f"max:    {np.max(diff)}")

    toc = time.time()
    total_time = toc - tic
    m, s = divmod(total_time, 60)
    h, m = divmod(m, 60)
    print(f"\nTime passed:  {h:0>2.0f}:{m:0>2.0f}:{s:0>2.0f}")

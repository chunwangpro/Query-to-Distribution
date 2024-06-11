import argparse
import itertools
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_lattice as tfl
from matplotlib import pyplot as plt
from tqdm import tqdm

import common
import datasets
import estimators as estimators_lib


def cal_true_card(query, table):
    cols, _, ops, vals = query
    ops = np.array(ops)
    oracle_est = estimators_lib.Oracle(table)
    cardinality = oracle_est.Query(cols, ops, vals)
    return cardinality


def SampleTupleThenRandom(table, num_filters, rng, dataset):
    idxs = rng.choice(len(table.columns), replace=False, size=num_filters)
    idxs = np.sort(idxs)
    cols = np.take(table.columns, idxs)

    # If dom size >= 10, okay to place a range filter.
    # Otherwise, low domain size columns should be queried with equality.
    # ops = rng.choice(['<=', '>=', '='], size=num_filters)
    # ops = rng.choice(['<=', '>'], size=num_filters)
    ops = rng.choice(["<="], size=num_filters)

    #     ops_all_eqs = ['='] * num_filters
    #     sensible_to_do_range = [c.DistributionSize() >= 10 for c in cols]
    #     ops = np.where(sensible_to_do_range, ops, ops_all_eqs)

    # if num_filters == len(table.columns):
    #     return table.columns,np.arange(len(table.columns)), ops, vals

    ### this is fixed to row
    # vals = vals[idxs]
    ###

    ### this will sample in rows
    vals = []
    for i in range(num_filters):
        s = table.data.iloc[rng.randint(0, table.data.shape[0])]
        vals.append(s.values[idxs][i])

    vals = np.array(vals)
    ###

    op_a = []
    val_a = []
    for i in range(len(vals)):
        val_a.append([vals[i]])
        op_a.append([ops[i]])

    return cols, idxs, pd.DataFrame(op_a).values, pd.DataFrame(val_a).values


def GenerateQuery(table, min_num_filters, max_num_filters, rng, dataset):
    """Generate a random query."""
    num_filters = rng.randint(min_num_filters, max_num_filters)
    cols, idxs, ops, vals = SampleTupleThenRandom(table, num_filters, rng, dataset)
    sel = cal_true_card((cols, idxs, ops, vals), table) / table.data.shape[0]
    return cols, idxs, ops, vals, sel


def dictionary_column_interval(table_size, query_set):
    # Traverse all queries to apply the intervalization skill for each column
    n_column = table_size[1]
    column_interval = {}
    for i in range(n_column):
        column_interval[i] = set()  # use set([0, sys.maxsize]) to adapt '>' and '<'.
    for query in query_set:
        _, col_idxs, _, vals, _ = query
        for i in range(len(col_idxs)):
            column_interval[col_idxs[i]].add(vals[i][0])
    for k, v in column_interval.items():
        if not v:
            column_interval[k] = [0]  # use [0] to represent empty column interval
        else:
            column_interval[k] = sorted(list(v))
    return column_interval


# 修改区间对 > >= < = 均适用
# 修改 初始interval
# 修改空interval使用【1】
# 修改最小值最大值为：0～min， max～max+max-max_2
def count_column_unique_interval(unique_intervals):
    # count unique query interval for each column
    return [len(v) for v in unique_intervals.values()]


OPS = {">": np.greater, "<": np.less, ">=": np.greater_equal, "<=": np.less_equal, "=": np.equal}


def calculate_query_cardinality(df, ops, vals):
    # assert len(df.columns) == len(ops) == len(vals)
    bools = np.ones(len(df), dtype=bool)
    for (_, c), o, v in zip(df.items(), ops, vals):
        bools &= OPS[o](c, v)
    return bools.sum()


# inclusion-exclusion principle is time-consuming
# here we query once before generate to calculate the shortfall cardinality


def calculate_query_cardinality_numpy(data, ops, vals):
    if data is None:
        return 0
    # assert data.shape[1] == len(ops) == len(vals)
    bools = np.ones(data.shape[0], dtype=bool)
    for i, (o, v) in enumerate(zip(ops, vals)):
        bools &= OPS[o](data[:, i], v)
    return bools.sum()


# n_column = 3
# data = np.array([[1, 2, 3, 4, 5], [10, 20, 30, 40, 50], [10, 20, 30, 40, 50]]).T
# df = pd.DataFrame(data, columns=[f"col_{i}" for i in range(n_column)])
# ops = [">=", ">=", ">="]
# vals = [3, 20, 20]
# print(calculate_query_cardinality(df, ops, vals))
# print(calculate_query_cardinality_numpy(data, ops, vals))


def calculate_Q_error(dataNew, query_set, n_row):
    Q_error = []
    dataNew = common.CsvTable("dataNew", dataNew, dataNew.columns)
    dataNew_cols = dataNew.columns
    our_table_row = dataNew.data.shape[0]

    for query in tqdm(query_set):
        _, idxs, ops, vals, _ = query
        cols = np.take(dataNew_cols, idxs)
        sel_pred = cal_true_card((cols, _, ops, vals), dataNew) / our_table_row
        sel_true = query[-1]
        if sel_pred == 0 and sel_pred == 0:
            Q_error.append(1)
            continue
        if sel_pred == 0:
            sel_pred = 1 / our_table_row
        elif sel_true == 0:
            sel_true = 1 / n_row
        Q_error.append(max(sel_pred / sel_true, sel_true / sel_pred))
    return Q_error


def calculate_Q_error_old(dataNew, query_set, n_row):
    Q_error = []
    our_table_row = dataNew.shape[0]
    for query in tqdm(query_set):
        df = dataNew.copy()
        for i in range(len(query[0])):
            if query[2][i][0] == "=":
                op = "=="
            else:
                op = query[2][i][0]
            df = df.query(f"col_{query[1][i]} {op} {query[3][i][0]}")
        sel = df.shape[0] / our_table_row

        sel1 = 1 / our_table_row if sel == 0 else sel
        sel2 = 1 / n_row if query[4] == 0 else query[4]
        Q_error.append(max(sel1 / sel2, sel2 / sel1))
    return Q_error


def print_Q_error(Q_error, args):
    print(
        f"\n\n Q-error of Lattice (dataset={args.dataset}, query size={args.query_size}, condition=[{args.min_conditions}, {args.max_conditions}], loss={args.loss}):\n"
    )
    print(f"min:    {np.min(Q_error)}")
    print(f"10:     {np.percentile(Q_error, 10)}")
    print(f"20:     {np.percentile(Q_error, 20)}")
    print(f"30:     {np.percentile(Q_error, 30)}")
    print(f"40:     {np.percentile(Q_error, 40)}")
    print(f"median: {np.median(Q_error)}")
    print(f"60:     {np.percentile(Q_error, 60)}")
    print(f"70:     {np.percentile(Q_error, 70)}")
    print(f"80:     {np.percentile(Q_error, 80)}")
    print(f"90:     {np.percentile(Q_error, 90)}")
    print(f"95:     {np.percentile(Q_error, 95)}")
    print(f"99:     {np.percentile(Q_error, 99)}")
    print(f"max:    {np.max(Q_error)}")
    print(f"mean:   {np.mean(Q_error)}")

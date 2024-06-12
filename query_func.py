import numpy as np
import pandas as pd
from tqdm import tqdm

OPS = {">": np.greater, "<": np.less, ">=": np.greater_equal, "<=": np.less_equal, "=": np.equal}


def generate_random_query(table, min_conditions, max_conditions, rng):
    """Generate a random query."""
    conditions = rng.randint(min_conditions, max_conditions)
    idxs = rng.choice(table.shape[1], replace=False, size=conditions)
    idxs = np.sort(idxs)
    cols = table[:, idxs]
    # ops = rng.choice(['<', '<=', '>', '>=', '='], replace=True, size=conditions)
    ops = rng.choice(["<="], replace=True, size=conditions)
    vals = table[rng.randint(0, table.shape[0]), idxs]
    sel = calculate_query_cardinality(cols, ops, vals) / table.shape[0]
    return idxs, ops, vals, sel


# 修改区间对 > >= < = 均适用
# 修改 初始interval/好像不用改
# 修改空interval使用【1】
# 修改最小值最大值为：min/2， max+max-max_2
def column_intervalization(table_size, query_set):
    # Traverse all queries to apply the intervalization skill for each column
    column_interval = {}
    for i in range(table_size[1]):
        column_interval[i] = set()  # use set([0, sys.maxsize]) to adapt '>' and '<'.
    for query in query_set:
        idxs, _, vals, _ = query
        for i in range(len(idxs)):
            column_interval[idxs[i]].add(vals[i])
    for k, v in column_interval.items():
        if not v:
            column_interval[k] = [1]  # use [0] to represent empty column interval
        else:
            column_interval[k] = sorted(list(v))
    return column_interval


def count_column_unique_interval(unique_intervals):
    # count unique query interval for each column
    return [len(v) for v in unique_intervals.values()]


# inclusion-exclusion principle is time-consuming
# here we query once before generate to calculate the shortfall cardinality


def calculate_query_cardinality(data, ops, vals):
    """
    Calculate the cardinality (number of rows) that satisfy a given query.

    Parameters:
    data (np.ndarray): A 2D numpy array representing a subset of a table (cols).
    ops (list): A list of operators, support operators: '>', '>=', '<', '<=', '='.
    vals (list): A list of values.

    Returns:
    int: The number of rows that satisfy all the query conditions (cardinality).

    """
    if data is None:
        return 0
    # assert data.shape[1] == len(ops) == len(vals)
    bools = np.ones(data.shape[0], dtype=bool)
    for i, (o, v) in enumerate(zip(ops, vals)):
        bools &= OPS[o](data[:, i], v)
    return bools.sum()


# n_column = 3
# data = np.array([[1, 2, 3, 4, 5], [10, 20, 30, 40, 50], [10, 20, 30, 40, 50]]).T
# ops = [">=", ">=", ">="]
# vals = [3, 20, 20]
# print(calculate_query_cardinality(data, ops, vals))


def calculate_Q_error(dataNew, query_set, table_size):
    Q_error = []
    n_row = table_size[0]
    our_table_row = dataNew.shape[0]

    for query in tqdm(query_set):
        idxs, ops, vals, sel_true = query
        cols = dataNew[:, idxs]
        sel_pred = calculate_query_cardinality(cols, ops, vals) / our_table_row
        if sel_pred == 0 and sel_pred == 0:
            Q_error.append(1)
            continue
        if sel_pred == 0:
            sel_pred = 1 / our_table_row
        elif sel_true == 0:
            sel_true = 1 / n_row
        Q_error.append(max(sel_pred / sel_true, sel_true / sel_pred))
    return Q_error


def print_Q_error(Q_error, args, savepath):
    print(
        f"\n\n Q-error of Lattice (dataset={args.dataset}, query size={args.query_size}, condition=[{args.min_conditions}, {args.max_conditions}], loss={args.loss}):\n"
    )
    statistics = {
        "min": np.min(Q_error),
        "10": np.percentile(Q_error, 10),
        "20": np.percentile(Q_error, 20),
        "30": np.percentile(Q_error, 30),
        "40": np.percentile(Q_error, 40),
        "median": np.median(Q_error),
        "60": np.percentile(Q_error, 60),
        "70": np.percentile(Q_error, 70),
        "80": np.percentile(Q_error, 80),
        "90": np.percentile(Q_error, 90),
        "95": np.percentile(Q_error, 95),
        "99": np.percentile(Q_error, 99),
        "max": np.max(Q_error),
        "mean": np.mean(Q_error),
    }
    df = pd.DataFrame.from_dict(statistics, orient="index", columns=["Value"])
    df.index.name = None
    df.to_csv(f"{savepath}/Qerror.csv", index=True, header=False)
    print(df)

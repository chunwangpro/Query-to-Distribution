import numpy as np
import pandas as pd
from tqdm import tqdm

OPS = {">": np.greater, "<": np.less, ">=": np.greater_equal, "<=": np.less_equal, "=": np.equal}


def generate_random_query(table, min_conditions, max_conditions, rng):
    """Generate a random query."""
    conditions = rng.randint(min_conditions, max_conditions + 1)
    idxs = rng.choice(table.shape[1], replace=False, size=conditions)
    idxs = np.sort(idxs)
    cols = table[:, idxs]
    # ops = rng.choice(['<', '<=', '>', '>=', '='], replace=True, size=conditions)
    ops = rng.choice(["<="], replace=True, size=conditions)
    vals = table[rng.randint(0, table.shape[0]), idxs]
    sel = calculate_query_cardinality(cols, ops, vals) / table.shape[0]
    return idxs, ops, vals, sel


def column_intervalization(query_set, table_size):
    # apply the query intervalization for each column
    # applied to <, <=, >, >=, =
    column_interval = {i: set() for i in range(table_size[1])}
    for query in query_set:
        idxs, _, vals, _ = query
        for i in range(len(idxs)):
            column_interval[idxs[i]].add(vals[i])
    for k, v in column_interval.items():
        if not v:
            # use [0] for empty column interval
            column_interval[k] = [0]
        else:
            interval_list = sorted(list(v))
            smallest = 2 * interval_list[0] - interval_list[1]
            largest = 2 * interval_list[-1] - interval_list[-2]
            column_interval[k] = [smallest] + interval_list + [largest]
    return column_interval


def count_column_unique_interval(unique_intervals):
    # count unique query interval number for each column
    return [len(v) for v in unique_intervals.values()]


# inclusion-exclusion principle is time-consuming
# here we query once before generate to calculate the shortfall cardinality


def calculate_query_cardinality(data, ops, vals):
    """
    Calculate the cardinality (number of rows) that satisfy a given query.

    Parameters:
    data (np.ndarray): A 2D-array representing a subset of a table (cols).
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
    print("Begin Calculating Q-error ...")
    Q_error = []
    n_row = table_size[0]
    for query in tqdm(query_set):
        idxs, ops, vals, sel_true = query
        cols = dataNew[:, idxs]
        card_pred = calculate_query_cardinality(cols, ops, vals)
        card_true = sel_true * n_row
        if card_pred == 0:
            card_pred = 1
        if card_true == 0:
            card_true = 1
        Q_error.append(max(card_pred / card_true, card_true / card_pred))
    print("Done.\n")
    return Q_error


def print_Q_error(Q_error, args, savepath):
    print("Summary of Q-error:")
    print(
        f"dataset={args.dataset}, query size={args.query_size}, condition=[{args.min_conditions}, {args.max_conditions}], loss={args.loss}):"
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
    df = pd.DataFrame.from_dict(statistics, orient="index", columns=["Q-error"])
    df.index.name = None
    df.to_csv(f"{savepath}/Q_error.csv", index=True, header=False)
    print(df)

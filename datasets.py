"""Dataset registrations."""

import csv
import json
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import common


def convert_type(val, column_type):

    column_type = column_type.replace(",", "")

    if column_type == "date":
        # val = datetime.strptime(val.split('\'')[1], "%Y-%m-%d")
        val = pd.to_datetime(val.split("'")[1]).to_datetime64()
        # try:
        #     val = val.split('\'')[1].to_datetime64()
        # except:
        #     import IPython as ip
        #     ip.embed()
    elif column_type == "int" or column_type == "integer":
        val = int(val)
    elif column_type == "varchar" or column_type == "char" or column_type == "character":
        val = str(val).strip("'")
    elif column_type == "decimal" or column_type == "float":
        val = float(val)

    return val


def convert_op(op, val):
    left = -np.inf
    left_include = False
    right = np.inf
    right_include = False

    for i in range(len(op)):
        if op[i] == ">" and val[i] >= left:
            left = val[i]
            left_include = False
        elif op[i] == ">=" and val[i] > left:
            left = val[i]
            left_include = True
        elif op[i] == "<" and val[i] <= right:
            right = val[i]
            right_include = False
        elif op[i] == "<=" and val[i] < right:
            right = val[i]
            right_include = True
        elif op[i] == "=":
            if val[i] > left:
                left = val[i]
                left_include = True
            if val[i] < right:
                right = val[i]
                right_include = True
        else:
            assert "wrong operator:{} in sql!".format(op[i])

    op = []
    val = []
    op1 = ">" if not left_include else ">="
    op2 = "<" if not right_include else "<="

    if left != -np.inf and right != np.inf:
        val = [left, right]
        op = [op1, op2]
    elif left != -np.inf:
        val = [left]
        op = [op1]
    elif right != np.inf:
        val = [right]
        op = [op2]

    return op, val


def convert(sql, schema):
    query = {}
    for table in sql:
        col = []
        op = []
        val = []
        join_col = []

        for column in sql[table]:
            for i, (c, c_type) in enumerate(schema[table]):

                if c == column:
                    col.append(i)
                    column_type = c_type
                    break
            tmp = sql[table][column].strip("[|]")
            column_type = column_type.split("(")[0]

            if tmp == "":
                join_col.append(col[-1])
                col.pop()
            else:
                tmp = tmp.split(",")
                if len(tmp) == 4:
                    op_col = [tmp[0].strip(" "), tmp[2].strip(" ")]
                    val1 = convert_type(tmp[1].strip(" "), column_type)
                    val2 = convert_type(tmp[3].strip(" "), column_type)
                    op_col, val_col = convert_op(op_col, [val1, val2])
                    op.append(op_col)
                    val.append(val_col)
                elif len(tmp) == 2:
                    op_col = [tmp[0].strip(" ")]
                    op_col, val_col = convert_op(
                        op_col, [convert_type(tmp[1].strip(" "), column_type)]
                    )
                    op.append(op_col)
                    val.append(val_col)
                else:
                    print("Convert failed!")
                    exit()

        query[table] = [[np.array(col, dtype=int), np.array(op), val], join_col]

    return query


def LoadTPCHSchema():
    schema_file = "/data1/alibench/data/tpch/schema.csv"
    schema = {}
    with open(schema_file) as file_in:
        for line in file_in:
            if line.startswith("CREATE TABLE"):
                part = line.split(" ")
                table = part[2].strip("`")
                schema[table] = []
            elif line.startswith("  `"):
                part = line.strip().split(" ")
                schema[table].append([part[0].strip("`").lower(), part[1].strip("`").lower()])
    return schema


def LoadJOBSchema():
    schema_file = "/data1/alibench/data/imdb-benchmark/schematext.sql"
    schema = {}
    with open(schema_file) as file_in:
        for line in file_in:
            if line.startswith("CREATE TABLE"):
                part = line.split(" ")
                table = part[2]
                schema[table] = []
            elif not line.startswith(");") and not line.startswith("\n"):
                part = line.strip().split(" ")
                schema[table].append([part[0], part[1]])
    return schema


def LoadQuery(schema, sql_file):
    # sql_file = '/data1/alibench/data/imdb-benchmark/job-light/sql/job_light_queries.json'

    with open(sql_file) as file_in:
        query = []
        for line in file_in:
            sql = json.loads(line)
            query.append(convert(sql, schema))

    return query


def LoadDmv(filename="Vehicle__Snowmobile__and_Boat_Registrations.csv"):
    csv_file = "./datasets/{}".format(filename)
    # csv_file = '/data1/alibench/data/{}'.format(filename)
    cols = [
        "Record Type",
        "Registration Class",
        "State",
        "County",
        "Body Type",
        "Fuel Type",
        "Reg Valid Date",
        "Color",
        "Scofflaw Indicator",
        "Suspension Indicator",
        "Revocation Indicator",
    ]
    # Note: other columns are converted to objects/strings automatically.  We
    # don't need to specify a type-cast for those because the desired order
    # there is the same as the default str-ordering (lexicographical).
    type_casts = {"Reg Valid Date": np.datetime64}
    return common.CsvTable("DMV", csv_file, cols, type_casts)


def LoadTitle(filename="title.csv"):
    csv_file = "../datasets/{}".format(filename)
    cols = ["kind_id", "production_year", "season_nr"]
    type_casts = {"production_year": str, "season_nr": str}
    # Note: other columns are converted to objects/strings automatically.  We
    # don't need to specify a type-cast for those because the desired order
    # there is the same as the default str-ordering (lexicographical).
    return common.CsvTable(
        "title",
        csv_file,
        cols,
        type_casts,
        low_memory=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
    )


def LoadTitleId(filename="title.csv"):
    csv_file = "../datasets/{}".format(filename)
    cols = ["kind_id", "production_year", "season_nr"]
    type_casts = {"production_year": str, "season_nr": str}
    # Note: other columns are converted to objects/strings automatically.  We
    # don't need to specify a type-cast for those because the desired order
    # there is the same as the default str-ordering (lexicographical).
    return common.CsvTable(
        "title",
        csv_file,
        cols,
        type_casts,
        low_memory=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
    )


def LoadMovieInfo(filename="movie_info.csv"):
    csv_file = "../datasets/{}".format(filename)
    cols = ["info_type_id", "info"]
    type_casts = {}
    # Note: other columns are converted to objects/strings automatically.  We
    # don't need to specify a type-cast for those because the desired order
    # there is the same as the default str-ordering (lexicographical).
    return common.CsvTable(
        "movie_info",
        csv_file,
        cols,
        type_casts,
        low_memory=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
    )


def LoadMovieCompany(filename="movie_companies.csv"):
    csv_file = "../datasets/{}".format(filename)
    cols = ["company_id", "company_type_id", "note"]
    type_casts = {}
    # Note: other columns are converted to objects/strings automatically.  We
    # don't need to specify a type-cast for those because the desired order
    # there is the same as the default str-ordering (lexicographical).
    return common.CsvTable(
        "movie_companies",
        csv_file,
        cols,
        type_casts,
        low_memory=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
    )


def LoadCastInfo(filename="cast_info.csv"):
    csv_file = "../datasets/{}".format(filename)
    cols = ["nr_order", "role_id"]
    type_casts = {"nr_order": str}
    # Note: other columns are converted to objects/strings automatically.  We
    # don't need to specify a type-cast for those because the desired order
    # there is the same as the default str-ordering (lexicographical).
    return common.CsvTable(
        "cast_info",
        csv_file,
        cols,
        type_casts,
        low_memory=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
    )


def LoadTitleJoinCastinfo(
    filename1="title.csv",
    filename2="cast_info.csv",
    left_key="id",
    right_key="movie_id",
    generated_file=None,
):
    cols = ["kind_id", "production_year", "season_nr", "nr_order", "role_id"]
    type_casts = {"production_year": str, "season_nr": str, "nr_order": str}
    if generated_file is not None:
        csv_file = "../datasets/{}".format(generated_file)
        return common.CsvTable("title_cast_info", csv_file, cols, type_casts)
    title_cols = ["id", "kind_id", "production_year", "season_nr"]
    cast_info_cols = ["movie_id", "nr_order", "role_id"]
    title = pd.read_csv(
        "../datasets/{}".format(filename1),
        usecols=title_cols,
        low_memory=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
    )
    cast_info = pd.read_csv(
        "../datasets/{}".format(filename2),
        usecols=cast_info_cols,
        low_memory=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
    )
    joined_table = title.join(cast_info.set_index("movie_id"), on="id", how="inner")
    return common.CsvTable("title_cast_info", joined_table, cols, type_casts)


def LoadTitleJoinMovieinfo(
    filename1="title.csv", filename2="movie_info.csv", left_key="id", right_key="movie_id"
):
    title_cols = ["id", "kind_id", "production_year", "season_nr"]
    movie_info_cols = ["movie_id", "info_type_id", "info"]
    title = pd.read_csv("../datasets/{}".format(filename1), usecols=title_cols, low_memory=False)
    movie_info = pd.read_csv(
        "../datasets/{}".format(filename2), usecols=movie_info_cols, low_memory=False
    )
    joined_table = title.join(movie_info.set_index("movie_id"), on="id", how="inner")
    cols = [
        "title",
        "kind_id",
        "production_year",
        "phonetic_code",
        "episode_of_id",
        "season_nr",
        "episode_nr",
        "info_type_id",
        "info",
    ]
    type_casts = {"production_year": str, "season_nr": str}
    return common.CsvTable("title_movie_info", joined_table, cols, type_casts)


def LoadTitleJoinMoviecompany(
    filename1="title.csv", filename2="movie_companies.csv", left_key="id", right_key="movie_id"
):
    title_cols = ["id", "kind_id", "production_year", "season_nr"]
    movie_company_cols = ["movie_id", "company_id", "company_type_id", "note"]
    title = pd.read_csv("../datasets/{}".format(filename1), usecols=title_cols, low_memory=False)
    movie_company = pd.read_csv(
        "../datasets/{}".format(filename2), usecols=movie_company_cols, low_memory=False
    )
    joined_table = title.join(movie_company.set_index("movie_id"), on="id", how="inner")
    cols = ["kind_id", "production_year", "season_nr", "company_id", "company_type_id", "note"]
    type_casts = {"production_year": str, "season_nr": str}
    return common.CsvTable("title_movie_company", joined_table, cols, type_casts)


def LoadDataset(filename, dataset_name, cols=None, type_casts={}, drop_col=[], na_values={}):
    # Make sure that this loads data correctly.
    csv_file = filename
    df = pd.read_csv(
        csv_file,
        header=None,
        na_values=na_values,
        quotechar='"',
        delimiter=",",
        quoting=csv.QUOTE_ALL,
        skipinitialspace=True,
        escapechar="\\",
    )
    n_col = df.shape[1]
    col_index = {}
    j = 0
    for i in range(n_col):
        if i not in drop_col:
            col_index[i] = j
            j += 1
        else:
            col_index[i] = -1

    new_type_casts = {}
    for key in type_casts.keys():
        if col_index[key] != -1:
            new_type_casts[col_index[key]] = type_casts[key]

    if len(drop_col) > 0:
        df = df.drop(drop_col, axis=1)
        df.columns = range(df.shape[1])

    # for key in new_type_casts.keys():
    #    df['key'] = df['key'].astype()
    # print("load dataset {} done".format(filename))
    # print(df.shape[0])
    """
    for i in range(df.shape[1]):

        print(df.iloc[:,i].dtype)
    """

    """
    if dataset_name == 'orders1':
        df = df.drop([5],axis = 1)
        df.columns = range(df.shape[1])

    for col in df.columns.tolist():
        df[col][df[col] == '\\N'] = np.nan
        df[col][df[col] == 'nan'] = np.nan
        df[col][df[col] == 'NAN'] = np.nan
    """
    cols = df.columns
    return common.CsvTable(dataset_name, df, cols, new_type_casts)


# if __name__ == "__main__":
#     schema = LoadJOBSchema()
#     query = LoadJOBQuery(schema)
#     print(query[16])
#     print(query[1])
#     print(query[2])

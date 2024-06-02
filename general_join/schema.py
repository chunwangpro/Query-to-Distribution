import numpy as np
import psqlparse
import datetime
import random
import math
import numpy
import os
import pickle

class Predicate():
    '''
    col_name: name of attribute
    func: function of expression on the attribute
    op: comparison operator (a numpy function)
    right_val: the comparison value in condition
    '''
    def __init__(self, col_name, func, op, right_val):
        # print (col_name, func, op, right_val)
        self.col_name = col_name
        self.func = func
        self.op = op
        self.right_val = right_val

    def apply(self, vals):
        # print (vals)
        # print (self.col_name, self.func, self.op, self.right_val)
        # print ('begin apply a predicate: ', vals, self.right_val, self.op, self.func(vals))
        # print (self.op(self.func(vals), self.right_val))
        return self.op(self.func(vals), self.right_val)

class Query():
    def __init__(self, sql, types, cardinality):
        self.query = sql
        self.types = types
        self.true_cardinality = cardinality
        self.where_clauses = self.extract_where_from_sql()
        self.filter_cols = set([x.col_name for x in self.get_all_leaves()])
        self.is_column_correlated_in_predicate = False

    def get_op(self, operator, right_val, T):
        if operator == '=':
            if type(right_val) is list:
                def in_list_op(x, y):
                    result = []
                    for v in x:
                        if v in y:
                            result.append(True)
                        else:
                            result.append(False)
                    return np.array(result)
                op = in_list_op
            else:
                if T is str:
                    op = np.char.equal
                else:
                    op = np.equal
        elif operator == '<>':
            if T is str:
                op = np.char.not_equal
            else:
                op = np.not_equal
        elif operator == '<=':
            if T is str:
                op = np.char.less_equal
            else:
                op = np.less_equal
        elif operator == '>=':
            if T is str:
                op = np.char.greater_equal
            else:
                op = np.greater_equal
        elif operator == '<':
            if T is str:
                op = np.char.less
            else:
                op = np.less
        elif operator == '>':
            if T is str:
                op = np.char.greater
            else:
                op = np.greater
        elif operator == '!~~':
            op = np.char.not_equal
        elif operator == '~~':
            op = np.char.equal
        else:
            raise Exception('Invalid Operator {}'.format(operator))
        return op
    
    def get_right_val(self, root):
        if type(root) is psqlparse.nodes.parsenodes.AConst:
            value = ''
            pos = root.location
            keep = False
            while pos < len(self.query) and (self.query[pos] not in [',', ' ', ';', ')'] or keep):
                if self.query[pos] == '\'':
                    keep = not keep
                else:
                    value += self.query[pos]
                pos += 1
            return value
        elif type(root) is psqlparse.nodes.parsenodes.FuncCall:
            return self.get_right_val(root.args[0])
        elif type(root) is list:
            return [self.get_right_val(x) for x in root]
        else:
            raise Exception('Invalid Right Type: {}'.format(root))
    
    def get_colname_from_fields(self, fields):
        # assert len(self.tables) == 1 or len(fields) > 1
        if len(fields) == 1 and len(self.tables) > 1:
            for t in self.tables:
                if '{}.{}'.format(t, fields[-1].val) in self.types:
                    col_name = '{}.{}'.format(t, fields[-1].val)
                    break
        else:
            if len(fields) == 1:
                col_name = self.tables[-1] + '.' + fields[-1].val
            else:
                if fields[0].val in self.alias2col:
                    col_name = self.alias2col[fields[0].val] + '.' + fields[1].val
                else:
                    col_name = fields[0].val + '.' + fields[1].val
        if col_name in self.join_keys:
            col_name = '_'.join(sorted(self.tables)) + '.' + col_name
        return col_name

    def get_type(self, col_name):
        if len(col_name.split('.')) == 3:
            col_name = '.'.join(col_name.split('.')[1:])
        return self.types[col_name]

    def get_left_func(self, root):
        if type(root) is psqlparse.nodes.parsenodes.ColumnRef:
            f = lambda x: x
            return f, self.get_colname_from_fields(root.fields)
        elif type(root) is psqlparse.nodes.parsenodes.AExpr:
            col_name = self.get_colname_from_fields(root.lexpr.fields)
            T = self.get_type(col_name)
            if root.name[0].val == '&':
                f = lambda x: x & self.type_convert(self.get_right_val(root.rexpr), T)
            else:
                raise Exception('Invalid Left Func {}'.format(root.name[0].val))
            return f, col_name
        else:
            raise Exception('Invalid Left Node Type {}'.format(root))

    def type_convert(self, val, T):
        # print ('into type_convert:', val, T, type(val))
        if T is datetime:
            if type(val) is list:
                return [datetime.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').timestamp() for x in val]
            elif type(val) is numpy.ndarray:
                try:
                    return np.array([datetime.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').timestamp() for x in val])
                except:
                    return val.astype(int)
            else:
                return datetime.datetime.strptime(str(val), '%Y-%m-%d %H:%M:%S').timestamp()
        elif T is str:
            if type(val) is list:
                return [str(x) for x in val]
            else:
                return str(val)
        elif T is int:
            if type(val) is list:
                return [int(x) for x in val]
            elif type(val) is numpy.ndarray:
                return val.astype(np.uint64)
            else:
                return int(val)
        elif T is float:
            if type(val) is list:
                return [float(x) for x in val]
            elif type(val) is numpy.ndarray:
                return val.astype(np.float64)
            else:
                return float(val)
        else:
            raise Exception('Invaid type {}'.format(T))

    def _get_all_leaves(self, root):
        if hasattr(root, 'predicate'):
            return [root.predicate]
        else:
            result = []
            for node in root.args:
                result += self._get_all_leaves(node)
            if type(root) is psqlparse.nodes.primnodes.BoolExpr:
                if root.boolop == 1 and len(set([x.col_name for x in result])) > 1:
                    self.is_column_correlated_in_predicate = True
            return result

    def get_all_leaves(self):
        return self._get_all_leaves(self.where_clauses)

    def set_leaf(self, root):
        if type(root) is psqlparse.nodes.parsenodes.AExpr:
            right_val = self.get_right_val(root.rexpr)
            left_func, left_column = self.get_left_func(root.lexpr)
            T = self.get_type(left_column)
            operator = self.get_op(root.name[0].val, right_val, T)
            root.predicate = Predicate(left_column, left_func, operator, self.type_convert(right_val, T))
        elif type(root) is psqlparse.nodes.primnodes.BoolExpr:
            for node in root.args:
                self.set_leaf(node)
        else:
            raise Exception('Invalid Node {}'.format(root))

    def get_join_keys_from_expr(self, expr):
        joins = set([])
        if type(expr) is psqlparse.nodes.parsenodes.AExpr:
            if expr.lexpr.fields[0].val in self.alias2col:
                joins.add(self.alias2col[expr.lexpr.fields[0].val] + '.' + expr.lexpr.fields[1].val)
            else:
                joins.add(expr.lexpr.fields[0].val + '.' + expr.lexpr.fields[1].val)
            if expr.rexpr.fields[0].val in self.alias2col:
                joins.add(self.alias2col[expr.rexpr.fields[0].val] + '.' + expr.rexpr.fields[1].val)
            else:
                joins.add(expr.rexpr.fields[0].val + '.' + expr.rexpr.fields[1].val)
        elif type(expr) is psqlparse.nodes.primnodes.BoolExpr:
            for a in expr.args:
                joins |= self.get_join_keys_from_expr(a)
        return joins

    def parse_from_clause(self, from_clause):
        tables = []
        joins = set([])
        if type(from_clause) is psqlparse.nodes.primnodes.RangeVar:
            tables.append(from_clause.relname)
            if from_clause.alias is not None:
                self.alias2col[from_clause.alias.aliasname] = from_clause.relname
        elif type(from_clause) is psqlparse.nodes.primnodes.JoinExpr:
            tables1, joins1 = self.parse_from_clause(from_clause.larg)
            tables2, joins2 = self.parse_from_clause(from_clause.rarg)
            tables = tables1 + tables2
            joins = joins1 | joins2 | self.get_join_keys_from_expr(from_clause.quals)
        return tables, joins

    def extract_where_from_sql(self):
        # print (self.query)
        statement = psqlparse.parse(self.query)[0]
        from_clause = statement.from_clause
        self.alias2col = {}
        self.tables, self.join_keys = self.parse_from_clause(from_clause[0])
        where_clause = statement.where_clause
        self.set_leaf(where_clause)
        return where_clause

    def valid_apply(self, rows, root):
        if type(root) is psqlparse.nodes.primnodes.BoolExpr:
            if root.boolop == 0:
                # and
                result = True
                for node in root.args:
                    result = result & self.valid_apply(rows, node)
                return result
            elif root.boolop == 1:
                result = False
                for node in root.args:
                    result = result | self.valid_apply(rows, node)
                return result
            else:
                raise Exception('Invalid Bool Operator {}'.format(root.boolop))
        elif type(root) is psqlparse.nodes.parsenodes.AExpr:
            pre = root.predicate
            if pre.col_name not in rows:
                return True
            else:
                T = self.types[pre.col_name]
                vals = np.array(rows[pre.col_name])
                vals = vals.astype(str)
                mask = (vals == '*')
                # print ('mask is: ', mask)
                vals[mask] = 1
                # str makes all the numpy.ndarray become string type
                if T is int or T is float or T is datetime:
                    # print ('str makes all the numpy.ndarray become string type')
                    vals = self.type_convert(vals, T)
                if pre.op is np.not_equal or pre.op is np.char.not_equal:
                    return pre.apply(vals) | mask
                else:
                    return pre.apply(vals) & (~mask)
        else:
            raise Exception('Invalid Node {}'.format(root))

    def valid_rows(self, rows):
        return self.valid_apply(rows, self.where_clauses)

class Column():
    def __init__(self, name, min, max, T):
        self.name = name
        self.min = self.type_convert(min, T)
        self.max = self.type_convert(max, T)
        self.type = T
        self.all_distinct_values = None

    def type_convert(self, val, T):
        # print ('into type_convert:', val, T, type(val))
        if T is datetime:
            if type(val) is list:
                return [datetime.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').timestamp() for x in val]
            elif type(val) is numpy.ndarray:
                try:
                    return np.array([datetime.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').timestamp() for x in val])
                except:
                    return val.astype(int)
            else:
                return datetime.datetime.strptime(str(val), '%Y-%m-%d %H:%M:%S').timestamp()
        elif T is str:
            if type(val) is list:
                return [str(x) for x in val]
            else:
                return str(val)
        elif T is int:
            if type(val) is list:
                return [int(x) for x in val]
            elif type(val) is numpy.ndarray:
                return val.astype(np.uint64)
            else:
                return int(val)
        elif T is float:
            if type(val) is list:
                return [float(x) for x in val]
            elif type(val) is numpy.ndarray:
                return val.astype(np.float64)
            else:
                return float(val)
        else:
            raise Exception('Invaid type {}'.format(T))

    def SetDistribution(self, distinct_values):
        """This is all the values this column will ever see."""
        print (self.name, self.all_distinct_values)
        assert self.all_distinct_values is None
        # pd.isnull returns true for both np.nan and np.datetime64('NaT').
        # NOTE: np.sort puts NaT values at beginning, and NaN values at end.
        # For our purposes we always add any null value to the beginning.
        self.all_distinct_values = distinct_values
        self.distribution_size = len(distinct_values)
        return self

class Schema():

    def __init__(self, name, columns, cardinalities):
        self.table_name = name
        self.columns = columns
        self.cardinalities = cardinalities
        self.distinct_vals = None

    def preprocess_cols(self):
        assert self.distinct_vals is not None
        self._build_columns(self.columns)
        self.split_col_start_index, self.rev_index, self.split_col_size, self.split_col_bit = self._build_sub_columns()

    def _build_columns(self, columns):
        """Example args:
            cols = ['Model Year', 'Reg Valid Date', 'Reg Expiration Date']
            type_casts = {'Model Year': int}
        Returns: a list of Columns.
        """
        # Discretize & create Columns.
        for col in self.columns:
            col.SetDistribution(self.distinct_vals[col.name])
    
    def _build_sub_columns(self):
        num_col = len(self.columns)
        split_col_size = []
        split_col_bit = [] 
        split_col_start_index = []
        rev_index = [] 
        cnt = 0 
        for i in range(num_col):
            split_col_start_index.append(cnt)
            bin_size = bin(self.columns[i].distribution_size-1) 
            print (self.columns[i].name, self.columns[i].distribution_size)
            rev_index.append(i)
            cnt += 1 
            split_col_size.append(self.columns[i].distribution_size)
            split_col_bit.append(len(bin_size)-2)
        return split_col_start_index,rev_index,split_col_size,split_col_bit

    def sample_val_by_range_type(self, start, end, T):
        if T is int:
            return random.randint(start, end)
        elif T is float:
            return random.uniform(start, end)
        elif T is datetime:
            return random.randint(start, end)
        else:
            raise Exception('Can not sample for type {}'.format(T))

    def extract_distinct_vals(self, queries):
        if os.path.exists('distinct_vals.pkl'):
            with open('distinct_vals.pkl', 'rb') as f:
                self.distinct_vals = pickle.load(f)
            return
        self.distinct_vals = {}
        distinct_values_equal = {}
        distinct_values_range = {}
        for q in queries:
            conditions = q.get_all_leaves()
            for c in conditions:
                if c.op in [np.greater, np.greater_equal, np.less, np.less_equal]:
                    if c.col_name in distinct_values_range:
                        distinct_values_range[c.col_name].add(c.right_val)
                    else:
                        distinct_values_range[c.col_name] = {c.right_val}
                else:
                    if c.col_name in distinct_values_equal:
                        if type(c.right_val) is list:
                            for v in c.right_val:
                                distinct_values_equal[c.col_name].add(v)
                        else:
                            distinct_values_equal[c.col_name].add(c.right_val)
                    else:
                        if type(c.right_val) is list:
                            distinct_values_equal[c.col_name] = set(c.right_val)
                        else:
                            distinct_values_equal[c.col_name] = {c.right_val}
        for col in self.columns:
            self.distinct_vals[col.name] = []
            if col.name in distinct_values_equal:
                self.distinct_vals[col.name] += sorted(distinct_values_equal[col.name])
            self.distinct_vals[col.name].append('*')
            if col.name in distinct_values_range:
                partitions = sorted(distinct_values_range[col.name])
                start = col.min
                for end in partitions:
                    self.distinct_vals[col.name].append(self.sample_val_by_range_type(start, end, col.type))
                self.distinct_vals[col.name].append(self.sample_val_by_range_type(end, col.max, col.type))
            self.distinct_vals[col.name] = list(set(self.distinct_vals[col.name]))
        with open('distinct_vals.pkl', 'wb') as f:
            pickle.dump(self.distinct_vals, f)


# cardinality = 13364709
# is_main = Column('is_main', 0, 1, int)
# biz_order_id = Column('biz_order_id', 31635, 1146860131006968630, int)
# pay_status = Column('pay_status', 1, 12, int)
# is_detail = Column('is_detail', 0, 1, int)
# auction_id = Column('auction_id', 0, 220463047172604086, int)
# biz_type = Column('biz_type', 100, 52001, int)
# buyer_flag = Column('buyer_flag', 0, 205, int)
# options = Column('options', 0, 4611686022722355200, int)
# buyer_id = Column('buyer_id', 21006, 2208724496142, int)
# seller_id = Column('seller_id', 73, 2208694966044, int)
# attribute4 = Column('attribute4', 0, 2, int)
# logistics_status = Column('logistics_status', 1, 8, int)
# status = Column('status', 0, 1, int)
# s_time = datetime.datetime.strptime('2007-01-26 22:49:39', '%Y-%m-%d %H:%M:%S').timestamp()
# e_time = datetime.datetime.strptime('2020-09-01 17:18:58', '%Y-%m-%d %H:%M:%S').timestamp()
# gmt_create = Column('gmt_create', s_time, e_time, datetime)
# s_time = datetime.datetime.strptime('2007-01-26 22:49:39', '%Y-%m-%d %H:%M:%S').timestamp()
# e_time = datetime.datetime.strptime('2020-09-01 17:18:58', '%Y-%m-%d %H:%M:%S').timestamp()
# end_time = Column('end_time', s_time, e_time, datetime)
# s_time = datetime.datetime.strptime('2008-10-04 17:50:50', '%Y-%m-%d %H:%M:%S').timestamp()
# e_time = datetime.datetime.strptime('2020-07-30 17:18:59', '%Y-%m-%d %H:%M:%S').timestamp()
# pay_time = Column('pay_time', s_time, e_time, datetime)
# from_group = Column('from_group', 0, 4, int)
# sub_biz_type = Column('sub_biz_type', 0, 5007, int)
# attributes = Column('attributes', '', '', str)
# buyer_rate_status = Column('buyer_rate_status', 4, 7,int)
# parent_id = Column('parent_id', 0, 1146860131006968630, int)
# refund_status = Column('refund_status', 0, 14, int)
# # columns = [biz_order_id, is_detail, seller_id, auction_id, biz_type, pay_status, options, buyer_id, status, gmt_create, from_group]
# columns = [is_main, options, attributes, biz_type, refund_status, buyer_id, seller_id, buyer_flag, sub_biz_type, status, gmt_create, from_group]
# types = {}
# for c in columns:
#     types[c.name] = c.type
# sql = "SELECT count(*) FROM tc_biz_order_0526 AS tc_biz_order WHERE biz_order_id = 1144842945832075801"
# sql = "SELECT count(*) FROM tc_biz_order_0526 AS tc_biz_order WHERE is_main <> 1 AND (options & 134217728 <> 134217728 OR options & 268435456 <> 268435456) AND biz_type IN (2700, 2000, 2400, 10000, 500, 100, 300, 200) AND buyer_id = 527250382 AND seller_id = 2887913566 AND sub_biz_type IN (2, 601, 701, 801, 901, 1) AND status = 0 AND gmt_create >= DATE_FORMAT('2020-04-26 20:06:17', '%Y-%m-%d %T') AND gmt_create <= DATE_FORMAT('2020-07-26 20:06:17', '%Y-%m-%d %T') AND from_group = 0"
# sql = "SELECT count(*) FROM tc_biz_order_0526 AS tc_biz_order WHERE is_main = 1 AND biz_type IN (1110, 6001) AND (options & 134217728 <> 134217728 OR options & 268435456 <> 268435456) AND buyer_id = 72410126 AND refund_status IN (3, 2, 6, 1, 10) AND status = 0 AND buyer_flag IN (5, 4, 3, 2, 1, 0) AND from_group = 0 AND attributes NOT LIKE '%;tbpwBizType:c2b2c;%'"
# sql = "SELECT count(*) FROM generic_edit_df edf JOIN generic_task df ON e.id = t.id WHERE id = 355463819"
# sql1 = "SELECT count(*) FROM generic_review_df WHERE resource_id in ( 'secret_edit_152204' )"
sql = "SELECT count(*) FROM generic_review_df r JOIN generic_task t ON r.task_id = t.id JOIN generic_edit_df e ON r.id = e.id WHERE r.package_id in (1) and r.task_type = 'qualify_judge' and r.work_type in (1,2) and e.work_type != 3 and r.task_id IN (1,2,3)"
print (psqlparse.parse(sql)[0].where_clause.args[0].rexpr)
# types = {'generic_review_df.task_id': int, 'generic_review_df.resource_id': str, 'generic_review_df.id': int, 'generic_review_df.package_id': int, 'generic_review_df.task_type': str, 'generic_review_df.work_type': int, 'generic_edit_df.work_type': int}
# q = Query(sql1, types, 10000)
# for pred in q.get_all_leaves():
#     print (pred.col_name)
# for p in q.get_all_leaves():
#     print (p.right_val)
# print (q.get_all_leaves())
# print (q.valid_rows({'is_main': ['1'], 'options': ['*'], 'biz_type': ['1110'], 'from_group': ['0'], 'attributes': ['%;tapwBizType:c2b2c;%']}))
# table = Schema(columns, cardinality)
# table.extract_distinct_vals([q])
# print (table.distinct_vals)

"""A suite of cardinality estimators.

In practicular, inference algorithms for autoregressive density estimators can
be found in 'ProgressiveSampling'.
"""
import pdb

from graphviz import Digraph
import torch
from torch.autograd import Variable
# make_dot was moved to https://github.com/szagoruyko/pytorchviz
from torchviz import make_dot
from torch.distributions import RelaxedOneHotCategorical

import bisect
import collections
import json
import operator
import time

import numpy as np
import pandas as pd
import torch

import common
import made
# import transformer
from sklearn.tree import DecisionTreeRegressor
import sys

from joblib import Parallel, delayed
# from bloomfilter import BloomFilter

import random
import math

OPS = {
    '>': np.greater,
    '<': np.less,
    '>=': np.greater_equal,
    '<=': np.less_equal,
    '=': np.equal
}

class CardEst(object):
    """Base class for a cardinality estimator."""

    def __init__(self):
        self.query_starts = []
        self.query_dur_ms = []
        self.errs = []
        
        self.lowerrs = []
        self.miderrs = []
        self.higherrs = []
        
        self.est_cards = []
        self.true_cards = []

        self.name = 'CardEst'

    def Query(self, columns, operators, vals):
        """Estimates cardinality with the specified conditions.

        Args:
            columns: list of Column objects to filter on.
            operators: list of string representing what operation to perform on
              respective columns; e.g., ['<', '>='].
            vals: list of raw values to filter columns on; e.g., [50, 100000].
              These are not bin IDs.
        Returns:
            Predicted cardinality.
        """
        raise NotImplementedError

    def OnStart(self):
        self.query_starts.append(time.time())

    def OnEnd(self):
        self.query_dur_ms.append((time.time() - self.query_starts[-1]) * 1e3)
        

    def AddError(self, err):
        self.errs.append(err)

    def AddlowError(self,err):
        self.lowerrs.append(err)

    def AddmidError(self,err):
        self.miderrs.append(err)

    def AddhighError(self,err):
        self.higherrs.append(err)

    def AddError(self, err, est_card, true_card):
        self.errs.append(err)
        self.est_cards.append(est_card)
        self.true_cards.append(true_card)

    def __str__(self):
        return self.name

    def get_stats(self):
        return [
            self.query_starts, self.query_dur_ms, self.errs, self.est_cards,
            self.true_cards
        ]

    def merge_stats(self, state):
        self.query_starts.extend(state[0])
        self.query_dur_ms.extend(state[1])
        self.errs.extend(state[2])
        self.est_cards.extend(state[3])
        self.true_cards.extend(state[4])

    def report(self):
        est = self
        print(est.name, "max", np.max(est.errs), "99th",
              np.quantile(est.errs, 0.99), "95th", np.quantile(est.errs, 0.95),
              "median", np.quantile(est.errs, 0.5), "time_ms",
              np.mean(est.query_dur_ms))

class Oracle(CardEst):
    """Returns true cardinalities."""

    def __init__(self, table, limit_first_n=None):
        super(Oracle, self).__init__()
        self.table = table
        self.limit_first_n = limit_first_n

    def __str__(self):
        return 'oracle'

    def Query(self, columns, operators, vals,return_masks=False):
        assert len(columns) == len(operators) == len(vals)
        self.OnStart()
        
        bools = None
        for c, o, v in zip(columns, operators, vals):
            # print (c, o, v)
            bools_i = None
            for i in range(len(o)):
                if self.limit_first_n is None:
                    inds = OPS[o[i]](c.data, v[i])
                else:
                    # For data shifts experiment.
                    inds = OPS[o[i]](c.data[:self.limit_first_n], v[i])

                if bools_i is None:
                    bools_i = inds
                else:
                    bools_i &= inds
        
            if bools is None:
                bools = bools_i
            else:
                bools &= bools_i

        c = bools.sum()
        self.OnEnd()
        if return_masks:
            return bools
        # print ('c', c)
        return c


def FillInUnqueriedColumns(table, columns, operators, vals, cols=None):
    """Allows for some columns to be unqueried (i.e., wildcard).

    Returns cols, ops, vals, where all 3 lists of all size len(table.columns),
    in the table's natural column order.

    A None in ops/vals means that column slot is unqueried.
    """

    ncols = len(table.columns)
   
    cs = table.columns
    os, vs = [None] * ncols, [None] * ncols

    for c, o, v in zip(columns, operators, vals):
        idx = table.ColumnIndex(c.name)
        os[idx] = o
        vs[idx] = v

    if cols is None:
        return cs, os, vs

    return cs, os, vs, table.ColumnIndex(col)

class ProgressiveSampling(CardEst):
    """Progressive sampling."""

    def __init__(
            self,
            model,
            output_encoding,
            table,
            r,
            device=None,
            seed=False,
            cardinality=None,
            shortcircuit=False  # Skip sampling on wildcards?
    ):
    
        self.device = device
        self.model = model
        if self.device == 'cpu':
            torch.set_num_threads(32)
            self.model = model.to(self.device)

        super(ProgressiveSampling, self).__init__()
        torch.set_grad_enabled(True)
        torch.autograd.set_detect_anomaly(True)
        
        self.output_encoding = output_encoding
       
        self.table = table
        self.shortcircuit = shortcircuit

        if r <= 1.0:
            self.r = r  # Reduction ratio.
            self.num_samples = None
        else:
            self.num_samples = r

        self.seed = seed

        self.cardinality = cardinality
        if cardinality is None:
            self.cardinality = table.cardinality

        self.init_logits = self.model(
            torch.zeros(1, self.model.nin, device=device, requires_grad=True))
        
        # Inference optimizations below.

        self.traced_fwd = None
        # We can't seem to trace this because it depends on a scalar input.
        self.traced_encode_input = model.EncodeInput

        # if 'MADE' in str(model):
        #     for layer in model.net:
        #         if type(layer) == made.MaskedLinear:
        #             if layer.masked_weight is None:
        #                 layer.masked_weight = layer.mask * layer.weight

                        # print('Setting masked_weight in MADE, do not retrain!')
        # for p in model.parameters():
        #     p.requires_grad = True
        self.kZeros = torch.zeros(self.num_samples,
                                    self.model.nin,
                                    device=self.device)
        self.inp = self.traced_encode_input(self.kZeros)
        # For transformer, need to flatten [num cols, d_model].
        self.inp = self.inp.view(self.num_samples, -1)

    def __str__(self):
        if self.num_samples:
            n = self.num_samples
        else:
            n = int(self.r * self.table.columns[0].DistributionSize())
        return 'psample_{}'.format(n)

    def cal_logits(self,inp,ordering):
        if hasattr(self.model, 'do_forward'):
        # With a specific ordering.
            logits = self.model.do_forward(inp, ordering)
        else:
            if self.traced_fwd is not None:
                logits = self.traced_fwd(inp)
            else:
                logits = self.model.forward_with_encoded_input(inp)

        return logits

    def split_val(self,val,bit2):
        
        bin_val = bin(val)
      
        if len(bin_val) -2 <= bit2:
            sub_val1 = 0 
            sub_val2 = val
        else:
            sub_val1 = int(bin_val[:-bit2],2)
            sub_val2 = int(bin_val[-bit2:],2)

        return sub_val1,sub_val2

    def findborderindex(self,sample,border,valid):
        res_valid = torch.ones((self.num_samples,valid[0].shape[0]), device=self.device)
        border_exist = False 
        two_op = len(border) ==2 
        for i in range(len(sample)):
            if sample[i] == border[0]:
                border_exist = True
                if two_op and sample[i] ==  border[1]:
                    res_valid[i,:] = valid[0]*valid[1]
                else:
                    res_valid[i,:] = valid[0]
            elif two_op and  sample[i] == border[1]:
                border_exist = True
                res_valid[i,:] = valid[1]
        return border_exist,res_valid

    def _sample_n(self,
                  num_samples,
                  ordering,
                  columns,
                  operators,
                  vals,
                  inp=None,
                  is_train=False):
        this_model = self.model
        ncols = len(self.table.split_col_size)
        logits = self.init_logits
        # print (logits.requires_grad)
        if inp is None:
            inp = self.inp[:num_samples]
        masked_probs = []

        # Use the query to filter each column's domain.
        valid_i_list = [None] * ncols  # None means all valid.
        sub_col = [False] * ncols
        sub_col_border = [[] for i in range(ncols)] 
        
        i = 0 
        while i < ncols:
            natural_idx = ordering[i] if ordering else i 
            column_idx = self.table.rev_index[natural_idx] 
            # Column i.
            op = operators[column_idx]
            # print (i, natural_idx, column_idx, op)
            all_val = columns[column_idx].all_distinct_values
            if op is not None:
                # There exists a filter.
                #no split col 
                if self.table.split_col_size[natural_idx] == len(all_val):
                    valid_i = None
                    for j in range(len(op)):
                        ind = OPS[op[j]](all_val,vals[column_idx][j])
                        if valid_i is None:
                            valid_i = ind
                        else:
                            valid_i &= ind
                    if pd.isnull(all_val[0]):
                        valid_i[0] = False
                    valid_i = valid_i.astype(np.float32,copy=False)
                    valid_i_list[i] = torch.as_tensor(valid_i, device=self.device)
                    # print (i, len(valid_i), len(op), op[0], vals[column_idx][0])
                    i += 1 
                # split col 
                else:
                    sub_col[natural_idx+1] = True
                    valid_i_1 = None
                    valid_i_2 = None
                    for j in range(len(op)):
                        index = np.searchsorted(all_val,vals[column_idx][j])
                        exist =  index<len(all_val) and (all_val[index] == vals[column_idx][j])
                        if op[j] == '>' or op[j] == '>=':
                            op1 = '>='
                            op2 = '>='
                            if op[j] == '>' and exist:
                                op2 = '>'
                        else:
                            op1 = '<='
                            op2 = '<'
                            if op[j] == '<=' and exist:
                                op2 = '<='

                        val1,val2 = self.split_val(index,self.table.split_col_bit[natural_idx+1])

                        sub_col_border[natural_idx+1].append(val1)
                        size1 = self.table.split_col_size[natural_idx]
                        size2 = self.table.split_col_size[natural_idx+1]

                        ind1 = OPS[op1](range(size1),val1)
                        ind2 = OPS[op2](range(size2),val2)
                        if valid_i_1 is None:
                            valid_i_1 = ind1
                        else:
                            valid_i_1 &= ind1

                        if pd.isnull(all_val[0]):
                            ind2[0] = False
                        ind2 = ind2.astype(np.float32,copy=False)

                        if valid_i_list[i+1] is None:
                            valid_i_list[i+1] = [torch.as_tensor(ind2, device=self.device)]
                        else:
                            valid_i_list[i+1].append(torch.as_tensor(ind2, device=self.device))
                    valid_i_1 = valid_i_1.astype(np.float32,copy=False)
                    valid_i_list[i] = torch.as_tensor(valid_i_1, device=self.device)
                    # print (i, len(valid_i_1), len(op), op[0], vals[column_idx][0])  
                    i += 2
            else:
                i += 1


        # Fill in wildcards, if enabled.
        if self.shortcircuit:
            for i in range(ncols):
                natural_idx = ordering[i] if ordering else i 
                column_idx = self.table.rev_index[natural_idx] 
                if operators[column_idx] is None and natural_idx != ncols - 1:
                    inp = this_model.EncodeInput(
                        None,
                        natural_col=natural_idx,
                        out=inp)

        # Actual progressive sampling.  Repeat:
        #   Sample next var from curr logits -> fill in next var
        #   Forward pass -> curr logits
        samples_i_list = [None] * ncols
        do_sample_list = [False] * ncols
        cnt = 0
        for i in range(ncols):
            natural_idx = ordering[i] if ordering else i 
            column_idx = self.table.rev_index[natural_idx] 
            
            # If wildcard enabled, 'logits' wasn't assigned last iter.
            # if not self.shortcircuit or operators[column_idx] is not None:
            do_sample = True 
            do_sample_list[i] = True
            if sub_col[natural_idx]:
                border_exist,valid_i = self.findborderindex(samples_i,sub_col_border[natural_idx],valid_i_list[i])
                if border_exist:
                    probs_i = torch.softmax(
                        this_model.logits_for_col(natural_idx, logits), 1)
                    probs_i = probs_i * valid_i
                    probs_i_summed = probs_i.sum(1)
                    masked_probs.append(probs_i_summed)

                    # If some paths have vanished (~0 prob), assign some nonzero
                    # mass to the whole row so that multinomial() doesn't complain.
                    # paths_vanished = (probs_i_summed <= 0).view(-1, 1)
                    # probs_i = probs_i.masked_fill_(paths_vanished, 1.0)
                else:
                    do_sample = False 
                    do_sample_list[i] = False
            else:
                probs_i = torch.softmax(
                    this_model.logits_for_col(natural_idx, logits), 1)
                # print ('probs_i: ', probs_i.grad_fn)
                valid_i = valid_i_list[i]
                # print (valid_i, len(valid_i), probs_i)
                if valid_i is not None:
                    
                    probs_i = probs_i * valid_i
            
                probs_i_summed = probs_i.sum(1)
                # print (probs_i_summed.mean())
                masked_probs.append(probs_i_summed)

            if i < ncols - 1:
                # Num samples to draw for column i.
                if i != 0:
                    num_i = 1
                else:
                    num_i = num_samples 
                # if (self.shortcircuit and operators[column_idx] is None) or not do_sample:
                #     data_to_encode = None
                # else:
                # print ('probs_i.shape', probs_i.shape)
                '''
                Gumbel-softmax
                '''
                import random
                # paths_vanished = (probs_i.sum(1) <= 0).view(-1, 1)
                # probs_i = probs_i.masked_fill_(paths_vanished, 1.0)
                # tau = torch.tensor([100], device=self.device)
                # tau.requires_grad = False
                # y_i = RelaxedOneHotCategorical(tau, logits=probs_i)
                # y_i = y_i.rsample()
                y_i = probs_i
                # print (y_i.shape, probs_i.shape)
                # y_i = y_i.masked_fill_((y_i.sum(1) <= 0).view(-1, 1), 1.0)
                samples_i = torch.multinomial(
                    y_i, num_samples=num_i,
                    replacement=True).view(-1, 1)  # [bs, num_i]
                data_to_encode = samples_i

                # print (y_hard.shape, data_to_encode.shape)
                y_hard = torch.zeros((data_to_encode.shape[0], y_i.shape[1]), device=self.device)
                # print (y_hard.shape, data_to_encode.shape)
                y_hard.scatter_(1, data_to_encode, 1)
                # print ('y_i', y_i.requires_grad, y_i.grad_fn)
                y_hard = (y_hard - y_i.detach() + y_i)
                # print ('y_hard', y_hard.requires_grad, y_hard.grad_fn)
                inp = this_model.EncodeInput(
                    y_hard,
                    natural_col=natural_idx,
                    out=inp)
                # print ('inp', inp.requires_grad, inp.grad_fn)
                # Actual forward pass.
                next_natural_idx  = i + 1 if ordering is None else ordering[i + 1]
                next_column_idx = self.table.rev_index[next_natural_idx] 
                logits = this_model.forward_with_encoded_input(inp)
        # print ('masked_probs size: ', len(masked_probs))
        if len(masked_probs) == 1:
            p = masked_probs[0]
            print (p.mean())
        else:
            p = masked_probs[1]
            for ls in masked_probs[2:]:
                p = p * ls
            p = p * masked_probs[0]
        return p.mean()
        # return valid_i_list, samples_i_list, sub_col, sub_col_border, do_sample_list

    def Query(self, columns, operators, vals):
        # Massages queries into natural order.
        for val in vals:
            if len(val) == 2 and val[0] > val[1] :
                return 0
        columns, operators, vals = FillInUnqueriedColumns(
            self.table, columns, operators, vals)
        # TODO: we can move these attributes to ctor.
        ordering = None
        if hasattr(self.model, 'orderings'):
            ordering = self.model.orderings[0]
            orderings = self.model.orderings
        elif hasattr(self.model, 'm'):
            # MADE.
            ordering = self.model.m[-1]
            orderings = [self.model.m[-1]]
        else:
            print('****Warning: defaulting to natural order')
            ordering = np.arange(len(columns))
            orderings = [ordering]

        num_orderings = len(orderings)

        # order idx (first/second/... to be sample) -> x_{natural_idx}.
       
        inv_ordering = [None] * len(self.table.split_col_size)
        for natural_idx in range(len(self.table.split_col_size)):
            inv_ordering[ordering[natural_idx]] = natural_idx
    
        # Fast (?) path.
        # print (num_orderings)
        if num_orderings == 1:
            ordering = orderings[0]
            self.OnStart()
            inp_buf = torch.zeros_like(self.inp, device=self.device)
            p = self._sample_n(
                self.num_samples,
                inv_ordering,
                columns,
                operators,
                vals,
                inp=inp_buf,is_train=True)
            self.OnEnd()
            # print ('here', p)
            return p
            
        ps = []
        self.OnStart()
        for ordering in orderings:
            p_scalar = self._sample_n(self.num_samples // num_orderings,
                                        ordering, columns, operators, vals)
            ps.append(p_scalar)
        self.OnEnd()
        return np.mean(ps)

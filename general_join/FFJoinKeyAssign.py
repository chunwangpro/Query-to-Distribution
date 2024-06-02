import pandas as pd
import networkx as nx

import numpy as np
import itertools
import random

import functools

import heapq
import math
import pickle
import time

def construct_graph(nodes, edges):
    G = nx.Graph()
    for node in nodes:
        G.add_node(node)
    for link in edges:
        G.add_edge(link[0], link[1], weight=1)
    return G

def get_key(edge):
    if edge[0].split('_')[0] == 'left':
        left_id = int(edge[0].split('_')[1])
        right_id = int(edge[1].split('_')[1])
    else:
        right_id = int(edge[0].split('_')[1])
        left_id = int(edge[1].split('_')[1])
    left = movie_info.loc[left_id,['info_type_id', 'info', 'note_right']].to_list()
    right = movie_companies.loc[right_id, ['company_id', 'company_type_id', 'note_left']].to_list()
    return (left[0], left[1], right[0], right[1])

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def add_edge_cost(subgraphs, inverted_components, edge, min_val):
    component_id_1 = inverted_components[edge[0]]
    component_id_2 = inverted_components[edge[1]]
    if component_id_1 == component_id_2:
        return 0
    else:
        return len(subgraphs[component_id_1].nodes) * len(subgraphs[component_id_2].nodes)
        # return subgraphs[component_id_1].left_count * subgraphs[component_id_2].right_count + subgraphs[component_id_1].right_count * subgraphs[component_id_2].left_count

def add_edge_cost_bipart(subgraphs, inverted_components, edge):
    component_id_1 = inverted_components[edge[0]]
    component_id_2 = inverted_components[edge[1]]
    if component_id_1 == component_id_2:
        return sigmoid(len(subgraphs[component_id_1].nodes))
    else:
        return len(subgraphs[component_id_1].nodes) * len(subgraphs[component_id_2].nodes)
        left_1 = 0
        right_1 = 0
        left_2 = 0
        right_2 = 0
        for node in subgraphs[component_id_1].nodes:
            if node.split('_')[0] == 'left':
                left_1 += 1
            else:
                right_1 += 1
        for node in subgraphs[component_id_2].nodes:
            if node.split('_')[0] == 'left':
                left_2 += 1
            else:
                right_2 += 1
        return (left_1 * right_2 + left_2 * right_1)

INFINIT = 9999999999

def update_weight(G, distinct_row_edges, inverted, subgraphs):
    for edge in G.edges:
        key = get_key(edge)
        # weight = 1.0 / (len(distinct_row_edges[key]) + 1.0)
        weight = INFINIT
        for e in distinct_row_edges[key]:
            w = add_edge_cost(subgraphs, inverted, e)
            # print (e, w, weight)
            if w < weight:
                weight = w
        # weight = 1
        G[edge[0]][edge[1]]["weight"] = weight

def update_weight_balance(G, distinct_row_edges, inverted, subgraphs):
    for edge in G.edges:
        key = get_key(edge)
        weight = (1.0 / (len(distinct_row_edges[key]) + 1.0)) ** 2
        G[edge[0]][edge[1]]["weight"] = weight

def get_connected_component(G):
    components = nx.connected_components(G)
    inverted_components = {}
    subgraphs = []
    for idx, nodes in enumerate(components):
        left_count = 0
        right_count = 0
        for node in nodes:
            inverted_components[node] = idx
            if node.split('_')[0] == 'left':
                left_count += 1
            else:
                right_count += 1
        subgraphs.append(G.subgraph(nodes).copy())
        subgraphs[-1].left_count = left_count
        subgraphs[-1].right_count = right_count
    return subgraphs, inverted_components

def cmp(a, b):
    return a[1] - b[1]

def unselected_edge_percentage(partition):
    left_size = 0
    right_size = 0
    for node in partition.nodes:
        if node.split('_')[0] == 'left':
            left_size += 1
        else:
            right_size += 1
    if left_size * right_size == 0:
        return 1.0, 0
    return 1.0 - float(len(partition.edges)) / left_size * right_size, left_size * right_size

def mini_bipartition_component(G, epoch, selected_pairs, remain_pairs):
    subgraphs, inverted_components = get_connected_component(G)
    subgraphs_with_score = [(part, *unselected_edge_percentage(part)) for part in subgraphs]
    pairs_sum = 0
    for p in subgraphs_with_score:
        pairs_sum += p[2]
    print (pairs_sum, pairs_sum / float(len(G.edges)), len(subgraphs))
    for subgraph in sorted(subgraphs_with_score, key=functools.cmp_to_key(cmp))[0:1]:
        subgraph = subgraph[0]
        # part1, part2 = nx.community.kernighan_lin_bisection(subgraph, max_iter=20, weight='weight')
        sum, (part1, part2) = nx.stoer_wagner(subgraph, weight='weight', heap=nx.utils.heaps.PairingHeap)
        print (len(subgraph.nodes), len(part1), len(part2))
        removed_edges = []
        for edge in subgraph.edges:
            if not (edge[0] in part1 and edge[1] in part1 or edge[0] in part2 and edge[1] in part2):
                removed_edges.append(edge)
                print (edge, G[edge[0]][edge[1]]['weight'])
                G.remove_edge(*edge)
        subgraphs, inverted_components = get_connected_component(G)
        print (len(removed_edges))
        for idx, edge in enumerate(removed_edges):
            # print (idx, len(removed_edges))
            key = get_key(edge)
            pairs = selected_pairs[key]
            if edge in pairs:
                selected_pairs[key].remove(edge)
            elif (edge[1], edge[0]) in pairs:
                selected_pairs[key].remove((edge[1], edge[0]))
            remain_pairs[key].add(edge)
            sorted_add_edges = sorted([(e, add_edge_cost(subgraphs, inverted_components, e)) for e in remain_pairs[key]], key=functools.cmp_to_key(cmp))
            add_edge = sorted_add_edges[0][0]
            print ('add: ', sorted_add_edges[0][0], sorted_add_edges[0][1])
            selected_pairs[key].add(add_edge)
            pairs = remain_pairs[key]
            if add_edge in pairs:
                remain_pairs[key].remove(add_edge)
            elif (add_edge[1], add_edge[0]) in pairs:
                remain_pairs[key].remove((add_edge[1], add_edge[0]))
            # G.add_edge(*add_edge, weight = (1.0 / (len(remain_pairs[key]) + 1.0)) ** 2)
            if len(sorted_add_edges) > 1:
                G.add_edge(*add_edge, weight = sorted_add_edges[1][1])
            else:
                G.add_edge(*add_edge, weight = INFINIT)
            # if sorted_add_edges[0][1] >= max(len(part1), len(part2)):
            #     print ('Updates All Weights')
            #     update_weight(G, remain_pairs, inverted_components, subgraphs)
            # G.add_edge(*add_edge, weight = 1)
            if idx % 100 == 0:
                subgraphs, inverted_components = get_connected_component(G)

def balance_bipartition_component(G, epoch, selected_pairs, remain_pairs):
    subgraphs, inverted_components = get_connected_component(G)
    subgraphs_with_score = [(part, *unselected_edge_percentage(part)) for part in subgraphs]
    pairs_sum = 0
    for p in subgraphs_with_score:
        pairs_sum += p[2]
    print (pairs_sum, pairs_sum / float(len(G.edges)), len(subgraphs))

    subgraph = sorted(subgraphs_with_score, key=functools.cmp_to_key(cmp))[0][0]
    part1, part2 = nx.community.kernighan_lin_bisection(subgraph, max_iter=20, weight='weight')
    # sum, (part1, part2) = nx.stoer_wagner(subgraph, weight='weight', heap=nx.utils.heaps.PairingHeap)
    print (len(subgraph.nodes), len(part1), len(part2))
    removed_edges = []
    for edge in subgraph.edges:
        if not (edge[0] in part1 and edge[1] in part1 or edge[0] in part2 and edge[1] in part2):
            removed_edges.append(edge)
            G.remove_edge(*edge)
    subgraphs, inverted_components = get_connected_component(G)
    print (len(removed_edges))
    for idx, remove_edge in enumerate(removed_edges):
        key = get_key(remove_edge)
        pairs = selected_pairs[key]
        if remove_edge in pairs:
            selected_pairs[key].remove(remove_edge)
        elif (remove_edge[1], remove_edge[0]) in pairs:
            selected_pairs[key].remove((remove_edge[1], remove_edge[0]))
        else:
            raise Exception('Error: {} Not In Selected Pairs'.format(remove_edge))
        remain_pairs[key].add(remove_edge)
        
        sorted_add_edges = sorted([(e, add_edge_cost_bipart(subgraphs, inverted_components, e)) for e in remain_pairs[key]], key=functools.cmp_to_key(cmp))
        add_edge = sorted_add_edges[0][0]
        selected_pairs[key].add(add_edge)
        pairs = remain_pairs[key]
        if add_edge in pairs:
            remain_pairs[key].remove(add_edge)
        elif (add_edge[1], add_edge[0]) in pairs:
            remain_pairs[key].remove((add_edge[1], add_edge[0]))
        else:
            raise Exception('Error: {} Not In Remain Pairs'.format(add_edge))
        G.add_edge(*add_edge, weight = (1.0 / (len(remain_pairs[key]) + 1.0)) ** 2)
        # if len(sorted_add_edges) > 1:
        #     G.add_edge(*add_edge, weight = sorted_add_edges[1][1])
        # else:
        #     G.add_edge(*add_edge, weight = INFINIT)
        # G.add_edge(*add_edge, weight = 1)
        if idx % 100 == 0:
            subgraphs, inverted_components = get_connected_component(G)

movie_info = pd.read_csv('../datasets/movie_info.csv')
movie_companies = pd.read_csv('../datasets/movie_companies.csv')
movie_info[['info']] = movie_info[['info']].astype(str)
movie_info[['note']] = movie_info[['note']].astype(str)
movie_companies[['note']] = movie_companies[['note']].astype(str)

joined = movie_companies.join(movie_info.set_index('movie_id'), on='movie_id', how='inner', lsuffix='_left', rsuffix='_right')
sample_joined = joined.iloc[0:1000000]
movie_companies = sample_joined[['id_left', 'movie_id', 'company_id', 'company_type_id', 'note_left']].drop_duplicates().set_index('id_left')
movie_info = sample_joined[['id_right', 'movie_id', 'info_type_id', 'info', 'note_right']].drop_duplicates().set_index('id_right')

print ('Data Sampling Completed!')

def run_greedy():
    print ('Greedy Algorithm!')
    movie_info_distinct = movie_info.groupby(['info_type_id', 'info', 'note_right']).groups
    movie_companies_distinct = movie_companies.groupby(['company_id', 'company_type_id', 'note_left']).groups
    joined_distinct = sample_joined.groupby(['info_type_id', 'info', 'note_right', 'company_id', 'company_type_id', 'note_left']).groups
    print (len(movie_info_distinct), len(movie_companies_distinct), len(joined_distinct))
    print ('Data Grouping Completed!')

    edges = []
    nodes = []
    import random
    l_mi = len(movie_info_distinct)
    l_mc = len(movie_companies_distinct)
    for idx, (mi_distinct, mi_ids) in enumerate(movie_info_distinct.items()):
        print ('{}/{}/{}'.format(idx, l_mi, len(mi_ids)))
        for mc_distinct, mc_ids in movie_companies_distinct.items():
            key = mi_distinct + mc_distinct
            if key in joined_distinct:
                total_num = len(joined_distinct[key])
                for i in mi_ids:
                    nodes.append('left_'+str(i))
                for i in mc_ids:
                    nodes.append('right_'+str(i))
                if len(mi_ids) > 2000:
                    mi_ids = [mi_ids[i] for i in random.sample(range(len(mi_ids)), 2000)]
                all_pairs = set(itertools.product(['left_'+str(x) for x in mi_ids], ['right_'+str(x) for x in mc_ids]))
                size = len(joined_distinct[key])
                edges.append((len(all_pairs) / float(size), key, all_pairs, size))
    # edges = sorted(edges, key=functools.cmp_to_key(cmp))
    heapq.heapify(edges)
    G = construct_graph(nodes, [])
    subgraphs, inverted_components = get_connected_component(G)

    start = time.time()
    selected = 0
    while len(edges) > 0:
        if selected % 100 == 0:
            print ('Complete: ', selected)
        edge = heapq.heappop(edges)
        _, key, pairs, size = edge
        if size > 0:
            min_pair = None
            min_val = INFINIT
            if len(pairs) > 100000:
                consider_pairs = random.sample(pairs, 100000)
            else:
                consider_pairs = pairs
            # consider_pairs = pairs
            for pair in consider_pairs:
                x = add_edge_cost(subgraphs, inverted_components, pair, min_val)
                if x < 20:
                    min_pair = [pair]
                    break
                if x < min_val:
                    min_val = x
                    min_pair = [pair]
            # labeled_pairs = [(pair, add_edge_cost(subgraphs, inverted_components, pair)) for pair in pairs]
            # min_pair = min(labeled_pairs, key=functools.cmp_to_key(cmp))
            left_node = min_pair[0][0]
            right_node = min_pair[0][1]
            left_group_id = inverted_components[left_node]
            right_group_id = inverted_components[right_node]
            if left_group_id == right_group_id:
                subgraphs[left_group_id].add_edge(*min_pair[0])
            else:
                left_left_cnt = subgraphs[left_group_id].left_count
                left_right_cnt = subgraphs[left_group_id].right_count
                right_left_cnt = subgraphs[right_group_id].left_count
                right_right_cnt = subgraphs[right_group_id].right_count
                merged = nx.union(subgraphs[left_group_id],subgraphs[right_group_id])
                merged.left_count = left_left_cnt + right_left_cnt
                merged.right_count = left_right_cnt + right_right_cnt
                merged.add_edge(*min_pair[0])
                if len(subgraphs[right_group_id].nodes) <= len(subgraphs[left_group_id].nodes):
                    subgraphs[left_group_id] = merged
                    for n in subgraphs[right_group_id].nodes:
                        inverted_components[n] = left_group_id
                else:
                    subgraphs[right_group_id] = merged
                    for n in subgraphs[left_group_id].nodes:
                        inverted_components[n] = right_group_id
            pairs.remove(min_pair[0])
            heapq.heappush(edges, (len(pairs) / float(size), key, pairs, size-1))
            selected += 1
    end = time.time()
    print ('Time: {}s'.format(end - start))
    G = None
    for group_id in set(inverted_components.values()):
        if G is None:
            G = subgraphs[group_id]
        else:
            G = nx.compose(G, subgraphs[group_id])
    with open('graph_greedy2.pkl', 'wb') as f:
        pickle.dump(G, f)

def run(construct_graph_from_scrach, do_bipart):
    if construct_graph_from_scrach:
        movie_info_distinct = movie_info.groupby(['info_type_id', 'info']).groups
        movie_companies_distinct = movie_companies.groupby(['company_id', 'company_type_id']).groups
        joined_distinct = sample_joined.groupby(['info_type_id', 'info', 'company_id', 'company_type_id']).groups

        print ('Data Grouping Completed!')

        edges = {}
        selected_pairs = {}
        remain_pairs = {}
        nodes = []
        for mi_distinct, mi_ids in movie_info_distinct.items():
            for mc_distinct, mc_ids in movie_companies_distinct.items():
                key = mi_distinct + mc_distinct
        #         print (key)
                if key in joined_distinct:
                    size = len(joined_distinct[key])
                    nodes += ['left_'+str(x) for x in mi_ids]
                    nodes += ['right_'+str(x) for x in mc_ids]
                else:
                    continue
                all_pairs = np.array(list(itertools.product(['left_'+str(x) for x in mi_ids], ['right_'+str(x) for x in mc_ids])))
                edges[key] = all_pairs
        #         print (all_pairs)
                if size < len(all_pairs):
                    selected_ids = random.sample(range(len(all_pairs)), size)
                    remain_ids = list(set(range(len(all_pairs))) - set(selected_ids))
                    selected_pairs[key] = set([(str(x[0]), str(x[1])) for x in all_pairs[selected_ids]])
                    remain_pairs[key] = set([(str(x[0]), str(x[1])) for x in all_pairs[remain_ids]])
                    edges[key] = set([(str(x[0]), str(x[1])) for x in all_pairs[selected_ids]])
                else:
                    selected_pairs[key] = set([(str(x[0]), str(x[1])) for x in all_pairs])
                    remain_pairs[key] = set([])
                    edges[key] = set([(str(x[0]), str(x[1])) for x in all_pairs])
        x = []
        for group in edges.values():
            x += group
        G = construct_graph(nodes, x)
        print ('Construction Completed!')

    else:
        with open('current_graph_finetune2.pkl', 'rb') as f:
                G, selected_pairs, remain_pairs, epoch = pickle.load(f)
        print ('Graph Loaded!')
    subgraphs, inverted_components = get_connected_component(G)

    if do_bipart:
        update_weight_balance(G, remain_pairs, inverted_components, subgraphs)
        print ('Weights Update Completed!')
        for epoch in range(10000):
            balance_bipartition_component(G, epoch, selected_pairs, remain_pairs)
            with open('current_graph_bipart.pkl', 'wb') as f:
                pickle.dump((G, selected_pairs, remain_pairs, epoch), f)
    else:
        update_weight(G, remain_pairs, inverted_components, subgraphs)
        print ('Weights Update Completed!')
        for epoch in range(10000):
            mini_bipartition_component(G, epoch, selected_pairs, remain_pairs)
            with open('current_graph_finetune3.pkl', 'wb') as f:
                pickle.dump((G, selected_pairs, remain_pairs, epoch), f)

run_greedy()


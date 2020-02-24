# -*-coding:utf8-*-
"""
author:Magicq
date:2020***
personalrank main algo
"""
from __future__ import division
from scipy.sparse import coo_matrix
import os
import operator
from scipy.sparse.linalg import gmres
import numpy as np


def get_graph_from_data(input_file):
    """
    Args:
    input_file: ratings.txt
    Return:
         a dict : {UserA:{itemb:1,itemc:1},itemb:{UserA:1}}
    """
    if not os.path.exists(input_file):
        return {}
    fp = open(input_file)
    line_num = 0
    score_thr = 4.0
    graph = {}
    for line in fp:
        if line_num == 0:
            line_num += 1
            continue
        item = line.strip().split(",")
        if len(item) < 3:
            continue
        userid, itemid, rating = item[0], "item_" + item[1], item[2]
        if float(rating) < score_thr:
            continue
        if userid not in graph:
            graph[userid] = {}
        graph[userid][itemid] = 1
        if itemid not in graph:
            graph[itemid] = {}
        graph[itemid][userid] = 1
    fp.close()
    return graph


def personal_rank(graph, root, alpha, item_num, recom_num=10):
    """
    Args:
        graph: 用户行为数据二分图
        root: 推荐用户（顶点）
        alpha: 正则化系数，以alph概率进行随机游走，以1-alph的概率回到起点
        item_num: 迭代次数
        recom_num: 推荐物品数
    Return:
        a dict key itemid , value pr
    """

    """
    算法原理：
        root初始化：pr顶点初始化为0,将推荐用户作为算法顶点root,设值为1.
        开始递归计算：首轮计算顶点及与之相关的item会计算出pr值,然后继续迭代,不断将更多顶点及其相关的顶点pr也计算出来.
        可见二分图算法是以某个顶点出发,每次迭代计算(概率法)其直接关联的顶点,最终计算出全部节点的pr值.
        算法收敛条件：某次迭代后所有顶点的pr值与上一次计算后的顶点pr值相同,提前收敛.
        递归次数：递归次数过少,可能会导致item未参与pr值计算,导致推荐结果过少.
        例如：下面的数据，对A进行推荐,首轮计算A,a,b,d顶点hr会被计算出来,如果只一轮迭代,则item_c和item_e均为0
        userid,itemdid,ratings
        A,a,4.5
        A,b,4.5
        A,d,4.5
        B,a,4.5
        B,c,4.5
        C,b,4.5
        C,e,4.5
        D,c,4.5
        D,d,4.5
    """
    rank = {}
    rank = {point: 0 for point in graph}
    rank[root] = 1
    recom_result = {}
    for iter_index in range(item_num):
        tmp_rank = {}
        tmp_rank = {point: 0 for point in graph}
        for out_point, out_dict in graph.items():
            for inner_point, value in graph[out_point].items():
                tmp_rank[inner_point] += round(alpha * rank[out_point] / len(out_dict), 4)
                if inner_point == root:
                    tmp_rank[inner_point] += round(1 - alpha, 4)
        if tmp_rank == rank:
            print("out============" + str(iter_index))
            break
        rank = tmp_rank
    right_num = 0
    for zuhe in sorted(rank.items(), key=operator.itemgetter(1), reverse=True):
        point, pr_score = zuhe[0], zuhe[1]
        if len(point.split('_')) < 2:
            continue
        if point in graph[root]:
            continue
        recom_result[point] = round(pr_score, 4)
        right_num += 1
        if right_num > recom_num:
            break
    return recom_result


def graph_to_m(graph):
    """
    Args:
        graph:user item graph
    Return:
        a coo_matrix, sparse mat M
        a list, total user item point
        a dict, map all the point to row index
    """
    vertex = []
    for v, v_dic in graph.items():
        vertex.append(v)
    address_dict = {}
    total_len = len(vertex)
    for index in range(len(vertex)):
        address_dict[vertex[index]] = index
    row = []
    col = []
    data = []
    for element_i in graph:
        weight = round(1 / len(graph[element_i]), 3)
        row_index = address_dict[element_i]
        for element_j in graph[element_i]:
            col_index = address_dict[element_j]
            row.append(row_index)
            col.append(col_index)
            data.append(weight)
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    m = coo_matrix((data, (row, col)), shape=(total_len, total_len))
    return m, vertex, address_dict


def mat_all_point(m_mat, vertex, alpha):
    """
    get E-alpha*m_mat.T
    Args:
        m_mat:
        vertex: total item and user point
        alpha: the prob for random walking
    Return:
        a sparse
    """
    total_len = len(vertex)
    row = []
    col = []
    data = []
    for index in range(total_len):
        row.append(index)
        col.append(index)
        data.append(1)
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    eye_t = coo_matrix((data, (row, col)), shape=(total_len, total_len))
    return eye_t.tocsr() - alpha * m_mat.tocsr().transpose()


def personal_rank_mat(graph, root, alpha, recom_num=10):
    """
    Args:
        graph:user item graph
        root:the fix user to recom
        alpha:the prob to random walk
        recom_num:recom item num
    Return:
        a dict, key: itemid, value: pr score
    A*r = r0
    """
    m, vertex, address_dict = graph_to_m(graph)
    if root not in address_dict:
        return {}
    score_dict = {}
    recom_dict = {}
    mat_all = mat_all_point(m, vertex, alpha)
    index = address_dict[root]
    initial_list = [[0] for row in range(len(vertex))]
    initial_list[index] = [1]
    r_zero = np.array(initial_list)
    # 解线性方程
    res = gmres(mat_all, r_zero, tol=1e-8)[0]
    for index in range(len(res)):
        point = vertex[index]
        if len(point.strip().split("_")) < 2:
            continue
        if point in graph[root]:
            continue
        score_dict[point] = round(res[index], 3)
    for zuhe in sorted(score_dict.items(), key=operator.itemgetter(1), reverse=True)[:recom_num]:
        point, score = zuhe[0], zuhe[1]
        recom_dict[point] = score
    return recom_dict


def get_one_user_recom():
    """
     give one fix user recom result
    """
    user = "1"
    alpha = 0.8
    graph = get_graph_from_data("../data/ratings.txt")
    item_num = 100
    return personal_rank(graph, user, alpha, item_num, 100)


def get_one_user_by_mat():
    """
    give one fix user by mat
    """
    user = "1"
    alpha = 0.8
    graph = get_graph_from_data("../data/ratings.txt")
    recom_result = personal_rank_mat(graph, user, alpha, 10)
    return recom_result


if __name__ == '__main__':
    recom_result_base = get_one_user_recom()
    recom_result_mat = get_one_user_by_mat()
    print(recom_result_mat)
    num = 0
    for ele in recom_result_base:
        if ele in recom_result_mat:
            num += 1
    # print(num)

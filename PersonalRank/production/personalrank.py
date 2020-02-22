# -*-coding:utf8-*-
"""
author:Magicq
date:2020***
personalrank main algo
"""
from __future__ import division
import os
import sys
import operator


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


def get_one_user_recom():
    """
     give one fix user recom result
    """
    user = "4"
    alpha = 0.8
    graph = get_graph_from_data("../data/ratings.txt")
    item_num = 100
    return personal_rank(graph, user, alpha, item_num, 9)


if __name__ == '__main__':
    print(get_one_user_recom())

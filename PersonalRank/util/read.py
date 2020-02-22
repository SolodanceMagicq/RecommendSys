# -*-coding:utf8-*-
"""
author:Magicq
date:2020***
get graph from user data
"""
import os


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


if __name__ == '__main__':
    graph = get_graph_from_data("../data/ratings.txt")
    print(graph["1"])

# *-*coding:utf8*-*
"""
author:Magicq
date:2020***
get up and online recommendation
"""
from __future__ import division
import os
import operator
import sys

sys.path.append("../")
import util.read as read


def get_up(item_cate, input_file):
    """
    Arges:
        item_cate: key itemid,value:dict,key categery value ratio
        input_file: user rating file
    Rreturn:
        a dict: key userid, value [(categery,ratio1),(categery2,ratio2)]
    """
    if not os.path.exists(input_file):
        return {}
    record = {}
    linenum = 0
    score_thr = 4.0
    up = {}
    fp = open(input_file)
    for line in fp:
        if linenum == 0:
            linenum += 1
            continue
        item = line.strip().split(",")
        if len(item) < 4:
            continue
        userid, itemid, rating, timestamp = item[0], item[1], float(item[2]), int(item[3])
        if rating < score_thr:
            continue
        if itemid not in item_cate:
            continue
        time_score = get_time_score(timestamp)
        if userid not in record:
            record[userid] = {}
        for fix_cate in item_cate[itemid]:
            if fix_cate not in record[userid]:
                record[userid][fix_cate] = 0
            record[userid][fix_cate] += rating * time_score * item_cate[itemid][fix_cate]
    fp.close()
    for userid in record:
        if userid not in up:
            up[userid] = []
        total_score = 0
        for zuhe in sorted(record[userid].iteritems(), key=operator.itemgetter(1), reverse=True)[:topk]:
            up[userid].append((zuhe[0], zuhe[1]))
            total_score += zuhe[1]
        for index in range(len(up[userid])):
            up[userid][index] = (up[userid][index][0], round(up[userid][index][1] / total_score, 3))
    return up


def get_time_score(timestamp):
    """
    Args:
        timestamp:input timestamp
    Return:
        time score
    """
    fix_time_stamp = 1476086345
    total_sec = 24 * 60 * 60
    delta = (fix_time_stamp - timestamp) / total_sec / 100
    return round(1 / (1 + delta), 3)


if __name__ == '__main__':
    pass

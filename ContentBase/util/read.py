# -*-coding:utf8-*-
"""
Author:Magicq
date:2020***
some util func
"""
from __future__ import division
import os


def get_avg_score(input_path):
    """
    Args:
        input_path:
    Return:
         dic: key:itemid value:avg_score
    """

    if not os.path.exists(input_path):
        return
    line_num = 0
    record = {}
    avg_score = {}
    fp = open(input_path)
    for line in fp:
        if line_num == 0:
            line_num += 1
            continue
        item = line.strip().split(",")
        if len(item) < 4:
            continue
        userid, itemid, rating = item[0], item[1], float(item[2])
        if itemid not in record:
            # [0,0] total_score and score_num
            record[itemid] = [0, 0]
        record[itemid][0] += rating
        record[itemid][1] += 1
    fp.close()
    for itemid in record:
        avg_score[itemid] = round(record[itemid][0] / record[itemid][1], 3)
    return avg_score

def get_item_cate(avg_score,input_file):
    """
    Args:
        avg_score:
        input_file:
    Return:
        a dict: key itemid value a dict ,key :cate value:ratio
        a dict: key cate value [itemid1,itemid2,itemid3]
    """
    pass

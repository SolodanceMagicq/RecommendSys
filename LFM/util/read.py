#-*-coding:utf8-*-
"""
author:david
date:2018****
util function
"""

import os


def get_item_info(input_file):
    """
    get item info:[title, genre]
    Args:
        input_file:item info file
    Return:
        a dict: key itemid, value:[title, genre]
    """
    if not os.path.exists(input_file):
        return {}
    item_info = {}
    linenum = 0
    fp = open(input_file)
    for line in fp:
        if linenum == 0:
            linenum += 1
            continue
        item = line.strip().split(',')
        if len(item) < 3:
            continue
        elif len(item) == 3:
            itemid, title, genre = item[0], item[1], item[2]
        elif len(item) > 3:
            itemid = item[0]
            genre = item[-1]
            title = ",".join(item[1:-1])
        item_info[itemid] = [title, genre]
    fp.close()
    return item_info


def get_ave_score(input_file):
    """
    get item ave rating score
    Args:
        input file: user rating file
    Return:
        a dict, key:itemid, value:ave_score
    """
    if not os.path.exists(input_file):
        return {}
    linenum = 0
    record_dict = {}
    score_dict = {}
    fp = open(input_file)
    for line in fp:
        if linenum == 0:
            linenum += 1
            continue
        item = line.strip().split(',')
        if len(item) < 4:
            continue
        userid, itemid, rating = item[0], item[1], float(item[2])
        if itemid not in record_dict:
            record_dict[itemid] = [0, 0]
        record_dict[itemid][0] += 1
        record_dict[itemid][1] += rating
    fp.close()
    for itemid in record_dict:
        score_dict[itemid] = round(record_dict[itemid][1]/record_dict[itemid][0], 3)
    return score_dict

def get_train_data(input_file):
    """
    get train data for LFM model train
    Args:
        input_file: user item rating file
    Return:
        a list:[(userid, itemid, label), (userid1, itemid1, label)]
    """
    if not os.path.exists(input_file):
        return []
    score_dict = get_ave_score(input_file)
    neg_dict = {}
    pos_dict = {}
    train_data = []
    linenum = 0
    score_thr = 4.0
    fp = open(input_file)
    for line in fp:
        if linenum == 0:
            linenum += 1
            continue
        item = line.strip().split(',')
        if len(item) < 4:
            continue
        userid, itemid, rating = item[0], item[1], float(item[2])
        if userid not in pos_dict:
            pos_dict[userid] = []
        if userid not in neg_dict:
            neg_dict[userid] = []
        if rating >= score_thr:
            pos_dict[userid].append((itemid, 1))
        else:
            score = score_dict.get(itemid, 0)
            neg_dict[userid].append((itemid, score))
    fp.close()
    for userid in pos_dict:
        data_num = min(len(pos_dict[userid]), len(neg_dict.get(userid, [])))
        if data_num > 0:
            train_data += [(userid, zuhe[0], zuhe[1]) for zuhe in pos_dict[userid]][:data_num]
        else:
            continue
        sorted_neg_list = sorted(neg_dict[userid], key=lambda element:element[1], reverse=True)[:data_num]
        train_data += [(userid, zuhe[0], 0) for zuhe in sorted_neg_list]
    return train_data






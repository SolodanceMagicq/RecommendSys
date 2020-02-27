# -*-coding:utf8-*-
"""
author:david
date:2018****
lfm model train main function
"""

import numpy as np
import os
import operator


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
        score_dict[itemid] = round(record_dict[itemid][1] / record_dict[itemid][0], 3)
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
        sorted_neg_list = sorted(neg_dict[userid], key=lambda element: element[1], reverse=True)[:data_num]
        train_data += [(userid, zuhe[0], 0) for zuhe in sorted_neg_list]
    return train_data


def lfm_train(train_data, F, alpha, beta, step):
    """
    Args:
        train_data: train_data for lfm
        F: user vector len, item vector len
        alpha:regularization factor
        beta: learning rate
        step: iteration num
    Return:
        dict: key itemid, value:np.ndarray
        dict: key userid, value:np.ndarray
    """
    user_vec = {}
    item_vec = {}
    for step_index in range(step):
        for data_instance in train_data:
            userid, itemid, label = data_instance
            if userid not in user_vec:
                user_vec[userid] = init_model(F)
            if itemid not in item_vec:
                item_vec[itemid] = init_model(F)
            delta = label - model_predict(user_vec[userid], item_vec[itemid])  # 视频讲解中此处代码有误，应该每个样本都更新
            for index in range(F):
                user_vec[userid][index] += beta * (delta * item_vec[itemid][index] - alpha * user_vec[userid][index])
                item_vec[itemid][index] += beta * (delta * user_vec[userid][index] - alpha * item_vec[itemid][index])
        beta = beta * 0.9
    return user_vec, item_vec


def init_model(vector_len):
    """
    Args:
        vector_len: the len of vector
    Return:
         a ndarray
    """
    return np.random.randn(vector_len)


def model_predict(user_vector, item_vector):
    """
    user_vector and item_vector distance
    Args:
        user_vector: model produce user vector
        item_vector: model produce item vector
    Return:
         a num
    """
    res = np.dot(user_vector, item_vector) / (np.linalg.norm(user_vector) * np.linalg.norm(item_vector))
    return res


def model_train_process():
    """
    test lfm model train
    """
    train_data = get_train_data("../data/ratings.txt")
    user_vec, item_vec = lfm_train(train_data, 50, 0.01, 0.1, 50)
    for userid in user_vec:
        recom_result = give_recom_result(user_vec, item_vec, userid)
        # ana_recom_result(train_data, userid, recom_result)


def give_recom_result(user_vec, item_vec, userid):
    """
    use lfm model result give fix userid recom result
    Args:
        user_vec: lfm model result
        item_vec:lfm model result
        userid:fix userid
    Return:
        a list:[(itemid, score), (itemid1, score1)]
    """
    fix_num = 10
    if userid not in user_vec:
        return []
    record = {}
    recom_list = []
    user_vector = user_vec[userid]
    for itemid in item_vec:
        item_vector = item_vec[itemid]
        res = np.dot(user_vector, item_vector) / (np.linalg.norm(user_vector) * np.linalg.norm(item_vector))
        record[itemid] = res
    for zuhe in sorted(record.iteritems(), key=operator.itemgetter(1), reverse=True)[:fix_num]:
        itemid = zuhe[0]
        score = round(zuhe[1], 3)
        recom_list.append((itemid, score))
    return recom_list


def ana_recom_result(train_data, userid, recom_list):
    """
    debug recom result for userid
    Args:
        train_data: train data for lfm model
        userid:fix userid
        recom_list: recom result by lfm
    """
    item_info = get_item_info("../data/movies.txt")
    for data_instance in train_data:
        tmp_userid, itemid, label = data_instance
        if tmp_userid == userid and label == 1:
            print
            item_info[itemid]
    print
    "recom result"
    for zuhe in recom_list:
        print
        item_info[zuhe[0]]


if __name__ == "__main__":
    model_train_process()

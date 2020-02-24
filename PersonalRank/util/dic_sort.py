# -*-coding:utf8-*-
"""
Author:Magicq
date:2020***
dic sorted option
"""

import operator


def dic_map():
    dic = {'a': 10.0, 'b': 2.1, 'c': 3.33, 'd': 4.1, 'e': 2.5}
    result = {}
    for zuhe in sorted(dic.items(), key=operator.itemgetter(1), reverse=True):
        key, value = zuhe[0], zuhe[1]
        if value > 9:
            continue
        result[key] = value
    return result


if __name__ == '__main__':
    result = dic_map()
    print(result)

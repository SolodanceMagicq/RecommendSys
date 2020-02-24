# -*-coding:utf8-*-
"""
Author:Magicq
date:2020***
scipy.sparse.coo_matrix option
"""
from scipy.sparse import coo_matrix
import numpy as np

if __name__ == '__main__':
    row = np.array([0, 0, 1, 3, 1, 0, 0])
    col = np.array([0, 2, 1, 3, 1, 0, 0])
    data = np.array([1, 1, 1, 1, 1, 1, 1])
    coo_matrix = coo_matrix((data, (row, col)), shape=(7, 7))
    row_eye = []
    col_eye = []
    data_eye = []
    for index in range(6):
        row1 = row_eye.append(index)
        col1 = col_eye.append(index)
        data1 = data_eye.append(1)
    eye_t = coo_matrix((data, (row, col)), shape=(7, 7))
    trans_m = eye_t.tocsr() - 0.8 * coo_matrix.tocsr().transpose()
    print(trans_m)

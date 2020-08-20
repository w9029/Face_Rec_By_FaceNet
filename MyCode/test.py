#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pymysql
import numpy as np
from scipy.spatial import distance

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    #output = x / np.sqrt(np.max(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

# a = np.array([[[1,2,3],
#               [4, 2, 3],
#               [1, 2, 7]],
#              [[1, 2, 3],
#              [4, 2, 3],
#              [1, 2, 7]]],dtype='float32')
# b = np.array([[1,2,3],
#               [4, 2, 3],
#               [1, 2, 7]],dtype='float32')

# print(l2_normalize(np.concatenate(b)))
#
# c = b.copy()
# c+=1
# print(b)

a = np.array([[1,2,3,4,5],[1,1,1,1,1]])
b = np.array([1,1,1,1,1])
# print(np.average(a,axis=0))
print(distance.euclidean(a[0],a[1])>1.0)
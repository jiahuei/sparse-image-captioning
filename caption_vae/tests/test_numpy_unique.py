# -*- coding: utf-8 -*-
"""
Created on 06 Oct 2020 23:03:40
@author: jiahuei
"""
import timeit

import_np = "import numpy as np"
test_np = ''' 
def test():
    x = np.random.randint(0, 2, 5000 * 224 * 224)
    return np.unique(x)
'''
test_np_mat = ''' 
def test():
    x = np.random.randint(0, 2, (5000, 224, 224))
    return np.unique(x, axis=0)
'''
import_pd = "import pandas as pd"
test_pd = ''' 
def test():
    x = np.random.randint(0, 2, 5000 * 224 * 224)
    return pd.unique(x)
'''
print(timeit.repeat(stmt=test_np_mat, setup=import_np))
print(timeit.repeat(stmt=test_np, setup=import_np))
print(timeit.repeat(stmt=test_pd, setup=import_pd))

import numpy as np

x = np.random.randint(0, 2, (5000, 4, 4))
x = np.unique(x, axis=0)
print(x.shape)

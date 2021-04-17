# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 17:56:03 2018

@author: Administrator
"""
from sklearn.model_selection import KFold
def cross_val(trainset, K_fold):
    kf = KFold(n_splits=K_fold)
    idx_train = []
    idx_test= []
    for train_index, test_index in kf.split(trainset):
        idx_train.append(train_index), idx_test.append(test_index)
    return idx_test, idx_train
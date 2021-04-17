# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:51:54 2018

@author: Administrator
"""
import numpy as np
from sklearn import preprocessing
def normalization(trainset, testset):
    selector = [x for x in range(trainset.shape[1]) if x != 3]
    trainset_features_scaled = np.matrix(preprocessing.scale(trainset[:,selector]))
    mean = trainset.mean(axis=0)
    var = trainset.var(axis=0)
    testset_scaled = np.zeros((len(testset), testset.shape[1]))
    testset_scaled = np.matrix(testset_scaled)
    for j in range(0,testset.shape[1]):
        if(j != 3):
            testset_scaled[:,j] = (testset[:,j]-mean[0,j])/(var[0,j]**(1/2))
        else:
            testset_scaled[:,j] = testset[:,j]-mean[0,j]
    testset_scaled = np.matrix(testset_scaled)
    
    target_train = trainset[:,3]-mean[0,3]
    testset_features_scaled = testset_scaled[:, selector]
    target_test = testset_scaled[:, 3]
    return trainset_features_scaled, testset_features_scaled, target_train, target_test
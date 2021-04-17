# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 12:22:19 2018

@author: Administrator
"""
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from normalization import normalization

def regression_all_linear(trainset, idx_test, idx_train):
    trainset_features_scaled, testset_features_scaled, target_train, target_test = normalization(trainset[idx_train, :], trainset[idx_test,:])
    
    LR = LinearRegression()
    LR.fit(trainset_features_scaled, target_train)
    E_LR = mean_squared_error(target_test, LR.predict(testset_features_scaled))
    return E_LR
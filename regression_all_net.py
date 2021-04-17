# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 13:44:41 2018

@author: Administrator
"""
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from normalization import normalization

def regression_all_net(trainset, lamda, idx_test, idx_train, ratio):
     trainset_features_scaled, testset_features_scaled, target_train, target_test = normalization(trainset[idx_train, :], trainset[idx_test,:])
     
     net = ElasticNet(alpha = lamda, l1_ratio = ratio)
     net.fit(trainset_features_scaled, target_train)     
     E_net = mean_squared_error(target_test, net.predict(testset_features_scaled))
     return E_net

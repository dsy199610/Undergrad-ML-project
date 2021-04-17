# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 13:33:38 2018

@author: Administrator
"""
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.metrics import mean_squared_error
from normalization import normalization

def regression_all(trainset, lamda, idx_test, idx_train):
    trainset_features_scaled, testset_features_scaled, target_train, target_test = normalization(trainset[idx_train, :], trainset[idx_test,:])
    rdg = Ridge(alpha=lamda)
    rdg.fit(trainset_features_scaled, target_train)
    
    las = Lasso(alpha=lamda)
    las.fit(trainset_features_scaled, target_train)
    
    E_rdg = mean_squared_error(target_test, rdg.predict(testset_features_scaled))
    E_las = mean_squared_error(target_test, las.predict(testset_features_scaled))
    
    return E_rdg, E_las 

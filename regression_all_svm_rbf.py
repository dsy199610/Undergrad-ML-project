# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 17:45:23 2018

@author: Administrator
"""

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from normalization import normalization

def regression_all_svm_rbf(trainset, C_, gamma_, idx_test, idx_train):
    trainset_features_scaled, testset_features_scaled, target_train, target_test = normalization(trainset[idx_train, :], trainset[idx_test,:])
    svm_rbf = SVR(kernel = 'rbf', C = C_, gamma = gamma_)
    svm_rbf.fit(trainset_features_scaled, target_train)
    E_svm_rbf = mean_squared_error(target_test, svm_rbf.predict(testset_features_scaled))
    
    return E_svm_rbf
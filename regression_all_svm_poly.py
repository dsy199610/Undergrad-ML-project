# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 13:06:29 2018

@author: Administrator
"""

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from normalization import normalization

def regression_all_svm_poly(trainset, C_, idx_test, idx_train):
    trainset_features_scaled, testset_features_scaled, target_train, target_test = normalization(trainset[idx_train, :], trainset[idx_test,:])
    
    svm_poly = SVR(kernel = 'poly', C = C_, degree = 2)
    svm_poly.fit(trainset_features_scaled, target_train)
    E_svm_poly = mean_squared_error(target_test, svm_poly.predict(testset_features_scaled))
    
    return E_svm_poly
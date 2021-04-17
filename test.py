# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:24:18 2018

@author: Administrator
"""
from normalization import normalization
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

def test(trainset, testset, lamda_rdg, lamda_las, lamda_net, ratio_net, C_svm_poly, C_svm_rbf, gamma_svm_rbf):
    
    trainset_features_scaled, testset_features_scaled, target_train, target_test = normalization(trainset, testset)
    
    LR = LinearRegression()
    LR.fit(trainset_features_scaled, target_train)
    E_LR = mean_squared_error(target_test, LR.predict(testset_features_scaled))
    
    rdg = Ridge(alpha=lamda_rdg)
    rdg.fit(trainset_features_scaled, target_train)
    E_rdg = mean_squared_error(target_test, rdg.predict(testset_features_scaled))
    
    
    las = Lasso(alpha=lamda_las)
    las.fit(trainset_features_scaled, target_train)
    E_las = mean_squared_error(target_test, las.predict(testset_features_scaled))
    
    net = ElasticNet(alpha = lamda_net, l1_ratio = ratio_net)
    net.fit(trainset_features_scaled, target_train)
    E_net = mean_squared_error(target_test, net.predict(testset_features_scaled))
    
    svm_rbf = SVR(kernel = 'rbf', C = C_svm_rbf, gamma = gamma_svm_rbf)
    svm_rbf.fit(trainset_features_scaled, target_train)
    E_svm_rbf = mean_squared_error(target_test, svm_rbf.predict(testset_features_scaled))
    
    svm_poly = SVR(kernel = 'poly', C = C_svm_rbf, degree = 2)
    svm_poly.fit(trainset_features_scaled, target_train)
    E_svm_poly = mean_squared_error(target_test, svm_poly.predict(testset_features_scaled))
    
    return E_LR, E_rdg, E_las, E_net, E_svm_poly, E_svm_rbf
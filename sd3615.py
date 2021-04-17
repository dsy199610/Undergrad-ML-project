# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from PM2matrix import PM2matrix
from cross_val import cross_val
from regression_cross import regression_cross
from test import test
import numpy as np

trainset, testset = PM2matrix()
K_fold = 10
idx_cross, idx_train_cross = cross_val(trainset, K_fold)

lamda = np.logspace(-8, +8, 20)
ratio = np.logspace(-2, 0, 5)
C = np.logspace(-5, +5, 20)
gamma = np.logspace(-5, +5, 20)


cross_validation_error_LR, cross_validation_error_rdg, cross_validation_error_las, cross_validation_error_net, cross_validation_error_svm_poly, cross_validation_error_svm_rbf, optimal_lamda_rdg, optimal_lamda_las, optimal_lamda_net, optimal_ratio_net, optimal_C_svm_poly, optimal_C_svm_rbf, optimal_gamma_svm_rbf  = regression_cross(trainset, idx_cross, idx_train_cross, lamda, K_fold, ratio, C, gamma)

print("LR CV error: ", cross_validation_error_LR)
print("Ridge CV error and best lamda: ", cross_validation_error_rdg, " ", optimal_lamda_rdg)
print("LASSO CV error and best lamda: ", cross_validation_error_las, " ", optimal_lamda_las)
print("Elastic CV error, best lamda and best ratio: ", cross_validation_error_net, " ", optimal_lamda_net, " ", optimal_ratio_net)
print("svm poly CV error, best C: ", cross_validation_error_svm_poly, " ", optimal_C_svm_poly)
print("svm rbf CV error, best C and best gamma: ", cross_validation_error_svm_rbf, " ", optimal_C_svm_rbf, " ", optimal_gamma_svm_rbf)
test_error_LR, test_error_rdg, test_error_las, test_error_net, test_error_svm_poly, test_error_svm_rbf= test(trainset, testset, optimal_lamda_rdg, optimal_lamda_las, optimal_lamda_net, optimal_ratio_net, optimal_C_svm_poly, optimal_C_svm_rbf, optimal_gamma_svm_rbf)
print("test error LR: ", test_error_LR)
print("test error rdg: ", test_error_rdg)
print("test error las: ", test_error_las)
print("test error net: ", test_error_net)
print("test error svm poly: ", test_error_svm_poly)
print("test error svm rbf: ", test_error_svm_rbf)
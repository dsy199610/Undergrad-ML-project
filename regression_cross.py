# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 17:55:19 2018

@author: Administrator
"""
from regression_all import regression_all
from regression_all_net import regression_all_net
from regression_all_linear import regression_all_linear
from regression_all_svm_poly import regression_all_svm_poly
from regression_all_svm_rbf import regression_all_svm_rbf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def regression_cross(trainset, X_test, X_train, lamda, K_fold, ratio, C, gamma):
    error_LR = np.zeros((K_fold, 1))
    error_rdg = np.zeros((K_fold, len(lamda)))
    error_las = np.zeros((K_fold, len(lamda)))
    error_net = np.zeros((K_fold, len(lamda), len(ratio)))
    error_svm_poly = np.zeros((K_fold, len(C)))
    error_svm_rbf = np.zeros((K_fold, len(C), len(gamma)))
    for i in range (0, K_fold):
        idx_validation = X_test[i]
        idx_train_validation = X_train[i]
        error_LR[i] = regression_all_linear(trainset, idx_validation, idx_train_validation)
        for j in range (0, len(lamda)):
            error_rdg[i, j], error_las[i, j]= regression_all(trainset, lamda[j], idx_validation, idx_train_validation)
            for k in range(0, len(ratio)):
                error_net[i,j,k] = regression_all_net(trainset, lamda[j], idx_validation, idx_train_validation, ratio[k])
        for l in range (0, len(C)):
            error_svm_poly[i,l] = regression_all_svm_poly(trainset, C[l], idx_validation, idx_train_validation)
            for m in range(0, len(gamma)):
                error_svm_rbf[i,l,m] = regression_all_svm_rbf(trainset, C[l], gamma[m], idx_validation, idx_train_validation)
    
    cross_validation_error_LR = error_LR.mean(0)
    validation_error_lamda_rdg = error_rdg.mean(0)
    
    plt.figure()
    plt.semilogx(lamda, validation_error_lamda_rdg)
    plt.title('Cross validaiton error and lamda using Ridge')
    plt.xlabel('lamda')
    plt.ylabel('cross validation error Ridge')
    
    validation_error_lamda_las = error_las.mean(0)
    
    plt.figure()
    plt.semilogx(lamda, validation_error_lamda_las)
    plt.title('Cross validaiton error and lamda using Lasso')
    plt.xlabel('lamda')
    plt.ylabel('cross validation error LASSO')
    
    validation_error_ratio_lamda_net = error_net.mean(0)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.title('Cross validaiton error, lamda and ratio using Elastic Net')
    lamda_, ratio_ = np.meshgrid(lamda, ratio)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-2, 0)
    ax.plot_surface(np.log10(lamda_), np.log10(ratio_), np.transpose(validation_error_ratio_lamda_net))
    ax.set_xlabel('log(lamda)')
    ax.set_ylabel('log(ratio)')
    ax.set_zlabel('cross vlidation error Elastic net')
    
    validation_error_C_svm_poly = error_svm_poly.mean(0)
    
    plt.figure()
    plt.semilogx(C, validation_error_C_svm_poly)
    plt.title('Cross validaiton error and C using svm poly')
    plt.xlabel('C')
    plt.ylabel('cross validation error svm poly')
    
    validation_error_C_gamma_svm_rbf = error_svm_rbf.mean(0)
    
    fig2 = plt.figure()
    ax2 = fig2.gca(projection='3d')
    plt.title('Cross validaiton error, C and gamma using svm rbf')
    C_, gamma_ = np.meshgrid(C, gamma)
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-5, 5)
    ax2.plot_surface(np.log10(C_), np.log10(gamma_), np.transpose(validation_error_C_gamma_svm_rbf))
    ax2.set_xlabel('log(C)')
    ax2.set_ylabel('log(gamma)')
    ax2.set_zlabel('cross vlidation error svm rbf')
    
    optimal_lamda_rdg = lamda [np.argmin(validation_error_lamda_rdg)]
    optimal_lamda_las = lamda [np.argmin(validation_error_lamda_las)]  
    optimal_ratio_net = ratio [np.argmin(validation_error_ratio_lamda_net.min(0))] 
    optimal_lamda_net = lamda [np.argmin(validation_error_ratio_lamda_net.min(1))]  
    optimal_C_svm_poly = C [np.argmin(validation_error_C_svm_poly)]
    optimal_gamma_svm_rbf = gamma [np.argmin(validation_error_C_gamma_svm_rbf.min(0))] 
    optimal_C_svm_rbf = C [np.argmin(validation_error_C_gamma_svm_rbf.min(1))]
    cross_validation_error_rdg = validation_error_lamda_rdg.min()
    cross_validation_error_las = validation_error_lamda_las.min()
    cross_validation_error_net = validation_error_ratio_lamda_net.min()
    cross_validation_error_svm_poly = validation_error_C_svm_poly.min() 
    cross_validation_error_svm_rbf = validation_error_C_gamma_svm_rbf.min()   
    return cross_validation_error_LR, cross_validation_error_rdg, cross_validation_error_las, cross_validation_error_net, cross_validation_error_svm_poly, cross_validation_error_svm_rbf, optimal_lamda_rdg, optimal_lamda_las, optimal_lamda_net, optimal_ratio_net, optimal_C_svm_poly, optimal_C_svm_rbf, optimal_gamma_svm_rbf
    
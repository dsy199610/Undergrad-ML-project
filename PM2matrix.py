# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 12:55:20 2018

@author: Administrator
"""
import pandas
import numpy as np

def PM2matrix():
    data = pandas.read_csv('../data/PM2.5.csv').as_matrix()
      
    i = 0
    while(i < len(data)):
        if(pandas.isnull(data[i,5]) == True):
            data = np.delete(data, i, 0)
        else:
            i += 1

    data = np.delete(data, 0, 1)
  
          
    for i in range(0, len(data)):
        if(data[i, 8] == "NW"):
            data[i, 8] = 1;
        elif(data[i, 8] == "NE"):
            data[i, 8] = 2;
        elif(data[i, 8] == "SE"):
            data[i, 8] = 3;
        elif(data[i, 8] == "SW"):
            data[i, 8] = 4;
        elif(data[i, 8] == "cv"):
            data[i, 8] = 0;
    
    data = np.matrix(data)
    new_data = np.zeros((77)) 
    new_data = np.matrix(new_data)
    for i in range (0, len(data)):
        if(data[i,3] == 8 and i>23):
            new_row = np.concatenate((data[i,0:5], data[(i-32):(i-23), 4:12].reshape(1, 72)), axis=1)
            new_data = np.append(new_data, new_row, axis = 0)
    new_data = np.delete(new_data, 0, 0)
    new_data = np.delete(new_data, 3, 1)

    data1 = new_data[0:1300, :]
    data2 = new_data[1301:len(new_data), :]
    np.random.shuffle(data1)
    np.random.shuffle(data2)
    trainset = np.matrix(data1)
    testset = np.matrix(data2)
    return trainset, testset

#coding=utf-8

import mat4py as mt
import os
import numpy as np

data = mt.loadmat(".//data//Mult-Class Problem.mat")

print(type(data))
#print(data.items())

print(data.keys())

#training_class1 = np.array(data['Training_data'])

#training_class2 = np.array(data['Testing'])

#testing = np.array(data['Testing'])

#testing_label = np.array(data['Label_Testing'])

data_ = []
for key in data.keys():
    data_.append(np.array(data[key]))

#np.array(data_)

for i in range(0,4):
    print(data_[i].shape)

print(data_[2])

#for key in data.items():
# print(key)
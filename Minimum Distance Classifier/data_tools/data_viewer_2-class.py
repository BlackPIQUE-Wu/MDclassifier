#coding=utf-8

import mat4py as mt
import os
import numpy as np

data = mt.loadmat(".//data//2-Class Problem.mat")

print(type(data))
#print(data.items())

print(data.keys())

training_class1 = np.array(data['Training_class1']).T

training_class2 = np.array(data['Training_class2'])

testing = np.array(data['Testing'])

testing_label = np.array(data['Label_Testing'])

data_ = []

data_.append(training_class1)
data_.append(training_class2)
data_.append(testing)
data_.append(testing_label)
#print(type(training_class1))

for i in range(0,4):
    print(data_[i].shape)

print(training_class1[1])
print(data['Training_class1'][1])
#for key in data.items():
# print(key)
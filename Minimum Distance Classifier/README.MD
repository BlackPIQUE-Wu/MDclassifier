# Minimum Distance Classifier
A simple Minimum Distance Classifier including Mahalanobis and Euclidean metric for 2-class and mult-class problems.

## INTRODUCTION

### Name
Minimum Distance Classifier

### Version
v1.0

### Distance Metric included
Mahalanobis, Euclidean

## DATA

### 2-Class Problem
File name: 2-Class Problem.mat \
Training set: 1364 instances \
Testing set: 1364 instances 

### Mult-Class Problem
File name: Mult-Class Problem.mat \
Class number: 17
Training set: 10515 instances \
Testing set: 10510 instances 

## USAGE

### Install
GPU Model (For example, your device has installed CUDA of version 11.6)

    conda create -n mdcls py=3.7
    conda activate name_env
    pip install cupy116 mat4py

CPU Model

    conda create -n mdcls py=3.7
    conda activate mdcls
    pip install numpy mat4py

### Quick Training and Testing
Modifying the value of 'data_dir':

    data_dir = './/data//Mult-Class Problem.mat'

In './Minimum Distance Classifier/':

    python MDClassifier.py

## EVALUATION

### Metirc
Accuracy

### logs
In './runs/logs/' \.
We have given the results demo named 'logs_2-Class_gpu' and 'logs_Mult-Class_gpu'.

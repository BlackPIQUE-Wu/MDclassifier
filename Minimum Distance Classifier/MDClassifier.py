#coding=utf-8

#import numpy as np
import cupy as np
import mat4py
import time
import os
from time import time as now


def training_data_loader(data_dir):
    
    data_origin = mat4py.loadmat(data_dir)
    if data_dir == ".//2-Class Problem.mat":
        data_class1 = np.array(data_origin['Training_class1']).T
        data_class2 = np.array(data_origin['Training_class2']).T
        data = [data_class1,data_class2]
    else:
        data = []
        training_data = np.array(data_origin['Training_data']).T
        training_label = np.array(data_origin['Label_training']).T
        #for i in range(np.max(training_label)):
        for i in range(17):
            data.append([])
        for i in range(len(training_data)):
            data[int(training_label[i])-1].append(training_data[i])
        #for i in range(np.max(training_label)):
        for i in range(17):
            data[i] = np.array(data[i])
            
    return data


def testing_data_loader(data_dir):
    data_origin = mat4py.loadmat(data_dir)
    if data_dir == ".//2-Class Problem.mat":
        data = np.array(data_origin['Testing']).T
        label = np.array(data_origin['Label_Testing']).T
        data = {
            'data_unlabelled': data,
            'label': label
        }
    else:
        data = np.array(data_origin['Testing_data']).T
        label = np.array(data_origin['Label_testing']).T
        data = {
            'data_unlabelled': data,
            'label': label
        }
    return data


def train(data):
    
    class_kernels = []
    print("Training......")
    for i in range(len(data)):
        m = np.mean(data[i], axis=0)
        s = np.cov(data[i].T)
        class_kernel = {
            'm': m,
            's': s
        }
        class_kernels.append(class_kernel)
    
    print("Training done......")
    with open('.//runs//logs//log.txt', mode='a') as f:
        print("### Training Results ###", file = f)
        print("In .//training_results.json", file = f)
        print("[{}]".format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(now()))), file = f)
        print("", file = f)
        f.close()
    with open('.//runs//logs//training_results_m&s.json', mode='a') as f:
        print(class_kernels, file = f)
        f.close()
    
    return class_kernels


def test(model_weights, testing_data):
    
    data = testing_data['data_unlabelled']
    labell = testing_data['label']
    mtl = np.matmul
    scores = []
    scores_euc = []
    pre_results = []
    val_results = []
    pre_results_euc = []
    val_results_euc = []
    
    print("Predicting......")
    for i in range(len(data)):
        x = data[i]
        score = {}
        score_euc = {}
        for j in range(len(model_weights)):
            m = model_weights[j]['m']
            s = model_weights[j]['s']
            x_normed = x - m
            D_mah = np.sqrt(mtl(mtl(x_normed.reshape(1,x_normed.shape[0]), np.linalg.inv(s)), x_normed.reshape(x_normed.shape[0],1)))
            D_euc = np.sqrt(mtl(x_normed.reshape(1,x_normed.shape[0]), x_normed.reshape(x_normed.shape[0],1)))
            score.update({str(j): D_mah})
            score_euc.update({str(j): D_euc})
        #print(score)
        scores.append(score)
        scores_euc.append(score_euc)
        pre_results.append(int(min(score,key=score.get))+1)
        pre_results_euc.append(int(min(score_euc,key=score_euc.get))+1)
        
        if (i+1)%50 == 0:
            print("{} instances have been predicted...".format(i+1))
            with open('.//runs//logs//log.txt', mode='a') as f:
                print("{} instances have been predicted...".format(i+1), file = f)
                print("[{}]".format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(now()))), file = f)
                print("", file = f)
                f.close()
    
    with open('.//runs//logs//log.txt', mode='a') as f:
        print("Prediction done......", file = f)
        print("### Testing Prediction Results ###", file = f)
        print("In .//prediction_results.json", file = f)
        print("[{}]".format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(now()))), file = f)
        print("", file = f)
        f.close()
    with open('.//runs//logs//prediction_results_mah.json', mode='a') as f:
        print(pre_results, file = f)
        f.close()
    with open('.//runs//logs//prediction_results_euc.json', mode='a') as f:
        print(pre_results_euc, file = f)
        f.close()
    print("Prediction done......")

    num_right = 0
    num_error = 0
    num_right_euc = 0
    num_error_euc = 0
    for i in range(len(pre_results)):
        if pre_results[i] == labell[i]:
            val_results.append('right')
            num_right += 1
        else:
            val_results.append('error')
            num_error += 1
    for i in range(len(pre_results_euc)):
        if pre_results_euc[i] == labell[i]:
            val_results_euc.append('right')
            num_right_euc += 1
        else:
            val_results_euc.append('error')
            num_error_euc += 1
    
    with open('.//runs//logs//log.txt', mode='a') as f:
        print("### Testing Validation Results ###", file = f)
        print("In .//val_results.json", file = f)
        print("", file = f)
        print("### Accuracy Evaluation ###")
        print("Accuracy via MAH: {:.2%}".format(num_right/(num_right+num_error)))
        print("Accuracy via MAH: {:.2%}".format(num_right/(num_right+num_error)), file = f)
        print("Accuracy via EUC: {:.2%}".format(num_right_euc/(num_right_euc+num_error_euc)))
        print("Accuracy via EUC: {:.2%}".format(num_right_euc/(num_right_euc+num_error_euc)), file = f)
        print("[{}]".format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(now()))), file = f)
        print("", file = f)
        f.close()
    with open('.//runs//logs//val_results_mah.json', mode='a') as f:
        print(val_results, file = f)
        f.close()
    with open('.//runs//logs//val_results_euc.json', mode='a') as f:
        print(val_results_euc, file = f)
        f.close()
    print("Everything done......")
    
    return





if __name__ == '__main__':
    
    data_dir = './/data//Mult-Class Problem.mat'
    os.makedirs('.//runs//logs') 

    start_time = now()
    with open('.//runs//logs//log.txt', mode='a') as f:
        print("--------Minimum Distance Classifier v1.0--------")
        print("### Solution for: {} ###".format(data_dir[9:]))
        print("---------------------START---------------------")
        print("--------Minimum Distance Classifier v1.0--------", file = f)
        print("### Solution for: {} ###".format(data_dir[9:]), file = f)
        print("---------------------START---------------------", file = f)
        print("[{}]".format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(now()))), file = f)
        print("", file = f)
        f.close()
    
    training_data = training_data_loader(data_dir)
    testing_data = testing_data_loader(data_dir)

    model_weights_trained = train(training_data)

    test(model_weights_trained, testing_data)

    with open('.//runs//logs//log.txt', mode='a') as f:
        print("Total time: {:.2f}s".format(now()-start_time))
        print("Total time: {:.2f}s".format(now()-start_time), file = f)
        print("----------------------END----------------------")
        print("----------------------END----------------------", file = f)
        f.close()
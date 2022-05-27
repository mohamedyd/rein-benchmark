import numpy as np

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.linear_model import LogisticRegression
import torch

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from scipy.optimize import minimize

from picket.transformer.utils import *
from picket.rvae.run import get_outlier_scores
from picket.transformer.PicketNet import *
from picket.wrappers.pytorchNNThreeClass import torchNNThreeClass
from picket.wrappers.pytorchNN import torchNN
from picket.prepare.dataInfo import *

import matplotlib.pyplot as plt

from sklearn.svm import OneClassSVM
from scipy.special import expit

import copy
import os
import time
import json

from joblib import dump, load

np.random.seed(123)

def dataPack(ds):
    X_dirty_correct = ds['X_dirty_correct']
    idx_dirty_correct = ds['idx_dirty_correct']
    X_dirty_wrong = ds['X_dirty_wrong']
    idx_dirty_wrong = ds['idx_dirty_wrong']
    X_clean_correct = ds['X_clean_correct']
    idx_clean_correct = ds['idx_clean_correct']
    
    X = np.concatenate((X_clean_correct, X_dirty_correct), axis=0)
    idx = np.concatenate((idx_clean_correct, idx_dirty_correct), axis=0)
    
    X = np.concatenate((X, X_dirty_wrong), axis=0)
    idx = np.concatenate((idx, idx_dirty_wrong), axis=0)
    
    label0 = np.zeros(X_clean_correct.shape[0]+X_dirty_correct.shape[0])
    label1 = np.ones(X_dirty_wrong.shape[0])
    label = np.concatenate((label0, label1), axis=0)
    
    return X, idx, label

def dataPackAdv(ds_adv, ds_random):
    X_adv = ds_adv['X_attack']
    adv_size = X_adv.shape[0]
    X_clean_correct = ds_random['X_clean_correct']
    idx_clean_correct = ds_random['idx_clean_correct']
    attr_num = X_clean_correct.shape[1]
    data_dim = X_clean_correct.shape[2]

    if np.abs(np.sum(X_adv[:, :, 1:])) > 1e-6:
        print('The data set is not valid.')

    indices = np.random.permutation(X_clean_correct.shape[0])
    X_clean_correct_subset = X_clean_correct[indices[: adv_size]]
    idx_clean_correct_subset = idx_clean_correct[indices[: adv_size]]

    X = np.concatenate((X_clean_correct_subset, X_adv), axis=0)
    idx = np.concatenate((idx_clean_correct_subset, idx_clean_correct_subset), axis=0)

    label0 = np.zeros(adv_size)
    label1 = np.ones(adv_size)
    label = np.concatenate((label0, label1), axis=0)

    return X, idx, label

def randomSelect(X, size):
    indices = np.random.permutation(X.shape[0])
    return X[indices[:size]]

def randomSelectWithIdx(X, idx, size):
    indices = np.random.permutation(X.shape[0])
    return X[indices[:size]], idx[indices[:size]]

def getArtificialSize(ds_artificial_group):
    sizes = []
    for ds in ds_artificial_group:
        if 'X_attack' in ds.files:
            ds_size = ds['X_attack'].shape[0]
        else:
            ds_size = ds['X_dirty_wrong'].shape[0]
        sizes.append(ds_size)
    return sizes

def dataPackArtificialTrain(ds_train, ds_artificial_group, fromFile=True):
    #min_size = min(getArtificialSize(ds_artificial_group))
    #min_size = min((min_size, ds_train['X_clean_correct'].shape[0]))

    X_mix_group = []
    label_group = []
    idx_mix_group = []

    for ds in ds_artificial_group:
        if fromFile:
            keys = ds.files
        else:
            keys = ds.keys() 
        if 'X_attack' in keys:
            X_attack = ds['X_attack']
            X_fail = ds['X_fail']
            idx_fail = np.zeros((X_fail.shape[0], X_fail.shape[1]))
            X_safe, X_victim = getSafeAndVictimAdv(ds_train['X_clean_correct'], X_attack, X_fail)
            idx_safe = np.zeros((X_safe.shape[0], X_safe.shape[1]))
            idx_victim = np.zeros((X_victim.shape[0], X_victim.shape[1]))
        else:
            X_victim = ds['X_dirty_wrong']
            idx_victim =ds['idx_dirty_wrong']
            X_safe = np.concatenate((ds['X_dirty_correct'], ds['X_clean_correct']), axis=0)
            idx_safe = np.concatenate((ds['idx_dirty_correct'], ds['idx_clean_correct']), axis=0)

        X_mix_tmp, idx_mix_tmp, label_tmp = mixCleanDirty(X_safe, idx_safe, X_victim, idx_victim)
        X_mix_group.append(X_mix_tmp)
        idx_mix_group.append(idx_mix_tmp)
        label_group.append(label_tmp)
    X_mix = np.concatenate(X_mix_group, axis=0)
    idx_mix = np.concatenate(idx_mix_group, axis=0)
    label = np.concatenate(label_group, axis=0)

    return X_mix, idx_mix, label

def getSafeAndVictimAdv(X_clean_correct, X_attack, X_fail):
    X_safe = np.concatenate((X_clean_correct, X_fail[np.random.randint(X_fail.shape[0], 
        size=X_clean_correct.shape[0])]), axis=0)
    X_victim = X_attack[np.random.randint(X_attack.shape[0], size=X_clean_correct.shape[0]*2)]
    return X_safe, X_victim

def mixCleanDirty(X_safe, idx_safe, X_victim, idx_victim):
    label0 = np.zeros(X_safe.shape[0])
    label1 = np.ones(X_victim.shape[0])
    label = np.concatenate((label0, label1), axis=0)
    return np.concatenate((X_safe, X_victim), axis=0), np.concatenate((idx_safe, idx_victim), axis=0), label

def dataPackMix(ds_adv, ds_random, ds_fgm):
    X_adv = ds_adv['X_attack']
    adv_size = X_adv.shape[0]
    X_fgm = ds_fgm['X_attack']
    fgm_size = X_fgm.shape[0]

    X_clean_correct = ds_random['X_clean_correct']
    idx_clean_correct = ds_random['idx_clean_correct']
    attr_num = X_clean_correct.shape[1]
    data_dim = X_clean_correct.shape[2]
    clean_size = X_clean_correct.shape[0]

    X_dirty_wrong = ds_random['X_dirty_wrong']
    idx_dirty_wrong = ds_random['idx_dirty_wrong']
    dirty_size = X_dirty_wrong.shape[0]

    if np.abs(np.sum(X_adv[:, :, 1:])) > 1e-6:
        print('The data set is not valid.')

    sizes = [fgm_size, clean_size//3, dirty_size, adv_size]
    final_size = min(sizes)

    X_clean_correct_subset = randomSelect(X_clean_correct, final_size*3)
    X_adv_subset = randomSelect(X_adv, final_size)
    X_dirty_wrong_subset = randomSelect(X_dirty_wrong, final_size)
    X_fgm_subset = randomSelect(X_fgm, final_size)

    X = np.concatenate((X_clean_correct_subset, X_adv_subset), axis=0)
    X = np.concatenate((X, X_dirty_wrong_subset), axis=0)
    X = np.concatenate((X, X_fgm_subset), axis=0)

    idx = np.zeros((final_size*6, attr_num))

    label0 = np.zeros(final_size*3)
    label1 = np.ones(final_size*3)
    label = np.concatenate((label0, label1), axis=0)

    return X, idx, label
    
def flattern(X, first_dim=False):
    if first_dim:
        return X[:, :, 0]
    else:
        return X.reshape(X.shape[0], -1)

def evaluation_LR(X_train, y_train, X_test, y_test, model='lr'):
    if model == 'lr':
        clf = LogisticRegression(random_state=0).fit(flattern(X_train), y_train)
    if model == 'svm':
        clf = LinearSVC().fit(flattern(X_train), y_train)
    if model == 'nn':
        clf = torchNN(input_size=flattern(X_train).shape[-1])
        clf.fit(flattern(X_train), y_train)
    y_pred = clf.predict(flattern(X_test))
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return acc, f1

def evaluation_LR_class(X_train, y_train, X_test, y_test, class_train, class_test, model='lr', folder=None, methodname=None, random_id=0):
    if class_train.max() != class_train.min():
        X_train0 = X_train[class_train==class_train.min()]
        X_train1 = X_train[class_train==class_train.max()]

        X_test0 = X_test[class_test==class_train.min()]
        X_test1 = X_test[class_test==class_train.max()]

        y_train0 = y_train[class_train==class_train.min()]
        y_train1 = y_train[class_train==class_train.max()]

        y_test0 = y_test[class_test==class_train.min()]
        y_test1 = y_test[class_test==class_train.max()]

        if model == 'lr':
            clf0 = LogisticRegression(random_state=0).fit(flattern(X_train0), y_train0)
            clf1 = LogisticRegression(random_state=0).fit(flattern(X_train1), y_train1)
            if folder is not None:
                dump(clf0, os.path.join(folder, 'lr_detector_zero_%s_%d.joblib' % (methodname, random_id)))
                dump(clf1, os.path.join(folder, 'lr_detector_one_%s_%d.joblib' % (methodname, random_id)))

        if model == 'svm':
            clf0 = LinearSVC().fit(flattern(X_train0), y_train0)
            clf1 = LinearSVC().fit(flattern(X_train1), y_train1)

        if model == 'nn':
            clf0 = torchNN(input_size=flattern(X_train0).shape[-1])
            clf1 = torchNN(input_size=flattern(X_train0).shape[-1])
            clf0.fit(flattern(X_train0), y_train0)
            clf1.fit(flattern(X_train1), y_train1)

        if X_test0.size != 0:
            y_pred0 = clf0.predict(flattern(X_test0))
        
        if X_test1.size != 0:
            y_pred1 = clf1.predict(flattern(X_test1))

        if X_test0.size == 0:
            y_test01 = y_test1
            y_pred01 = y_pred1
        elif X_test1.size == 0:
            y_test01 = y_test0
            y_pred01 = y_pred0
        else:
            y_test01 = np.concatenate((y_test0, y_test1))
            y_pred01 = np.concatenate((y_pred0, y_pred1))
    else:
        class_exist = class_train.max()
        X_train_exist = X_train[class_train==class_exist]
        y_train_exist = y_train[class_train==class_exist]
        if model == 'lr':
            clf_exist = LogisticRegression(random_state=0).fit(flattern(X_train_exist), y_train_exist)
            if folder is not None:
                dump(clf_exist, os.path.join(folder, 'lr_detector_exist_%s_%d.joblib' % (methodname, random_id)))
        if model == 'svm':
            clf_exist = LinearSVC().fit(flattern(X_train_exist), y_train_exist)
        if model == 'nn':
            clf_exist = torchNN(input_size=flattern(X_train_exist).shape[-1])
            clf_exist.fit(flattern(X_train_exist), y_train_exist)

        X_test_exist = X_test[class_test==class_exist]
        y_test_exist = y_test[class_test==class_exist]
        X_test_nonexist = X_test[class_test!=class_exist]
        y_test_nonexist = y_test[class_test!=class_exist]

        y_pred_exist = clf_exist.predict(flattern(X_test_exist))
        if X_test_nonexist.size != 0:
            y_pred_nonexist = np.zeros(X_test_nonexist.shape[0])
            y_test01 = np.concatenate((y_test_exist, y_test_nonexist))
            y_pred01 = np.concatenate((y_pred_exist, y_pred_nonexist))
        else:
            y_test01 = y_test_exist
            y_pred01 = y_pred_exist

    acc = accuracy_score(y_test01, y_pred01)
    f1 = f1_score(y_test01, y_pred01)
    return acc, f1

def evaluation_LR_class_no_train(X_test, y_test, class_train, class_test,  model='lr', folder=None, methodname=None, random_id=0):
    if class_train.max() != class_train.min():
        X_test0 = X_test[class_test==class_train.min()]
        X_test1 = X_test[class_test==class_train.max()]

        y_test0 = y_test[class_test==class_train.min()]
        y_test1 = y_test[class_test==class_train.max()]


        clf0 = load(os.path.join(folder, 'lr_detector_zero_%s_%d.joblib' % (methodname, random_id)))
        clf1 = load(os.path.join(folder, 'lr_detector_one_%s_%d.joblib' % (methodname, random_id)))

        if X_test0.size != 0:
            y_pred0 = clf0.predict(flattern(X_test0))
        
        if X_test1.size != 0:
            y_pred1 = clf1.predict(flattern(X_test1))

        if X_test0.size == 0:
            y_test01 = y_test1
            y_pred01 = y_pred1
        elif X_test1.size == 0:
            y_test01 = y_test0
            y_pred01 = y_pred0
        else:
            y_test01 = np.concatenate((y_test0, y_test1))
            y_pred01 = np.concatenate((y_pred0, y_pred1))
    else:
        class_exist = class_train.max()
        clf_exist = load(os.path.join(folder, 'lr_detector_exist_%s_%d.joblib' % (methodname, random_id)))

        X_test_exist = X_test[class_test==class_exist]
        y_test_exist = y_test[class_test==class_exist]
        X_test_nonexist = X_test[class_test!=class_exist]
        y_test_nonexist = y_test[class_test!=class_exist]

        y_pred_exist = clf_exist.predict(flattern(X_test_exist))
        if X_test_nonexist.size != 0:
            y_pred_nonexist = np.zeros(X_test_nonexist.shape[0])
            y_test01 = np.concatenate((y_test_exist, y_test_nonexist))
            y_pred01 = np.concatenate((y_pred_exist, y_pred_nonexist))
        else:
            y_test01 = y_test_exist
            y_pred01 = y_pred_exist

    acc = accuracy_score(y_test01, y_pred01)
    f1 = f1_score(y_test01, y_pred01)
    return acc, f1

def evaluation_LR_class_time_analysis(X_train, y_train, X_test, y_test, class_train, class_test, model='lr'):
    if class_train.max() != class_train.min():
        X_train0 = X_train[class_train==class_train.min()]
        X_train1 = X_train[class_train==class_train.max()]

        X_test0 = X_test[class_test==class_train.min()]
        X_test1 = X_test[class_test==class_train.max()]

        y_train0 = y_train[class_train==class_train.min()]
        y_train1 = y_train[class_train==class_train.max()]

        y_test0 = y_test[class_test==class_train.min()]
        y_test1 = y_test[class_test==class_train.max()]

        if model == 'lr':
            clf0 = LogisticRegression(random_state=0).fit(flattern(X_train0), y_train0)
            clf1 = LogisticRegression(random_state=0).fit(flattern(X_train1), y_train1)
        if model == 'svm':
            clf0 = LinearSVC().fit(flattern(X_train0), y_train0)
            clf1 = LinearSVC().fit(flattern(X_train1), y_train1)
        if model == 'nn':
            clf0 = torchNN(input_size=flattern(X_train0).shape[-1])
            clf1 = torchNN(input_size=flattern(X_train0).shape[-1])
            clf0.fit(flattern(X_train0), y_train0)
            clf1.fit(flattern(X_train1), y_train1)

        start = time.time()
        if X_test0.size != 0:
            y_pred0 = clf0.predict(flattern(X_test0))
        
        if X_test1.size != 0:
            y_pred1 = clf1.predict(flattern(X_test1))
        elapsed = time.time() - start

        if X_test0.size == 0:
            y_test01 = y_test1
            y_pred01 = y_pred1
        elif X_test1.size == 0:
            y_test01 = y_test0
            y_pred01 = y_pred0
        else:
            y_test01 = np.concatenate((y_test0, y_test1))
            y_pred01 = np.concatenate((y_pred0, y_pred1))
    else:
        class_exist = class_train.max()
        X_train_exist = X_train[class_train==class_exist]
        y_train_exist = y_train[class_train==class_exist]
        if model == 'lr':
            clf_exist = LogisticRegression(random_state=0).fit(flattern(X_train_exist), y_train_exist)
        if model == 'svm':
            clf_exist = LinearSVC().fit(flattern(X_train_exist), y_train_exist)
        if model == 'nn':
            clf_exist = torchNN(input_size=flattern(X_train_exist).shape[-1])
            clf_exist.fit(flattern(X_train_exist), y_train_exist)

        X_test_exist = X_test[class_test==class_exist]
        y_test_exist = y_test[class_test==class_exist]
        X_test_nonexist = X_test[class_test!=class_exist]
        y_test_nonexist = y_test[class_test!=class_exist]

        start = time.time()
        y_pred_exist = clf_exist.predict(flattern(X_test_exist))
        elapsed = time.time() - start

        if X_test_nonexist.size != 0:
            y_pred_nonexist = np.zeros(X_test_nonexist.shape[0])
            y_test01 = np.concatenate((y_test_exist, y_test_nonexist))
            y_pred01 = np.concatenate((y_pred_exist, y_pred_nonexist))
        else:
            y_test01 = y_test_exist
            y_pred01 = y_pred_exist

    acc = accuracy_score(y_test01, y_pred01)
    f1 = f1_score(y_test01, y_pred01)
    return elapsed

def acc_for_thres(outlier_scores, thres, labels):
    correct = ((outlier_scores > thres) == labels)
    num_correct = np.sum(correct)
    acc = float(num_correct)/labels.shape[0]
    return acc

def f1_for_thres(outlier_scores, thres, labels):
    pred = outlier_scores > thres
    f1 = f1_score(labels, pred)
    return f1

def find_optimal_thres_acc(outlier_scores, labels):
    if outlier_scores.shape[0] < 1000:
        max_acc = -1
        max_id = None
        for i in range(outlier_scores.shape[0]):
            acc = acc_for_thres(outlier_scores, outlier_scores[i], labels)
            if acc > max_acc:
                max_acc = acc
                max_id = i
        return outlier_scores[max_id]
    else:
        permutation = np.random.permutation(outlier_scores.shape[0])
        max_acc = -1
        max_id = None

        for i in range(1000):
            acc = acc_for_thres(outlier_scores, outlier_scores[permutation[i]], labels)
            if acc > max_acc:
                max_acc = acc
                max_id = permutation[i]
        return outlier_scores[max_id]


def find_optimal_thres_f1(outlier_scores, labels):
    if outlier_scores.shape[0] < 1000:
        max_f1 = -1
        max_id = None
        for i in range(outlier_scores.shape[0]):
            f1 = f1_for_thres(outlier_scores, outlier_scores[i], labels)
            if f1 > max_f1:
                max_f1 = f1
                max_id = i
        return outlier_scores[max_id]
    else:
        permutation = np.random.permutation(outlier_scores.shape[0])
        max_f1 = -1
        max_id = None

        for i in range(1000):
            f1 = f1_for_thres(outlier_scores, outlier_scores[permutation[i]], labels)
            if f1 > max_f1:
                max_f1 = f1
                max_id = permutation[i]
        return outlier_scores[max_id]        


def NLL_SVM(T, y, Z):
    L = 1./(1+np.exp(-y*Z/T[0]))
    NLL_all = -np.log(L)
    return np.sum(NLL_all)

def NLL(T, y, Z):
    Z = np.choose(y, Z.T)
    L = expit(Z/T[0])
    NLL_all = -np.log(L)
    return np.sum(NLL_all)

def getTemperature(X, y, clf, folder_num=5, modeltype='svm'):
    num_tuples = X.shape[0]
    indices = np.random.permutation(num_tuples)
    
    T_product = 1
    
    for cv in range(folder_num):
        v_idx = np.arange(int(cv*num_tuples/5), int((cv+1)*num_tuples/5))
        t_idx = np.delete(np.arange(num_tuples), v_idx)  
        clf.fit(X[indices[t_idx]], y[indices[t_idx]])
        
        y_pred = clf.predict(X[indices[v_idx]])
        raw_scores = clf.decision_function(X[indices[v_idx]])
        
        T0 = [1]
        if modeltype == 'svm':
            T = minimize(NLL_SVM, T0, args=(y[indices[v_idx]], raw_scores), bounds=[(0.01, None)]).x[0]
        if modeltype == 'nn':
            T = minimize(NLL, T0, args=(y[indices[v_idx]], raw_scores), bounds=[(0.01, None)]).x[0]
        if modeltype == 'lr':
            assert is_zero_one(y), "Should be zero one for lr!"
            T = minimize(NLL_SVM, T0, args=(2*y[indices[v_idx]]-1, raw_scores), bounds=[(0.01, None)]).x[0]

        T_product = T_product*T
    
    T_mean = T_product**(1.0/folder_num)
    return T_mean

def is_zero_one(y):
    return (np.sum(y >= 0) == y.shape[0])

def getCalibratedScore(X_test, T, clf, modeltype):
    y_pred = clf.predict(X_test)
    raw_scores = clf.decision_function(X_test)
    if modeltype == 'svm':
        cali_scores = expit(y_pred*raw_scores/T)
    if modeltype == 'nn':
        cali_scores = expit(np.choose(y_pred, raw_scores.T)/T)
    if modeltype == 'lr':
        assert is_zero_one(y_pred), "Should be zero one for lr!"
        cali_scores = expit((2*y_pred-1)*raw_scores/T) 
    return cali_scores

def distanceScore(X_train, sample, dtypes, alpha):
    cos_sim = np.einsum('ijk,jk->ij', X_train, sample)
    cos_sim = cos_sim/np.linalg.norm(X_train, axis=-1)
    cos_sim = cos_sim/np.linalg.norm(sample, axis=-1)
    numeric_distance = X_train-np.stack([sample]*X_train.shape[0], axis=0)
    numeric_distance = np.abs(numeric_distance)[:, :, 0]
    distance = np.zeros(X_train.shape[0])
    for i, attr_type in enumerate(dtypes):
        if attr_type == 'numeric':
            distance_tmp = numeric_distance[:, i]
            thres = np.std(X_train[:, i, 0])*alpha
            distance_tmp[distance_tmp > thres] = thres
            distance_tmp = distance_tmp/thres
            distance += distance_tmp
    
        if attr_type == 'categorical':
            distance_tmp = cos_sim[:, i]
            distance += distance_tmp
        
        if attr_type == 'text':
            distance_tmp = (1-cos_sim[:, i])
            distance += distance_tmp
        
    return distance

def knnScore(X_train, y_train, X_test, k, dtypes, alpha):
    is_all_num = (dtypes.count('numeric') == len(dtypes))
    clf = LinearSVC()
    clf.fit(flattern(X_train, is_all_num), y_train)
    y_pred = clf.predict(flattern(X_test, is_all_num))
    
    res = []
    print_granularity = X_test.shape[0] // 20

    for i in range(X_test.shape[0]):
        if i % print_granularity == 0:
            print("%d / %d" % (i, X_test.shape[0]))
        distance = distanceScore(X_train, X_test[i], dtypes, alpha)
        knn_idx = np.flip(np.argsort(distance))[:k]
        knn_y = y_train[knn_idx]
        res.append(np.sum(knn_y != y_pred[i])/float(k))
    return np.array(res)

class RVAE_ds:
    def __init__(self, X_raw, idx, vecs, dtypes, means=None, stds=None, dataset_type='mixed'):
        self.cat_cols = []
        self.num_cols = []
        self.feat_info = []
        self.dataset_type = dataset_type
        self.X = []
        self.means_ = []
        self.stds_ = []
        
        for i in range(X_raw.shape[1]):
            if dtypes[i] in ['categorical', 'text']:
                self.X.append(idx[:, i])
                self.cat_cols.append(str(i))
                self.feat_info.append((str(i), 'categ', vecs[i].shape[0]))
                self.means_.append(0)
                self.stds_.append(0)
            elif dtypes[i] == 'numeric':
                feature_tmp = X_raw[:, i, 0]
                if means is None:
                    mean = np.mean(feature_tmp)
                    std = np.std(feature_tmp)
                else:
                    mean = means[i]
                    std = stds[i]
                #Don't need this since the data have been standardized
                #feature_tmp = (feature_tmp-mean)/std
                self.means_.append(mean)
                self.stds_.append(std)
                
                self.X.append(feature_tmp)
                self.num_cols.append(str(i))
                self.feat_info.append((str(i), 'real', 1))
                
            else:
                print('Something is wrong.')
        self.X = np.stack(self.X, axis=-1)
        
        self.X = torch.Tensor(self.X)
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        ret_list = [torch.Tensor(self.X[idx])]
        # Just a placeholder
        ret_list += [torch.Tensor([0, 0, 0])]
        return ret_list

class Metadata:
    def __init__(self):
        self.AVI=False
        self.activation='relu'
        self.alpha_prior=0.95
        self.batch_size=150 
        self.cuda_on=torch.cuda.is_available() 
        self.embedding_size=50
        self.inference_type='vae' 
        self.is_one_hot=False 
        self.l2_reg=0.0 
        self.latent_dim=20
        self.layer_size=400
        self.log_interval=50
        self.lr=0.001
        self.number_epochs=100
        self.outlier_model='RVAE'
        self.seqvae_bprop=False
        self.seqvae_steps=4
        self.seqvae_two_stage=False 
        self.std_gauss_nll=2.0
        self.steps_2stage=4

class Attribute:
    def __init__(self, vec):
        self.vec = vec

def add_random_noise(x, idx, eps, vec, dtypes, stds, num_std=5, useAbs=False):
    x_res = np.copy(x)
    idx_res = np.copy(idx)
    for i in range(x.shape[0]):
        if np.random.rand() < eps:
            if dtypes[i] != 'numeric':
                noise = np.random.normal(0, num_std, size=x_res[i].shape[-1])
                if dtypes[i] == 'categorical':
                    useful_dim = vec[i].shape[0]
                    noise[useful_dim:] = 0
                x_res[i, :] = x_res[i, :] + noise 
            else:
                noise = np.random.normal(0, num_std)
                if useAbs:
                    x_res[i, 0] = x_res[i, 0] + np.abs(noise)
                else:
                    x_res[i, 0] = x_res[i, 0] + noise
    return x_res, idx_res

def add_random_noise_ds(X_input, idx, eps, vec, dtypes, stds, num_std=5, useAbs=False):
    X = np.copy(X_input)
    for i in range(X.shape[0]):
        x_flipped, _ = add_random_noise(X[i], idx[i], eps, vec, dtypes, stds, num_std, useAbs)
        X[i] = x_flipped
    return X

def add_random_noise_odd(X_input, vec, dtypes, ntype, magnitude):
    X = np.copy(X_input)
    if ntype == 'normal':
        noise = np.random.normal(0, np.abs(magnitude), X_input.shape)
    if ntype == 'uniform':
        noise = np.random.rand(*X_input.shape)*magnitude
    if ntype == 'bernoulli':
        noise = np.random.randint(0, 2, X_input.shape)*magnitude

    for i in range(X_input.shape[1]):
        if dtypes[i] == 'categorical':
            useful_dim = vec[i].shape[0]
            noise[:, i, useful_dim:] = 0

    return (X + noise).astype(np.float32)

def get_log_odds_diff(X, idx, eps, vec, dtypes, stds, sample_count, num_std, clf, useAbs, modeltype):
    is_all_num = (dtypes.count('numeric') == len(dtypes))
    sample_log_odds = []
    for i in range(sample_count):
        X_noise = add_random_noise_ds(X, idx, eps, vec, dtypes, stds, num_std, useAbs)
        y_pred = clf.predict(flattern(X_noise, is_all_num))
        logits = clf.decision_function(flattern(X_noise, is_all_num))
        if modeltype == 'svm':
            sample_log_odds.append(-logits*y_pred*2)
        if modeltype == 'nn':
            sample_log_odds.append(np.choose(1-y_pred, logits.T) - np.choose(y_pred, logits.T))
        if modeltype == 'lr':
            y_pred = 2*y_pred - 1
            sample_log_odds.append(-logits*y_pred*2)

    expected_log_odds = np.mean(np.stack(sample_log_odds, axis=0), axis=0)

    y_pred = clf.predict(flattern(X, is_all_num))
    logits = clf.decision_function(flattern(X, is_all_num))
    if modeltype == 'svm':
        origin_log_odds = -logits*y_pred*2
    if modeltype == 'nn':
        origin_log_odds = np.choose(1-y_pred, logits.T) - np.choose(y_pred, logits.T)
    if modeltype == 'lr':
        y_pred = 2*y_pred - 1
        origin_log_odds = -logits*y_pred*2
    
    return (expected_log_odds-origin_log_odds)

def get_log_odds_diff_new(X, vec, dtypes, sample_count, clf, ntype, magnitude, modeltype):
    is_all_num = (dtypes.count('numeric') == len(dtypes))
    sample_log_odds = []
    for i in range(sample_count):
        X_noise = add_random_noise_odd(X, vec, dtypes, ntype, magnitude)
        y_pred = clf.predict(flattern(X_noise, is_all_num))
        logits = clf.decision_function(flattern(X_noise, is_all_num))
        if modeltype == 'svm':
            sample_log_odds.append(-logits*y_pred*2)
        if modeltype == 'nn':
            sample_log_odds.append(np.choose(1-y_pred, logits.T) - np.choose(y_pred, logits.T))
        if modeltype == 'lr':
            assert is_zero_one(y_pred), "Should be zero one for lr!"
            y_pred = 2*y_pred - 1
            sample_log_odds.append(-logits*y_pred*2)

    expected_log_odds = np.mean(np.stack(sample_log_odds, axis=0), axis=0)

    y_pred = clf.predict(flattern(X, is_all_num))
    logits = clf.decision_function(flattern(X, is_all_num))
    if modeltype == 'svm':
        origin_log_odds = -logits*y_pred*2
    if modeltype == 'nn':
        origin_log_odds = np.choose(1-y_pred, logits.T) - np.choose(y_pred, logits.T)
    if modeltype == 'lr':
        assert is_zero_one(y_pred), "Should be zero one for lr!"
        y_pred = 2*y_pred - 1
        origin_log_odds = -logits*y_pred*2
    
    return (expected_log_odds-origin_log_odds)

def getRobustMeanScore(x, X_train):
    X_all = np.concatenate((X_train, x.reshape(1, -1)), axis=0)
    mean = np.mean(X_all, axis=0)
    cov = np.cov(X_all, rowvar=False)

    cov = cov - np.eye(cov.shape[0])

    w, v = np.linalg.eig(cov)
    max_eig_id = np.argmax(np.abs(w))
    v_star = v[:, max_eig_id]

    return np.abs(np.dot(v_star, x-mean))

def getRobustMeanScoreOnebyOne(X_test, X_train):
    scores = []

    for i in range(X_test.shape[0]):
        scores.append(getRobustMeanScore(X_test[i, :], X_train))

    return np.array(scores)

class Filters:
    def __init__(self, name, dtypes, clf, ntype, modelname, random_id, artificial_components, adv_method='pgd',
        level='medium', sample_count=10, computeLoss=True, load=False, save=False):
        self.name = name
        self.modelname=modelname
        self.dtypes = dtypes
        self.is_all_num = (dtypes.count('numeric') == len(dtypes))
        self.ntype = ntype
        self.adv_method = adv_method
        self.DATA_DIR = '../data/' + name + '/'
        self.clf = clf
        self.random_id = random_id
        self.level = level
        self.artificial_components = artificial_components
        self.load = load
        self.save = save

        self.LoadDataTrainClean()
        self.LoadDataMix()
        if self.save:
            self.SaveDataMix()

        if computeLoss:
            self.ComputePKLoss(sample_count=sample_count)

        if modelname in ['svm', 'lr']:
            self.clf.fit(flattern(self.X_train, self.is_all_num), self.y_train)

        self.res = {}
    
    def LoadDataTrainClean(self):
        ds = np.load(self.DATA_DIR + '%s_train_test_%s.npz' % (self.name, self.random_id), 
                    allow_pickle=True, encoding="latin1")
        self.X_train = ds['X_train']
        if self.modelname == 'svm':
            self.y_train = 2*ds['y_train'] - 1
        else:
            self.y_train = ds['y_train']
        self.vec = ds['vec']
        self.idx_train = ds['idx_train']
        self.attribute_info = [Attribute(self.vec[i]) for i in range(len(self.vec))]
        self.data_dim = self.X_train.shape[-1]

    def ComputePKLoss(self, sample_count=1):
        if self.name == 'HTRU2':
            hidden_dim = 16
            transformer_layer = 1
        else:
            hidden_dim = 64
            transformer_layer = 6

        param = {
            'description': 'Contrastive Transformer',
            'model_dim': self.data_dim,
            'input_dim': self.data_dim,
            'attribute_num': len(self.attribute_info),
            'transformer_layer': transformer_layer,
            'head_num': 2,
            'hidden_dim': hidden_dim,
            'dropout': 0.1,
            'numerical_ids': [i for i, x in enumerate(self.dtypes) if x == "numeric"],
            'batch_size': 1000,
            'epochs': 100,
            'opt_factor': 0.1,
            'warmup': 300,
            'adam_lr': 0,
            'adam_betas': (0.9, 0.98),
            'adam_eps': 1e-9,
            'neg_sample_num': 4,
            'random_mask': False,
            'fast': True,
            'structure_mask_type': 'none', # could be 'hard', 'soft', 'sample'
            'holdout_ratio': 0.2,
            'useEncoding': True,
            'categorical_ids': [i for i, x in enumerate(self.dtypes) if x == 'categorical'],
        }

        PicketN = PicketNetModel(param)
        PicketN.loadData(None, None, self.attribute_info)

        # Should change according to random_id after debug
        PicketN.loadModel(self.DATA_DIR+'%s_PKModel_%d.pt'%(self.name, self.random_id))

        if not self.load:
            loss_train = None

            for _ in range(sample_count):
                loss1, loss2 = PicketN.getLossTest(torch.Tensor(self.X_train_mix).double(), torch.Tensor(self.idx_train_mix))
                if loss_train is None:
                    loss_train = (loss1 + loss2)
                else:
                    loss_train += (loss1 + loss2)

            loss_train = loss_train/sample_count
            self.PKloss_train = loss_train.astype(np.float32)
        ##

        loss_test = None

        for _ in range(sample_count):
            loss1, loss2 = PicketN.getLossTest(torch.Tensor(self.X_test_mix).double(), torch.Tensor(self.idx_test_mix))
            if loss_test is None:
                loss_test = (loss1 + loss2)
            else:
                loss_test += (loss1 + loss2)

        loss_test = loss_test/sample_count

        self.PKloss_test = loss_test.astype(np.float32)

    def runTimeAnalysis(self, sample_count=1, time_measures=10):
        if self.name == 'HTRU2':
            hidden_dim = 16
            transformer_layer = 1
        else:
            hidden_dim = 64
            transformer_layer = 6

        param = {
            'description': 'Contrastive Transformer',
            'model_dim': self.data_dim,
            'input_dim': self.data_dim,
            'attribute_num': len(self.attribute_info),
            'transformer_layer': transformer_layer,
            'head_num': 2,
            'hidden_dim': hidden_dim,
            'dropout': 0.1,
            'numerical_ids': [i for i, x in enumerate(self.dtypes) if x == "numeric"],
            'batch_size': 1000,
            'epochs': 100,
            'opt_factor': 0.1,
            'warmup': 300,
            'adam_lr': 0,
            'adam_betas': (0.9, 0.98),
            'adam_eps': 1e-9,
            'neg_sample_num': 4,
            'random_mask': False,
            'fast': True,
            'structure_mask_type': 'none', # could be 'hard', 'soft', 'sample'
            'holdout_ratio': 0.2,
            'useEncoding': True,
            'categorical_ids': [i for i, x in enumerate(self.dtypes) if x == 'categorical'],
        }

        PicketN = PicketNetModel(param)
        PicketN.loadData(None, None, self.attribute_info)

        # Should change according to random_id after debug
        PicketN.loadModel(self.DATA_DIR+'%s_PKModel_%d.pt'%(self.name, self.random_id))

        loss_train = None

        for _ in range(sample_count):
            loss1, loss2 = PicketN.getLossTest(torch.Tensor(self.X_train_mix).double(), torch.Tensor(self.idx_train_mix))
            if loss_train is None:
                loss_train = (loss1 + loss2)
            else:
                loss_train += (loss1 + loss2)

        loss_train = loss_train/sample_count
        PKloss_train = loss_train.astype(np.float32)
        features_train = np.concatenate((flattern(self.X_train_mix, self.is_all_num), self.PKloss_train), axis=-1)
        class_train = self.clf.predict(flattern(self.X_train_mix, self.is_all_num))

        ##

        PKloss_time_group = []
        prediction_time_group = []
        detector_time_group = []
        for i in range(time_measures):
            start = time.time()
            loss_test = None

            for _ in range(sample_count):
                loss1, loss2 = PicketN.getLossTest(torch.Tensor(self.X_test_mix[:100]).double(), 
                    torch.Tensor(self.idx_test_mix[:100]))
                if loss_test is None:
                    loss_test = (loss1 + loss2)
                else:
                    loss_test += (loss1 + loss2)

            loss_test = loss_test/sample_count        
            PKloss_test = loss_test.astype(np.float32)
            PKloss_time = time.time() - start
            PKloss_time_group.append(PKloss_time)

            features_test = np.concatenate((flattern(self.X_test_mix[:100], self.is_all_num), PKloss_test), axis=-1)

            start = time.time()
            class_test = self.clf.predict(flattern(self.X_test_mix[:100], self.is_all_num))
            prediction_time = time.time() - start
            prediction_time_group.append(prediction_time)

            detector_time = evaluation_LR_class_time_analysis(features_train, self.isbad_train,
                  features_test, self.isbad_test[:100], class_train, class_test)
            detector_time_group.append(detector_time)
        print('PKLoss Time: %f Detector Time: %f Prediction Time: %f' % 
            (np.mean(PKloss_time_group), np.mean(prediction_time_group), np.mean(detector_time_group)))



    def LoadDataMix(self):
        name_tuple = (self.name, self.modelname, 'random', self.level, self.random_id)
        ds_train = np.load(self.DATA_DIR + '%s_%s_%s_flip_train_%s_%d.npz' % name_tuple, 
                        allow_pickle=True, encoding="latin1")

        if self.load:
            mtname = '%s_%s' % (self.modelname, self.ntype=='adv')
            filename = os.path.join(self.DATA_DIR, 'train_data_mix_%s_%d.npz' % (mtname, self.random_id))
            ds_mix = np.load(filename, allow_pickle=True, encoding="latin1")  
            self.X_train_mix = ds_mix['X_train_mix']
            self.idx_train_mix = ds_mix['idx_train_mix']
            self.isbad_train = ds_mix['isbad_train'] 
        else:
            ds_artificial_group = []
            for artificial_comp in self.artificial_components:
                name_tuple = (self.name, self.modelname, artificial_comp[0], artificial_comp[1], self.random_id)
                ds_artificial = np.load(self.DATA_DIR + '%s_%s_%s_flip_train_%s_%d.npz' % name_tuple, 
                            allow_pickle=True, encoding="latin1")
                ds_artificial_group.append(ds_artificial)

                self.X_train_mix, self.idx_train_mix, self.isbad_train = dataPackArtificialTrain(ds_train, ds_artificial_group)

        if self.ntype != 'adv':
            name_tuple = (self.name, self.modelname, self.ntype, self.level, self.random_id)
            ds_test = np.load(self.DATA_DIR + '%s_%s_%s_flip_test_%s_%d.npz' % name_tuple, 
                            allow_pickle=True, encoding="latin1")
            self.X_test_mix, self.idx_test_mix, self.isbad_test = dataPack(ds_test)
        else:
            name_tuple = (self.name, self.modelname, 'random', self.level, self.random_id)
            ds_test = np.load(self.DATA_DIR + '%s_%s_%s_flip_test_%s_%d.npz' % name_tuple, 
                            allow_pickle=True, encoding="latin1")  
            ds_test_adv = np.load(self.DATA_DIR + '%s_%s_%s_flip_test_%s_%d.npz' % (self.name, self.modelname, self.ntype, self.adv_method, self.random_id), 
                                allow_pickle=True, encoding="latin1")
            self.X_test_mix, self.idx_test_mix, self.isbad_test = dataPackAdv(ds_test_adv, ds_test)
    
    def SaveDataMix(self):
        ds = {}
        ds['X_train_mix'] = self.X_train_mix
        ds['idx_train_mix'] = self.idx_train_mix
        ds['isbad_train'] = self.isbad_train

        mtname = '%s_%s' % (self.modelname, self.ntype=='adv')
        filename = os.path.join(self.DATA_DIR, 'train_data_mix_%s_%d.npz' % (mtname, self.random_id))
        np.savez(filename, **ds)


    def RawFeatureFilter(self, usePKLoss=False):
        if usePKLoss:
            mtname = 'RFPK_%s_%s' % (self.modelname, self.ntype=='adv')
        else:
            mtname = 'RF_%s_%s' % (self.modelname, self.ntype=='adv')

        if not self.load:
            if usePKLoss:
                print('Method: Raw Feature w/ PK Loss')
                features_train = np.concatenate((flattern(self.X_train_mix, self.is_all_num), self.PKloss_train), axis=-1)
                features_test = np.concatenate((flattern(self.X_test_mix, self.is_all_num), self.PKloss_test), axis=-1)
            else:
                print('Method: Raw Feature w/o PK Loss')
                features_train = flattern(self.X_train_mix, self.is_all_num)
                features_test = flattern(self.X_test_mix, self.is_all_num)

            class_train = self.clf.predict(flattern(self.X_train_mix, self.is_all_num))
            if self.save:
                np.save(os.path.join(self.DATA_DIR, 'class_train_%s_%d.npy' % (mtname, self.random_id)), class_train)
            class_test = self.clf.predict(flattern(self.X_test_mix, self.is_all_num))

            if self.save:
                acc, f1 = evaluation_LR_class(features_train, self.isbad_train,
                      features_test, self.isbad_test, class_train, class_test, 
                      folder=self.DATA_DIR, methodname=mtname, random_id=self.random_id)
            else:
                acc, f1 = evaluation_LR_class(features_train, self.isbad_train,
                      features_test, self.isbad_test, class_train, class_test)
        else:
            if usePKLoss:
                print('Method: Raw Feature w/ PK Loss')
                features_test = np.concatenate((flattern(self.X_test_mix, self.is_all_num), self.PKloss_test), axis=-1)
            else:
                print('Method: Raw Feature w/o PK Loss')
                features_test = flattern(self.X_test_mix, self.is_all_num)

            class_test = self.clf.predict(flattern(self.X_test_mix, self.is_all_num))
            class_train = np.load(os.path.join(self.DATA_DIR, 'class_train_%s_%d.npy' % (mtname, self.random_id)))

            acc, f1 = evaluation_LR_class_no_train(features_test, self.isbad_test, class_train, class_test,
                  folder=self.DATA_DIR, methodname=mtname, random_id=self.random_id)           

        if usePKLoss:
            self.res['PKRaw_acc'] = acc
            self.res['PKRaw_f1'] = f1
        else:
            self.res['Raw_acc'] = acc
            self.res['Raw_f1'] = f1

    def RawFeatureFilterUnified(self, usePKLoss=False):
        if usePKLoss:
            print('Method: Raw Feature w/ PK Loss')
            features_train = np.concatenate((flattern(self.X_train_mix, self.is_all_num), self.PKloss_train), axis=-1)
            features_test = np.concatenate((flattern(self.X_test_mix, self.is_all_num), self.PKloss_test), axis=-1)
        else:
            print('Method: Raw Feature w/o PK Loss')
            features_train = flattern(self.X_train_mix, self.is_all_num)
            features_test = flattern(self.X_test_mix, self.is_all_num)

        acc, f1 = evaluation_LR(features_train, self.isbad_train,
              features_test, self.isbad_test)

        if usePKLoss:
            self.res['PKRawU_acc'] = acc
            self.res['PKRawU_f1'] = f1
        else:
            self.res['RawU_acc'] = acc
            self.res['RawU_f1'] = f1


    def PKLossFilter(self):
        mtname = 'PK_%s_%s' % (self.modelname, self.ntype=='adv')
        print('Method: PK Loss Only')

        if not self.load:
            features_train = self.PKloss_train
            features_test = self.PKloss_test
            class_train = self.clf.predict(flattern(self.X_train_mix, self.is_all_num))
            if self.save:
                np.save(os.path.join(self.DATA_DIR, 'class_train_%s_%d.npy' % (mtname, self.random_id)), class_train)          
            class_test = self.clf.predict(flattern(self.X_test_mix, self.is_all_num))
            if self.save:       
                acc, f1 = evaluation_LR_class(features_train, self.isbad_train,
                    features_test, self.isbad_test, class_train, class_test,
                    folder=self.DATA_DIR, methodname=mtname, random_id=self.random_id)
            else:
                acc, f1 = evaluation_LR_class(features_train, self.isbad_train,
                    features_test, self.isbad_test, class_train, class_test)                
        else:
            features_test = self.PKloss_test
            class_train = np.load(os.path.join(self.DATA_DIR, 'class_train_%s_%d.npy' % (mtname, self.random_id)))     
            class_test = self.clf.predict(flattern(self.X_test_mix, self.is_all_num))          
            acc, f1 = evaluation_LR_class_no_train(features_test, self.isbad_test, class_train, class_test,
                folder=self.DATA_DIR, methodname=mtname, random_id=self.random_id)
         
        self.res['PK_acc'] = acc
        self.res['PK_f1'] = f1

    def PKLossFilterUnified(self):
        print('Method: PK Loss Only')
        features_train = self.PKloss_train
        features_test = self.PKloss_test

        acc, f1 = evaluation_LR(features_train, self.isbad_train,
              features_test, self.isbad_test)
        self.res['PKU_acc'] = acc
        self.res['PKU_f1'] = f1

    def PKLossFilterThreshold(self):
        print('Method: PK Loss Only')
        features_train = self.PKloss_train
        features_test = self.PKloss_test

        PK_scores_train = features_train/np.median(features_train, axis=0)
        PK_scores_test = features_test/np.median(features_train, axis=0)

        PK_scores_train = np.sum(PK_scores_train, axis=-1)
        PK_scores_test = np.sum(PK_scores_test, axis=-1)

        reject_thres_f1 = find_optimal_thres_f1(PK_scores_train, self.isbad_train)
        reject_thres_acc = find_optimal_thres_acc(PK_scores_train, self.isbad_train)

        isbad_pred_acc = (PK_scores_test > reject_thres_acc)
        acc, _ = evaluationMetric(self.isbad_test, isbad_pred_acc)
        isbad_pred_f1 = (PK_scores_test > reject_thres_f1)
        _, f1 = evaluationMetric(self.isbad_test, isbad_pred_f1)

        self.res['PKT_acc'] = acc
        self.res['PKT_f1'] = f1                           


    def CalibratedConfidenceScoreFilter(self, usePKLoss=False):
        mtname = 'CCS_%s_%s' % (self.modelname, self.ntype=='adv')
        if not self.load:        
            T = getTemperature(flattern(self.X_train, self.is_all_num), self.y_train, copy.deepcopy(self.clf), modeltype=self.modelname)
            cali_scores_train = getCalibratedScore(flattern(self.X_train_mix, self.is_all_num), T, self.clf, modeltype=self.modelname)
            reject_thres_acc = -find_optimal_thres_acc(-cali_scores_train, self.isbad_train)
            reject_thres_f1 = -find_optimal_thres_f1(-cali_scores_train, self.isbad_train)
            if self.save:
                np.save(os.path.join(self.DATA_DIR, 'thres_%s_%d.npy' % (mtname, self.random_id)), np.array([T, reject_thres_acc, reject_thres_f1]))
        else:
            thres = np.load(os.path.join(self.DATA_DIR, 'thres_%s_%d.npy' % (mtname, self.random_id)))
            T = thres[0]
            reject_thres_acc = thres[1]
            reject_thres_f1 = thres[2]

        cali_scores_test = getCalibratedScore(flattern(self.X_test_mix, self.is_all_num), T, self.clf, modeltype=self.modelname)

        print('Method: Calibrated Confidence Score w/o PK Loss')    
        isbad_pred_acc = (cali_scores_test < reject_thres_acc)
        acc, _ = evaluationMetric(self.isbad_test, isbad_pred_acc)
        isbad_pred_f1 = (cali_scores_test < reject_thres_f1)
        _, f1 = evaluationMetric(self.isbad_test, isbad_pred_f1)
        self.res['CCS_acc'] = acc
        self.res['CCS_f1'] = f1        

    def NearestNeighbourFilter(self, alpha=0.05, usePKLoss=False):
        mtname = 'KNN_%s_%s' % (self.modelname, self.ntype=='adv')
        if not self.load: 
            knn_scores_train = knnScore(self.X_train, self.y_train, self.X_train_mix, 10, self.dtypes, alpha)
            reject_thres_acc = find_optimal_thres_acc(knn_scores_train, self.isbad_train)
            reject_thres_f1 = find_optimal_thres_f1(knn_scores_train, self.isbad_train)
            if self.save:
                np.save(os.path.join(self.DATA_DIR, 'thres_%s_%d.npy' % (mtname, self.random_id)), np.array([0, reject_thres_acc, reject_thres_f1]))
        else:
            thres = np.load(os.path.join(self.DATA_DIR, 'thres_%s_%d.npy' % (mtname, self.random_id)))
            reject_thres_acc = thres[1]
            reject_thres_f1 = thres[2]

        knn_scores_test = knnScore(self.X_train, self.y_train, self.X_test_mix, 10, self.dtypes, alpha)

        print('Method: Nearest Neighbour w/o PK Loss')    
        isbad_pred_acc = (knn_scores_test > reject_thres_acc)
        acc, _ = evaluationMetric(self.isbad_test, isbad_pred_acc)
        isbad_pred_f1 = (knn_scores_test > reject_thres_f1)
        _, f1 = evaluationMetric(self.isbad_test, isbad_pred_f1)
        self.res['KNN_acc'] = acc
        self.res['KNN_f1'] = f1        

    def RVAEFilter(self, useRawFeature=False, usePKLoss=False):
        if 'text' not in self.dtypes:
            train_mix_size = self.X_train_mix.shape[0]
            X_mix = np.concatenate((self.X_train_mix, self.X_test_mix), axis=0)
            idx_mix = np.concatenate((self.idx_train_mix, self.idx_test_mix), axis=0)
            RVAE_train_set = RVAE_ds(self.X_train, self.idx_train, self.vec, self.dtypes)
            RVAE_mix_set = RVAE_ds(X_mix, idx_mix, self.vec, self.dtypes)

            train_loader = torch.utils.data.DataLoader(RVAE_train_set, batch_size=150, shuffle=True)
            mix_loader = torch.utils.data.DataLoader(RVAE_mix_set, batch_size=150, shuffle=True)

            md = Metadata()

            scores_train, scores_mix = get_outlier_scores(md, train_loader, RVAE_train_set.X, mix_loader, RVAE_mix_set.X, RVAE_train_set)

            features_train = scores_mix[:train_mix_size]
            features_test = scores_mix[train_mix_size:]

            if useRawFeature:
                print('RVAE w/ Raw Feature')
                features_train = np.concatenate((features_train, flattern(self.X_train_mix, self.is_all_num)), axis=-1)
                features_test = np.concatenate((features_test, flattern(self.X_test_mix, self.is_all_num)), axis=-1)

            class_train = self.clf.predict(flattern(self.X_train_mix, self.is_all_num))
            class_test = self.clf.predict(flattern(self.X_test_mix, self.is_all_num))

            acc, f1 = evaluation_LR_class(features_train, self.isbad_train,
                  features_test, self.isbad_test, class_train, class_test)
        else:
            acc = 0
            f1 = 0
            
        if useRawFeature:
            self.res['RVAERaw_acc'] = acc
            self.res['RVAERaw_f1'] = f1
        else:
            self.res['RVAE_acc'] = acc
            self.res['RVAE_f1'] = f1

    def TheOddsAreOddFilter(self, sample_count=5, eps=1, num_std=0.05, useAbs=False):
        mtname = 'TOAO_%s_%s' % (self.modelname, self.ntype=='adv')
        noise_type_group = ['normal', 'uniform', 'bernoulli']
        magnitude_group = [0.1, 0.5, -0.1, -0.5]
        isbad_pred_group_acc = []
        isbad_pred_group_f1 = []
        thres_dict = {}
        for noise_type in noise_type_group:
            thres_dict[noise_type] = {}

        if self.load:
            with open(os.path.join(self.DATA_DIR, 'thres_%s_%d.json' % (mtname, self.random_id)), 'r') as fp:
                thres_dict = json.load(fp)

        for noise_type in noise_type_group:
            for magnitude in magnitude_group:
                if noise_type == 'normal' and magnitude < 0:
                    continue
                print('noise type: %s magnitude: %f' % (noise_type, magnitude))
                if not self.load:
                    log_odds_diff_train = get_log_odds_diff_new(self.X_train_mix, self.vec, self.dtypes, 
                            sample_count, self.clf, noise_type, magnitude, modeltype = self.modelname)
                    reject_thres_acc = find_optimal_thres_acc(log_odds_diff_train, self.isbad_train)

                    thres_dict[noise_type][magnitude] = reject_thres_acc.item()
                    #reject_thres_f1 = find_optimal_thres_f1(log_odds_diff_train, self.isbad_train)
                else:
                    reject_thres_acc = thres_dict[noise_type][str(magnitude)]

                log_odds_diff_test = get_log_odds_diff_new(self.X_test_mix, self.vec, self.dtypes, 
                        sample_count, self.clf, noise_type, magnitude, modeltype = self.modelname)
                isbad_pred_group_acc.append((log_odds_diff_test > reject_thres_acc))
                isbad_pred_group_f1.append((log_odds_diff_test > reject_thres_acc))

        if self.save:
            with open(os.path.join(self.DATA_DIR, 'thres_%s_%d.json' % (mtname, self.random_id)), 'w') as fp:
                print(thres_dict)
                json.dump(thres_dict, fp)


        isbad_mean_acc = np.mean(np.stack(isbad_pred_group_acc, axis=0), axis=0)
        isbad_pred_acc = isbad_mean_acc >= 0.5

        isbad_mean_f1 = np.mean(np.stack(isbad_pred_group_f1, axis=0), axis=0)
        isbad_pred_f1 = isbad_mean_f1 >= 0.5
        
        acc, _ = evaluationMetric(self.isbad_test, isbad_pred_acc)
        _, f1 = evaluationMetric(self.isbad_test, isbad_pred_f1)
        self.res['TOAO_acc'] = acc
        self.res['TOAO_f1'] = f1           

    def ExtraClassFilter(self, usePKLoss=False):
        if usePKLoss:
            mtname = 'MWOCPK_%s_%s' % (self.modelname, self.ntype=='adv')
        else:
            mtname = 'MWOC_%s_%s' % (self.modelname, self.ntype=='adv')

        if not self.load:
            if usePKLoss:
                print('Method: MWOC w/ PK Loss')
                features_train = np.concatenate((flattern(self.X_train_mix, self.is_all_num), self.PKloss_train), axis=-1)
                features_test = np.concatenate((flattern(self.X_test_mix, self.is_all_num), self.PKloss_test), axis=-1)
            else:
                print('Method: MWOC w/o PK Loss')
                features_train = flattern(self.X_train_mix, self.is_all_num)
                features_test = flattern(self.X_test_mix, self.is_all_num)
        else:
            if usePKLoss:
                print('Method: MWOC w/ PK Loss')
                features_test = np.concatenate((flattern(self.X_test_mix, self.is_all_num), self.PKloss_test), axis=-1)
            else:
                print('Method: MWOC w/o PK Loss')
                features_test = flattern(self.X_test_mix, self.is_all_num)

        if not self.load:         
            if self.modelname == 'svm':
                exclf = LinearSVC()
                pred_mix = self.clf.predict(flattern(self.X_train_mix, self.is_all_num))
                X_class_zero = features_train[np.logical_and(pred_mix == -1, self.isbad_train == 0)]
                X_class_one = features_train[np.logical_and(pred_mix == 1, self.isbad_train == 0)]
                X_class_two = features_train[self.isbad_train == 1][:max(X_class_zero.shape[0], X_class_one.shape[0])]

            if self.modelname == 'lr':
                exclf = LogisticRegression(random_state=0)
                pred_mix = self.clf.predict(flattern(self.X_train_mix, self.is_all_num))
                X_class_zero = features_train[np.logical_and(pred_mix == 0, self.isbad_train == 0)]
                X_class_one = features_train[np.logical_and(pred_mix == 1, self.isbad_train == 0)]
                X_class_two = features_train[self.isbad_train == 1][:max(X_class_zero.shape[0], X_class_one.shape[0])]

            if self.modelname == 'nn':
                input_size = features_train.shape[-1]
                exclf = torchNNThreeClass(input_size=input_size)
                pred_mix = self.clf.predict(flattern(self.X_train_mix, self.is_all_num))
                X_class_zero = features_train[np.logical_and(pred_mix == 0, self.isbad_train == 0)]
                X_class_one = features_train[np.logical_and(pred_mix == 1, self.isbad_train == 0)]
                X_class_two = features_train[self.isbad_train == 1][:max(X_class_zero.shape[0], X_class_one.shape[0])]

            X = X_class_zero
            X = np.concatenate((X, X_class_one), axis=0)
            X = np.concatenate((X, X_class_two), axis=0)

            y_class_zero = np.zeros(X_class_zero.shape[0])
            y_class_one = np.zeros(X_class_one.shape[0]) + 1
            y_class_two = np.zeros(X_class_two.shape[0]) + 2

            y = y_class_zero
            y = np.concatenate((y, y_class_one))
            y = np.concatenate((y, y_class_two))
            
            exclf.fit(X.astype(np.float32), y.astype(np.float32))

            if self.save:
                if self.modelname in ['svm', 'lr']:
                    dump(exclf, os.path.join(self.DATA_DIR, 'exclf_%s_%d.joblib' % (mtname, self.random_id)))
                if self.modelname == 'nn':
                    exclf.save(self.DATA_DIR, 'exclf_%s_%d' % (mtname, self.random_id))
        else:
            if self.modelname in ['svm', 'lr']:
                exclf = load(os.path.join(self.DATA_DIR, 'exclf_%s_%d.joblib' % (mtname, self.random_id)))
            if self.modelname == 'nn':
                exclf = torchNNThreeClass(input_size=features_test.shape[-1])
                exclf.load(self.DATA_DIR, 'exclf_%s_%d' % (mtname, self.random_id))

        isbad_pred = (exclf.predict(features_test.astype(np.float32)) == 2)
        acc, f1 = evaluationMetric(self.isbad_test, isbad_pred)

        if usePKLoss:
            self.res['PKMWOC_acc'] = acc
            self.res['PKMWOC_f1'] = f1
        else:
            self.res['MWOC_acc'] = acc
            self.res['MWOC_f1'] = f1


def evaluationMetric(truth, pred):
    acc = accuracy_score(truth, pred)
    f1 = f1_score(truth, pred)
    return acc, f1

def pack_res(res):
    packed_res = {}
    for key in res[0]:
        packed_res[key] = np.stack([sub_res[key] for sub_res in res])

    return packed_res

def repeated_exps(name, ntype, modelname, artificial_components, level='medium', count_group=range(5), runTOAO=True, 
    filterLoad=False, filterSave=False):
    DATA_DIR = '../data/%s/' % name
    dtypes = dtype_dict[name]
    is_all_num = (dtypes.count('numeric') == len(dtypes))

    if not os.path.exists('../data/res/'):
        os.makedirs('../data/res/')
    if not os.path.exists('../data/res/test/'):
        os.makedirs('../data/res/test/')


    if is_all_num:
        sample_count = 1
    else:
        sample_count = 10

    main_res = []

    for count in count_group:
        print('############################################')
        print('#                    %d                     #' % count)
        print('############################################')

        if modelname == 'svm':
            clf = LinearSVC()
        if modelname == 'lr':
            clf = LogisticRegression(random_state=0)
        if modelname == 'nn':
            ds = np.load(DATA_DIR + '%s_train_test_%d.npz' % (name, count), allow_pickle=True, encoding="latin1")
            X_train = flattern(ds['X_train'], is_all_num)
            input_size = X_train.shape[-1]
            clf = torchNN(input_size=input_size)
            clf.load(DATA_DIR, name+'_'+str(count))

        testFilter = Filters(name, dtypes, clf, ntype, modelname, count, artificial_components=artificial_components, 
            level=level, sample_count=sample_count, load=filterLoad, save=filterSave)
        testFilter.RawFeatureFilter(usePKLoss=False)
        testFilter.RawFeatureFilter(usePKLoss=True)
        testFilter.PKLossFilter()
        testFilter.NearestNeighbourFilter()
        testFilter.RVAEFilter(useRawFeature=False)
        testFilter.RVAEFilter(useRawFeature=True)
        testFilter.ExtraClassFilter(usePKLoss=False)
        testFilter.ExtraClassFilter(usePKLoss=True)
        if runTOAO:
            testFilter.TheOddsAreOddFilter()
        testFilter.CalibratedConfidenceScoreFilter()

        main_res.append(testFilter.res)

    main_res_packed = pack_res(main_res)

    if len(artificial_components) == 1:
        print('The artificial noise is exact.')
        np.savez('../data/' + 'res/test/%s_%s_%s_victim_detect_exact_%s.npz' % (name, ntype, modelname, level), **main_res_packed)
    else:
        np.savez('../data/' + 'res/test/%s_%s_%s_victim_detect_res_%s.npz' % (name, ntype, modelname, level), **main_res_packed)

def time_analysis(name, ntype, modelname, artificial_components, level='medium', count_group=range(5), runTOAO=True):
    DATA_DIR = '../data/%s/' % name
    dtypes = dtype_dict[name]
    is_all_num = (dtypes.count('numeric') == len(dtypes))

    if is_all_num:
        sample_count = 1
    else:
        sample_count = 10

    count = 0

    if modelname == 'svm':
        clf = LinearSVC()
    if modelname == 'lr':
        clf = LogisticRegression(random_state=0)
    if modelname == 'nn':
        ds = np.load(DATA_DIR + '%s_train_test_%d.npz' % (name, count), allow_pickle=True, encoding="latin1")
        X_train = flattern(ds['X_train'], is_all_num)
        input_size = X_train.shape[-1]
        clf = torchNN(input_size=input_size)
        clf.load(DATA_DIR, name+'_'+str(count))

    testFilter = Filters(name, dtypes, clf, ntype, modelname, count, artificial_components=artificial_components, 
        level=level, sample_count=sample_count)
    testFilter.runTimeAnalysis(sample_count=sample_count)


def repeated_exps_bc(name, ntype, modelname, artificial_components, level='medium', count_group=range(5)):
    DATA_DIR = '../data/%s/' % name
    dtypes = dtype_dict[name]
    is_all_num = (dtypes.count('numeric') == len(dtypes))

    if not os.path.exists('../data/res/'):
        os.makedirs('../data/res/')
    if not os.path.exists('../data/res/test/'):
        os.makedirs('../data/res/test/')

    if is_all_num:
        sample_count = 1
    else:
        sample_count = 10

    main_res = []

    for count in count_group:
        print('############################################')
        print('#                    %d                     #' % count)
        print('############################################')

        if modelname == 'svm':
            clf = LinearSVC()
        if modelname == 'lr':
            clf = LogisticRegression(random_state=0)
        if modelname == 'nn':
            ds = np.load(DATA_DIR + '%s_train_test_%d.npz' % (name, count), allow_pickle=True, encoding="latin1")
            X_train = flattern(ds['X_train'], is_all_num)
            input_size = X_train.shape[-1]
            clf = torchNN(input_size=input_size)
            clf.load(DATA_DIR, name+'_'+str(count))

        testFilter = Filters(name, dtypes, clf, ntype, modelname, count, artificial_components=artificial_components,
            level=level, sample_count=sample_count)

        testFilter.RawFeatureFilter(usePKLoss=True)
        testFilter.RawFeatureFilterUnified(usePKLoss=True)
        testFilter.PKLossFilterThreshold()

        main_res.append(testFilter.res)

    main_res_packed = pack_res(main_res)
  
    np.savez('../data/' + 'res/test/%s_%s_%s_victim_detect_mbres_%s.npz' % (name, ntype, modelname, level), **main_res_packed)

def evaluateTestTime(name, level='medium', runAdv=False, ntype_group = ['random', 'system'], modelname_group = ['nn', 'svm', 'lr'], 
    mix_artificial=True, runTOAO=True, useTiny=True, filterLoad=False, filterSave=False):   
    if runAdv:
        ntype_group.append('adv')

    for ntype in ntype_group:
        for modelname in modelname_group:
            if mix_artificial:         
                if ntype == 'adv':
                    artificial_components = [('random', 'alow'), ('random', 'amedium'), ('random', 'ahigh'), ('adv', 'fgm')]
                else:
                    artificial_components = [('random', 'alow'), ('random', 'amedium'), ('random', 'ahigh')]
            else:
                if ntype == 'adv':
                    artificial_components = [('adv', 'fgm')]
                else:
                    artificial_components = [('random', level)]

            if ntype == 'adv' and useTiny:
                artificial_components.append(('random', 'atiny'))

            print('Noise Type: %s, Model: %s, Level: %s' % (ntype, modelname, level))
            repeated_exps(name, ntype, modelname, artificial_components, level=level, runTOAO=runTOAO, 
                filterLoad=filterLoad, filterSave=filterSave)

def testTimeAnalysis(name, level='medium', runAdv=False, ntype_group = ['random'], modelname_group = ['nn', 'svm', 'lr'], 
    mix_artificial=True, runTOAO=True):   

    for ntype in ntype_group:
        for modelname in modelname_group:
            if mix_artificial:         
                if ntype == 'adv':
                    artificial_components = (('random', 'alow'), ('random', 'amedium'), ('random', 'ahigh'), ('adv', 'fgm'))
                else:
                    artificial_components = (('random', 'alow'), ('random', 'amedium'), ('random', 'ahigh'))
            else:
                if ntype == 'adv':
                    artificial_components = [('adv', 'fgm')]
                else:
                    artificial_components = [('random', level)]
            print('Noise Type: %s, Model: %s, Level: %s' % (ntype, modelname, level))
            time_analysis(name, ntype, modelname, artificial_components, level=level, runTOAO=runTOAO)


def printResTestTime(name, level='medium'):
    methods = ['Raw', 'RVAE', 'RVAERaw', 'CCS', 'KNN', 'TOAO', 'MWOC', 'PKRaw']
    methodnames = {'Raw': 'RF', 'RVAE': 'RVAE', 'RVAERaw': 'RVAE+', 'CCS': 'CCS', 'KNN': 'KNN', 'TOAO': 'TOAO',
                'MWOC': 'MWOC', 'PKRaw': 'Picket'}
    if name in ['wine', 'HTRU2']:
        ntypes = ['random', 'system', 'adv']
    else:
        ntypes = ['random', 'system']
    models = ['lr', 'svm', 'nn']

    print('F1 Scores of Victim Sample Detection')
    for ntype in ntypes:
        print('---------------------------------------------')
        print('Noise Type: %s' % ntype)

        for modelname in models:
            print('Downstream Model: %s' % modelname)         
            ds = np.load('../data/' + 'res/test/%s_%s_%s_victim_detect_res_%s.npz' % (name, ntype, modelname, level), 
                         allow_pickle=True, encoding="latin1")
            for mt in methods:
                print('%s: %.4f' % (methodnames[mt], np.mean(ds['%s_f1'%mt])), end=' ')
            print(' ')    
            for mt in methods:
                print('& %.4f' % (np.mean(ds['%s_f1'%mt])), end=' ')
            print(' ') 

def printResTestTimeExact(name, level='medium', ntype_group = ['random'], modelname_group = ['nn', 'svm', 'lr']):
    methods = ['Raw', 'RVAE', 'RVAERaw', 'CCS', 'KNN', 'TOAO', 'MWOC', 'PKRaw']
    methodnames = {'Raw': 'RF', 'RVAE': 'RVAE', 'RVAERaw': 'RVAE+', 'CCS': 'CCS', 'KNN': 'KNN', 'TOAO': 'TOAO',
                'MWOC': 'MWOC', 'PKRaw': 'Picket'}
    if name in ['wine', 'HTRU2']:
        ntypes = ['random', 'system', 'adv']
    else:
        ntypes = ['random', 'system']
    models = modelname_group

    ntypes = ntype_group

    print('F1 Scores of Victim Sample Detection')
    for ntype in ntypes:
        print('---------------------------------------------')
        print('Noise Type: %s' % ntype)

        for modelname in models:
            print('Downstream Model: %s' % modelname)         
            ds = np.load('../data/' + 'res/test/%s_%s_%s_victim_detect_exact_%s.npz' % (name, ntype, modelname, level), 
                         allow_pickle=True, encoding="latin1")
            for mt in methods:
                print('%s: %.4f' % (methodnames[mt], np.mean(ds['%s_f1'%mt])), end=' ')
            print(' ')    
            for mt in methods:
                print('& %.4f' % (np.mean(ds['%s_f1'%mt])), end=' ')
            print(' ') 

def validate_per_class_detector(name, level='medium', runAdv=False, ntype_group = ['random', 'system'], modelname_group = ['nn', 'svm', 'lr']):   
    if runAdv:
        ntype_group.append('adv')

    for ntype in ntype_group:
        for modelname in modelname_group:
            if ntype == 'adv':
                artificial_components = (('random', 'alow'), ('random', 'amedium'), ('random', 'ahigh'), ('adv', 'fgm'), ('random', 'atiny'))
            else:
                artificial_components = (('random', 'alow'), ('random', 'amedium'), ('random', 'ahigh'))
            print('Noise Type: %s, Model: %s, Level: %s' % (ntype, modelname, level))
            repeated_exps_bc(name, ntype, modelname, artificial_components=artificial_components, level=level)

def printTestTimeMicroBM(name, ntype, modelname, level='medium',):
    methods = ['PKRaw', 'PKRawU', 'PKT']
    methodnames = {'PKRaw': 'Per-class Detectors', 'PKRawU': 'Unified Detector', 'PKT': 'Score-based Detector'}

    print('F1 Scores of Victim Sample Detection')
    print('---------------------------------------------')
    print('Noise Type: %s' % ntype)

    if ntype == 'random':
        ntype_known = True
    else:
        ntype_known = False

    print('Downstream Model: %s' % modelname)         
    ds = np.load('../data/' + 'res/test/%s_%s_%s_victim_detect_mbres_%s.npz' % (name, ntype, modelname, level), 
                 allow_pickle=True, encoding="latin1")
    for mt in methods:
        print('%s: %.4f' % (methodnames[mt], np.mean(ds['%s_f1'%mt])), end=' ')
    print(' ')    


def PK_model_prepare(name, batch_size=1000, epochs=500, transformer_layer=6, hidden_dim=64, count_group=range(5)):
    DATA_DIR = '../data/%s/' % name

    dtypes = dtype_dict[name]
    input_dim = input_dim_dict[name]

    if name == 'HTRU2':
        hidden_dim = 16
        transformer_layer = 1
        epochs = 300

    for count in count_group:
        print('############################################')
        print('#                    %d                     #' % count)
        print('############################################')

        ds = np.load(DATA_DIR + '%s_train_test_%d.npz' % (name, count), allow_pickle=True, encoding="latin1")
        X_train = ds['X_train']
        vec = ds['vec']
        idx_train = ds['idx_train']

        attribute_info = [Attribute(vec[i]) for i in range(len(vec))]

        param = {
            'description': 'Contrastive Transformer',
            'model_dim': input_dim,
            'input_dim': input_dim,
            'attribute_num': len(attribute_info),
            'transformer_layer': transformer_layer,
            'head_num': 2,
            'hidden_dim': hidden_dim,
            'dropout': 0.1,
            'numerical_ids': [i for i, x in enumerate(dtypes) if x == "numeric"],
            'batch_size': batch_size,
            'epochs': epochs,
            'opt_factor': 0.1,
            'warmup': 300,
            'adam_lr': 0,
            'adam_betas': (0.9, 0.98),
            'adam_eps': 1e-9,
            'neg_sample_num': 4,
            'random_mask': False,
            'fast': True,
            'structure_mask_type': 'none', # could be 'hard', 'soft', 'sample'
            'holdout_ratio': 0.2,
            'useEncoding': True,
            'categorical_ids': [i for i, x in enumerate(dtypes) if x == 'categorical'],
        }

        

        X_train_PK = torch.Tensor(X_train)

        flag = 1
        while flag == 1:
            PicketN = PicketNetModel(param)
            PicketN.loadData(X_train_PK.double(), None, attribute_info, 
                    tuple_idx = torch.Tensor(idx_train))
            flag = PicketN.train()
        PicketN.saveModel(DATA_DIR+'%s_PKModel_%d.pt' % (name, count))


def runtime_test(batch_size=1000, epochs=100, input_dim=8, attribute_num=2):
    dtypes = ['text']*attribute_num
    vec = [np.random.rand(100, input_dim)]*attribute_num

    X_train = np.random.rand(10000, attribute_num, input_dim)
    idx_train = np.random.randint(100, size=(10000, attribute_num, input_dim))

    attribute_info = [Attribute(vec[i]) for i in range(len(vec))]

    param = {
        'description': 'Contrastive Transformer',
        'model_dim': input_dim,
        'input_dim': input_dim,
        'attribute_num': len(attribute_info),
        'transformer_layer': 6,
        'head_num': 2,
        'hidden_dim': 64,
        'dropout': 0.1,
        'numerical_ids': [i for i, x in enumerate(dtypes) if x == "numeric"],
        'batch_size': batch_size,
        'epochs': epochs,
        'opt_factor': 0.1,
        'warmup': 300,
        'adam_lr': 0,
        'adam_betas': (0.9, 0.98),
        'adam_eps': 1e-9,
        'neg_sample_num': 4,
        'random_mask': False,
        'fast': True,
        'structure_mask_type': 'none', # could be 'hard', 'soft', 'sample'
        'holdout_ratio': 0.2,
        'useEncoding': True,
        'categorical_ids': [i for i, x in enumerate(dtypes) if x == 'categorical'],
    }

    PicketN = PicketNetModel(param)

    X_train_PK = torch.Tensor(X_train)
    PicketN.loadData(X_train_PK.double(), None, attribute_info, 
                tuple_idx = torch.Tensor(idx_train))

    PicketN.train()

def runtime_test_real(name, epochs=500, transformer_layer=6):
    DATA_DIR = '../data/%s/' % name

    dtypes = dtype_dict[name]
    input_dim = input_dim_dict[name]

    count = 0

    ds = np.load(DATA_DIR + '%s_train_test_%d.npz' % (name, count), allow_pickle=True, encoding="latin1")
    X_train = ds['X_train']
    batch_size = X_train.shape[0]//5 + 1
    vec = ds['vec']
    idx_train = ds['idx_train']

    attribute_info = [Attribute(vec[i]) for i in range(len(vec))]

    param = {
        'description': 'Contrastive Transformer',
        'model_dim': input_dim,
        'input_dim': input_dim,
        'attribute_num': len(attribute_info),
        'transformer_layer': transformer_layer,
        'head_num': 2,
        'hidden_dim': 64,
        'dropout': 0.1,
        'numerical_ids': [i for i, x in enumerate(dtypes) if x == "numeric"],
        'batch_size': batch_size,
        'epochs': epochs,
        'opt_factor': 0.1,
        'warmup': 300,
        'adam_lr': 0,
        'adam_betas': (0.9, 0.98),
        'adam_eps': 1e-9,
        'neg_sample_num': 4,
        'random_mask': False,
        'fast': True,
        'structure_mask_type': 'none', # could be 'hard', 'soft', 'sample'
        'holdout_ratio': 0.2,
        'useEncoding': True,
        'categorical_ids': [i for i, x in enumerate(dtypes) if x == 'categorical'],
    }

    PicketN = PicketNetModel(param)

    X_train_PK = torch.Tensor(X_train)
    PicketN.loadData(X_train_PK.double(), None, attribute_info, 
                tuple_idx = torch.Tensor(idx_train))
    PicketN.train()

def getPKLossInference(PicketN, X_train_mix, X_test_mix, idx_train_mix, idx_test_mix, random_repeat=10):
    sample_count = random_repeat

    loss_train = None

    for _ in range(sample_count):
        loss1, loss2 = PicketN.getLossTest(torch.Tensor(X_train_mix).double(), torch.Tensor(idx_train_mix))
        if loss_train is None:
            loss_train = (loss1 + loss2)
        else:
            loss_train += (loss1 + loss2)

    loss_train = loss_train/sample_count

    ##

    loss_test = None

    for _ in range(sample_count):
        loss1, loss2 = PicketN.getLossTest(torch.Tensor(X_test_mix).double(), torch.Tensor(idx_test_mix))
        if loss_test is None:
            loss_test = (loss1 + loss2)
        else:
            loss_test += (loss1 + loss2)

    loss_test = loss_test/sample_count

    return loss_train, loss_test











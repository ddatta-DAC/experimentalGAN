#!/usr/bin/env python
# coding: utf-8

# In[ ]:



from scipy.stats import entropy
from collections import Counter
import sklearn
import os
import sys
import numpy as np
import pandas as pd
from math import log
import math
from sklearn.metrics import mutual_info_score
from itertools import combinations
from tqdm import tqdm
from pandarallel import pandarallel
pandarallel.initialize()
import multiprocessing as mp
from joblib import Parallel,delayed
from sklearn.svm import OneClassSVM
import pickle
from pathlib import Path

from sklearn.ensemble import IsolationForest

# =========================
# Check the mutual information among columns
# =========================
def calculate_MI(data_np):
    def mi_aux(idx_i, idx_j):
        X = data_np[:,idx_i]
        Y = data_np[:,idx_j]
        return mutual_info_score(X, Y)
    res = Parallel(n_jobs=mp.cpu_count())(delayed(mi_aux)(i,j) for i,j in combinations(np.arange(data_np.shape[1]),2))
    return res
    


# =======================================
def convert_to_01(data, domain_dims):
    res = []
    
    def get_one_hot(targets, nb_classes):
        res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape)+[nb_classes])
    dims_list = list(domain_dims.values())
    
    for i in range(len(dims_list)):
        _x = data[:,i]
        _x = get_one_hot(_x, dims_list[i])
        res.append(_x)
    res = np.concatenate(res,axis=1)
    print('>>',res.shape)
    return res

def get_domain_dims(DIR='us_import1'):
    with open('./../generated_data_v1/{}/domain_dims.pkl'.format(DIR),'rb') as fh:
        domain_dims = pickle.load(fh)
    return domain_dims




def check_diversity(data, domain_dims):
    a = [len(set(data[:,i])) for i in range(len(domain_dims))]
    b = list(domain_dims.values())
    return [1 - (abs(i-j)/j) for i,j in zip(a,b) ]

    
def check_relative_anomaly_score(real_data=None, test_data=None, DIR=None , domain_dims = None):
    
    _cur_path_ = os.path.abspath(__file__).replace('.py','')
    _cur_path_ = '/'.join(_cur_path_.split('/')[:-1])
    save_dir = os.path.join(_cur_path_, 'saved_model/{}'.format(DIR))
   
    path = Path(save_dir)
    path.mkdir(parents=True,exist_ok=True)
    f_path = os.path.join(save_dir, 'ad_if.pkl')
    print(f_path)
    AD_obj = None
    if not os.path.exists(f_path) and real_data is not None:
        AD_obj = IsolationForest( 
            n_estimators=100, 
            contamination=0.01, 
            n_jobs=mp.cpu_count(), 
            verbose=True
        )
        # Convert real data to one-hot encoded
        oh_data = convert_to_01( real_data, domain_dims)
        AD_obj.fit(
            oh_data
        )
        print("Model fitting done.")
        pickle.dump(AD_obj, open(f_path, 'wb'), pickle.HIGHEST_PROTOCOL)
    
    elif os.path.exists(f_path):
        
        AD_obj = pickle.load( open(f_path, 'rb'))
    print(AD_obj)
    
    if test_data is not None and AD_obj is not None:
        print(test_data.shape)

        oh_data = convert_to_01( test_data, domain_dims)
        print(oh_data.shape)

        # -1 for outliers
        y = AD_obj.predict(oh_data)
        # percentage of data points predicted as anomalies
        count_outliers = np.where(y == -1.0)[0].shape[0]
        data_len  = test_data.shape[0]
        return (count_outliers/data_len)
        
def check_KLDiv(real_data, test_data):
    column_results = []
    for i in range(real_data.shape[1]):
        x = real_data[:,i].astype(int)
        y = test_data[:,i].astype(int)
        arity = len(set(x))
        dist_x = np.zeros(arity)
        dist_y = np.zeros(arity)
        c_x = Counter(x)
        c_y = Counter(y)
        N = x.shape[0]
        
        for v,c in c_x.items():
            dist_x[v] = c
        dist_x = dist_x / N
        
        
        for v,c in c_y.items():
            dist_y[v] = c
        dist_y = dist_y / N
        column_results.append(entropy(dist_x, dist_y))
                              
    return column_results
                             
                                 
# check_relative_anomaly_score(real_data=real_data, DIR='us_import1',domain_dims= domain_dims)
# a = check_relative_anomaly_score(test_data=real_data, DIR='us_import1',domain_dims= domain_dims)
# real_data = np.load('./../generated_data_v1/us_import1/pos_data.npy')
# calculate_MI(real_data)
# domain_dims = get_domain_dims()
# check_diversity(real_data, domain_dims)
# a = convert_to_10(x, domain_dims)
# a.shape


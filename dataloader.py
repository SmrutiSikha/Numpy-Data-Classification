# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
with open('FileXY.npy', 'rb') as f:  
    X_train = np.load(f,allow_pickle=True) ;
    Y_train = np.load(f,allow_pickle=True) ;
    x_test = np.load(f,allow_pickle=True) ;

#y_score = predict_proba(x_test)

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import torch

with open('FileXY.npy', 'rb') as f:  
    X_train = np.load(f,allow_pickle=True)
    Y_train = np.load(f,allow_pickle=True)
    x_test = np.load(f,allow_pickle=True)

#y_score = predict_proba(x_test)
with open('y_test_pred.npy','rb') as f:
    y_test =np.load(f,allow_pickle=True)
print(X_train.shape)
print(Y_train.shape)
print(x_test.shape)
print(y_test.shape)

# print(X_train[0],Y_train[0])

X_train = np.expand_dims(X_train,axis=0)
Y_train = np.expand_dims(Y_train,axis=0)

print(X_train.shape)
print(Y_train.shape)

# X_train = torch.stack([torch.from_numpy(np.array(i.astype(np.float32))) for i in X_train])
# Y_train = torch.stack([torch.from_numpy(np.array(i.astype(np.float32))) for i in Y_train])

X_train = torch.from_numpy(X_train.astype(np.float32))
Y_train =torch.from_numpy(Y_train.astype(np.float32))

print(X_train.shape)
print(Y_train.shape)

train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)







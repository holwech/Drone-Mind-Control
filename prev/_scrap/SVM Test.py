import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import  svm

import numpy as np

test_data_filename = 'emd_filt_fp1_test.csv'
training_data_filename = 'emd_filt_fp1_training.csv'
filepath = 'C:\droneMC\\NN_data\\'

prefetch = 50
postfetch = 200

#Parameters for SVM
gamma=0.001
C=100

input_size = prefetch + postfetch

training_data = np.loadtxt(filepath + training_data_filename, delimiter=',', skiprows=0, usecols=(0, 1))
test_data = np.loadtxt(filepath + test_data_filename, delimiter=',', skiprows=0, usecols=(0, 1))

data_length = len(training_data[:, 0])

X_train = np.empty((data_length - prefetch + 2, prefetch + postfetch), dtype=float)
Y_train = np.empty(data_length - prefetch + 2, dtype=float)
X_test = np.empty((data_length - prefetch + 2, prefetch + postfetch), dtype=float)
Y_test = np.empty(data_length - prefetch + 2, dtype=float)

for i in range(0, data_length):
    if (i >= prefetch) and (i <= data_length - postfetch):
        X_train[i, :] = training_data[i - prefetch:i + postfetch, 0]
        Y_train[i] = training_data[i, 1]
        X_test[i, :] = test_data[i - prefetch:i + postfetch, 0]
        Y_test[i] = test_data[i, 1]

print(X_test.shape)
print(Y_test.shape)
clf = svm.SVC(gamma, C)
clf.fit(X_train,Y_train)



print('Prediction:',clf.predict(test_data[-1]))


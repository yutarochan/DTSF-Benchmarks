'''
Data Split Utility
Author: Yuya Jeremy Ong (yuyajeremyong@gmail.com)
'''
from __future__ import print_function
import sys
import torch
import numpy as np
from torch.autograd import Variable

# Auxillary Functions
def normal_std(x): return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

class DataUtil(object):
    def __init__(self, file_name, train, valid, cuda, horizon, window, normalize = 2, nan=-200):
        self.cuda = cuda
        self.P = window
        self.h = horizon

        fin = open(file_name).read().split('\n')[:-1]
        data = []
        for i in fin:
            row = []
            for y in i.split(','):
                try:
                    row.append(float(y))
                except Exception as e:
                    row.append(nan)
            data.append(row)


        self.rawdat = np.array(data)

        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape

        self.normalize = normalize
        self.scale = np.ones(self.m)
        self._normalized(self.normalize)

        self._split(int(train * self.n), int((train+valid) * self.n), self.n)

        self.scale = torch.from_numpy(self.scale).float()
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)

        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

        if self.cuda:
            self.scale = self.scale.cuda()
            self.rse = self.rse.cuda()
            self.rae = self.rae.cuda()

        self.scale = Variable(self.scale)

    def _normalized(self, normalize):
        if (normalize == 0):
            self.dat = self.rawdat

        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)

        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:,i]))
                self.dat[:,i] = self.rawdat[:,i] / np.max(np.abs(self.rawdat[:,i]))

    def _split(self, train, valid, test):
        train_set = range(self.P+self.h-1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)

    def _batchify(self, idx_set, horizon):
        n = len(idx_set)
        X = torch.zeros((n,self.P,self.m))
        Y = torch.zeros((n,self.m))

        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i,:,:] = torch.from_numpy(self.dat[start:end, :])
            Y[i,:] = torch.from_numpy(self.dat[idx_set[i], :])

        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]; Y = targets[excerpt]
            if (self.cuda):
                X = X.cuda()
                Y = Y.cuda()
            yield Variable(X), Variable(Y)
            start_idx += batch_size

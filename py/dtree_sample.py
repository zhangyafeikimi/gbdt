#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# decision tree training samples
#
# author: yafei(zhangyafeikimi@gmail.com)
#
import string
import sys


class DTreeSample:
    def __init__(self):
        self.X = []
        self.Y = []
        self.m = 0
        self.n = 0
        self.pos_n = 0
        self.neg_n = 0
        self.Y_residual = []# only for loss based decision tree
        self.Y_response = None# only for loss based decision tree


    def __len__(self):
        return self.n


    def load_liblinear(self, filename):
        """load liblinear format training samples"""
        self.__init__()

        def __feature_map_2_list(feature, max_index):
            x = [0.0] * max_index
            for index in feature.keys():
                x[index-1] = feature[index]
            return x

        fp = open(filename, 'r');
        try:
            while 1:
                line = fp.readline()
                if not line:
                    break

                array = line.split(' ')
                if len(array) == 0:
                    continue

                if array[0] == '+1' or array[0] == '1':
                    y = 1
                    self.pos_n += 1
                else:
                    y = 0
                    self.neg_n += 1

                x = {}
                max_index = 0
                for array_element in array[1:]:
                    pair = array_element.split(':')
                    if len(pair) != 2:
                        continue
                    index = string.atoi(pair[0])
                    max_index = max(max_index, index)
                    value = string.atof(pair[1])
                    x[index] = value
                x = __feature_map_2_list(x, max_index)

                self.X.append(x)
                self.Y.append(y)
                self.m = max(self.m, max_index)
        finally:
            fp.close()

        self.n = len(self.X)
        print >>sys.stderr, '[load]# of training samples %d(%d pos, %d neg), # of features %d' % (self.n, self.pos_n, self.neg_n, self.m)


    def load(self, filename):
        """load our format training samples"""
        self.__init__()
        self.m = -1

        fp = open(filename, 'r');
        try:
            while 1:
                line = fp.readline()
                if not line:
                    break

                array = line.split(' ')
                if len(array) == 0:
                    continue

                if array[0] == '+1' or array[0] == '1':
                    y = 1
                    self.pos_n += 1
                elif array[0] == '-1' or array[0] == '0':
                    y = 0
                    self.neg_n += 1
                else:
                    y = string.atof(array[0])

                x = []
                for array_element in array[1:]:
                    value = string.atof(array_element)
                    x.append(value)

                if self.m == -1:
                    self.m = len(x) - 1
                elif self.m != len(x) - 1:
                    raise Exception('training samples have different feature number')
                self.X.append(x)
                self.Y.append(y)
        finally:
            fp.close()

        self.n = len(self.X)
        print >>sys.stderr, '[load]# of training samples %d(%d pos, %d neg), # of features %d' % (self.n, self.pos_n, self.neg_n, self.m)


    def add_xy(self, x, y):
        self.X.append(x)
        self.Y.append(y)
        self.n += 1
        if y == 1:
            self.pos_n += 1
        else:
            self.neg_n += 1


    def add_xyr(self, x, y, y_residual):
        self.add_xy(x, y)
        self.Y_residual.append(y_residual)

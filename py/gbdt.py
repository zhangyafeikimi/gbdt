#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# GBDT
#
# author: yafei(zhangyafeikimi@gmail.com)
#
# TODO sample rate
#
import sys
from dtree_loss import DTreeLoss
from dtree_parameter import DTreeParameter
from dtree_sample import DTreeSample


class GBDT:
    def __init__(self, sample):
        self.sample = sample
        self.trees = None
        self.F0 = None


    def train(self, param):
        self.trees = []
        self.param = param

        for i in range(0, self.param.tree_number):
            tree = DTreeLoss(self.sample, param)

            if i == 0:
                residual, self.F0 = tree.initial_residual()
            else:
                last_tree = self.trees[i-1]
                residual = last_tree.next_residual()

            print >>sys.stderr, 'training tree #%d' % (i)
            tree.train(residual)
            self.trees.append(tree)


    def predict(self, x):
        y = self.F0
        for tree in self.trees:
            y += tree.predict(x)
        return y


if __name__ == '__main__':
    param = DTreeParameter()
    param.max_level = 4
    param.split_threshold = 0.8
    param.max_attr_try_time = 1000
    param.tree_number = 20
    param.learning_rate = 0.5

    sample = DTreeSample()
    sample.load_liblinear('heart_scale.txt')

    gbdt = GBDT(sample)
    gbdt.train(param)
    print gbdt.predict([0.708333,1,1,-0.320755,-0.105023,-1,1,-0.419847,-1,-0.225806,0,1,-1])
    print gbdt.predict([0.583333,-1,0.333333,-0.603774,1,-1,1,0.358779,-1,-0.483871,0,-1,1])

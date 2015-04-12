#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# decision tree based on information gain
#
# author: yafei(zhangyafeikimi@gmail.com)
#
import math
import sys
from dtree_node import DTreeNode
from dtree_parameter import DTreeParameter
from dtree_sample import DTreeSample


def entropy(pos_n, neg_n):
    def log2(x):
        return math.log(x, 2)

    def part(n, p):
        if n == 0:
            return 0
        else:
            return - p * log2(p)

    total = pos_n + neg_n
    if total == 0:
        return 0.0

    total = float(total)
    pos_p = pos_n / total
    neg_p = neg_n / total
    return part(pos_n, pos_p) + part(neg_n, neg_p)


def information_gain(pos_n, neg_n, le_pos_n, le_neg_n, gt_pos_n, gt_neg_n):
    n = pos_n + neg_n
    le_n = le_pos_n + le_neg_n
    gt_n = gt_pos_n + gt_neg_n
    ig = entropy(pos_n, neg_n) - le_n / float(n) * entropy(le_pos_n, le_neg_n) - gt_n / float(n) * entropy(gt_pos_n, gt_neg_n)
    return ig


class DTreeGain(DTreeNode):
    """decision tree based on information gain"""
    def __init__(self, sample, param):
        DTreeNode.__init__(self, sample)
        self.param = param


    def __information_gain(self):
        """get the largest information gain over all attributes"""
        max_gain = -1.0
        attr_index = None
        partition_value = None

        for i in range(0, self.sample.m+1):
            gain, _partition_value = self.__information_gain_attr(i)
            if gain > max_gain:
                max_gain = gain
                attr_index = i
                partition_value = _partition_value
        return max_gain, attr_index, partition_value


    def __information_gain_attr(self, attr_index):
        """get the largest information gain over a attribute"""
        try_values = self.sample_attribute(attr_index, self.param.max_attr_try_time)
        max_gain = -1.0
        partition_value = None

        for _partition_value in try_values:
            gain = self.__information_gain_attr_value(attr_index, _partition_value)
            if (gain > max_gain):
                max_gain = gain
                partition_value = _partition_value

        return max_gain, partition_value


    def __information_gain_attr_value(self, attr_index, partition_value):
        """get the information gain over a attribute at a partition value"""
        le_pos_n = 0# <= partition_value
        le_neg_n = 0# <= partition_value
        gt_pos_n = 0# > partition_value
        gt_neg_n = 0# > partition_value

        for i in range(0, self.sample.n):
            value = self.sample.X[i][attr_index]
            if value <= partition_value:
                if self.sample.Y[i] == 1:
                    le_pos_n += 1
                else:
                    le_neg_n += 1
            else:
                if self.sample.Y[i] == 1:
                    gt_pos_n += 1
                else:
                    gt_neg_n += 1

        return information_gain(self.sample.pos_n, self.sample.neg_n, le_pos_n, le_neg_n, gt_pos_n, gt_neg_n)


    def __split(self):
        """split current node"""
        max_gain, attr_index, partition_value = self.__information_gain()

        self.leaf = False
        self.attr_index = attr_index
        self.partition_value = partition_value

        tn_le = DTreeGain(DTreeSample(), self.param)
        tn_le.sample.m = self.sample.m
        tn_gt = DTreeGain(DTreeSample(), self.param)
        tn_gt.sample.m = self.sample.m

        for i in range(0, self.sample.n):
            x = self.sample.X[i]
            y = self.sample.Y[i]
            attr_value = x[attr_index]
            if attr_value <= partition_value:
                tn_le.sample.add_xy(x, y)
            else:
                tn_gt.sample.add_xy(x, y)

        left_n = len(tn_le.sample)
        right_n = len(tn_gt.sample)
        print >>sys.stderr, '[split]level=%d, max_gain=%f, attr_index=%d, partition_value=%f, left/right=%d/%d' % (self.level, max_gain, attr_index, partition_value, left_n, right_n)

        if left_n == 0:
            tn_le = None
        if right_n == 0:
            tn_gt = None
        return tn_le, tn_gt


    def train(self):
        """train the model"""
        self.level = 0

        stack = [self]
        while 1:
            if len(stack) == 0:
                break

            tn = stack.pop()
            level = tn.level
            if (level >= self.param.max_level) or (max(tn.sample.pos_n, tn.sample.neg_n) / float(tn.sample.n)) >= self.param.split_threshold:
                tn.leaf = True
                tn.value = float(tn.sample.pos_n) / float(tn.sample.n)
                print >>sys.stderr, '[train]leaf node level=%d, pos/neg=%d/%d' % (level, tn.sample.pos_n, tn.sample.neg_n)
                continue

            left, right = tn.__split()
            tn.left = left
            tn.right = right

            if left is not None:
                left.level = level + 1
                stack.append(left)
            if right is not None:
                right.level = level + 1
                stack.append(right)

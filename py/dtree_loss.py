#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# decision tree based on squared error loss
#
# author: yafei(zhangyafeikimi@gmail.com)
#
import copy
import sys
from dtree_node import DTreeNode
from dtree_parameter import DTreeParameter
from dtree_sample import DTreeSample


class DTreeLoss(DTreeNode):
    """decision tree based on loss function"""
    def __init__(self, sample, param):
        DTreeNode.__init__(self, sample)
        self.param = param


    def __residual_2_response(self):
        # residual and response are identical
        self.sample.Y_response = self.sample.Y_residual


    def __gain(self):
        """get the largest gain over all attributes"""
        max_gain = -sys.float_info.max
        attr_index = None
        partition_value = None
        y_value_left = None
        y_value_right = None

        for i in range(0, self.sample.m):
            gain, _partition_value, _y_value_left, _y_value_right = self.__gain_attr(i)
            if gain > max_gain:
                max_gain = gain
                attr_index = i
                partition_value = _partition_value
                y_value_left = _y_value_left
                y_value_right = _y_value_right
        return max_gain, attr_index, partition_value, y_value_left, y_value_right


    def __gain_attr(self, attr_index):
        """get the largest gain over a attribute"""
        try_values = self.sample_attribute(attr_index, self.param.max_attr_try_time)
        max_gain = -sys.float_info.max
        partition_value = None
        y_value_left = None
        y_value_right = None

        for _partition_value in try_values:
            gain, _y_value_left, _y_value_right = self.__gain_attr_value(attr_index, _partition_value)
            if gain > max_gain:
                max_gain = gain
                partition_value = _partition_value
                y_value_left = _y_value_left
                y_value_right = _y_value_right

        return max_gain, partition_value, y_value_left, y_value_right


    def __gain_attr_value(self, attr_index, partition_value):
        """get the gain over a attribute at a partition value"""
        return self.__ls_loss_attr_value_for_ls_tree(attr_index, partition_value)


    def __ls_loss_attr_value_for_ls_tree(self, attr_index, partition_value):
        le_n = 0
        gt_n = 0
        y_value_left = 0.0
        y_value_right = 0.0
        for i in range(0, self.sample.n):
            x = self.sample.X[i]
            x_attr_value = x[attr_index]
            response = self.sample.Y_response[i]
            if x_attr_value <= partition_value:
                y_value_left += response
                le_n += 1
            else:
                y_value_right += response
                gt_n += 1

        if le_n != 0:
            y_value_left /= float(le_n)
        if gt_n != 0:
            y_value_right /= float(gt_n)
        loss = self.__ls_loss_attr_value(attr_index, partition_value, y_value_left, y_value_right)
        return -loss, y_value_left, y_value_right


    def __ls_loss_attr_value(self, attr_index, partition_value, y_value_left, y_value_right):
        loss = 0.0
        for i in range(0, self.sample.n):
            x = self.sample.X[i]
            x_attr_value = x[attr_index]
            if x_attr_value <= partition_value:
                response = self.sample.Y_response[i] - y_value_left
            else:
                response = self.sample.Y_response[i] - y_value_right
            loss += (response * response)
        return loss


    def __split(self):
        """split current node"""
        max_gain, attr_index, partition_value, y_value_left, y_value_right = self.__gain()

        self.leaf = False
        self.attr_index = attr_index
        self.partition_value = partition_value
        self.value = None

        tn_le = DTreeLoss(DTreeSample(), self.param)
        tn_le.sample.m = self.sample.m
        tn_le.level = self.level + 1
        tn_le.value = y_value_left
        tn_gt = DTreeLoss(DTreeSample(), self.param)
        tn_gt.sample.m = self.sample.m
        tn_gt.level = self.level + 1
        tn_gt.value = y_value_right

        for i in range(0, self.sample.n):
            x = self.sample.X[i]
            y = self.sample.Y[i]
            y_residual = self.sample.Y_residual[i]
            attr_value = x[attr_index]
            if attr_value <= partition_value:
                tn_le.sample.add_xyr(x, y, y_residual)
            else:
                tn_gt.sample.add_xyr(x, y, y_residual)
        tn_le.__residual_2_response()
        tn_gt.__residual_2_response()

        left_n = len(tn_le.sample)
        right_n = len(tn_gt.sample)
        print >>sys.stderr, '[split]level=%d, max_gain=%f, attr_index=%d, partition_value=%f, left/right=%d/%d' % (self.level, max_gain, attr_index, partition_value, left_n, right_n)

        if left_n == 0:
            tn_le = None
        if right_n == 0:
            tn_gt = None
        return tn_le, tn_gt


    def train(self, residual):
        if residual is None:
            residual = copy.copy(self.sample.Y)
        self.level = 0
        self.sample.Y_residual = residual
        self.__residual_2_response()

        stack = [self]
        while 1:
            if len(stack) == 0:
                break

            tn = stack.pop()
            level = tn.level
            if (level >= self.param.max_level) or (tn.sample.n <= self.param.min_samples_in_leave):
                tn.leaf = True
                tn.value = sum(tn.sample.Y) / float(tn.sample.n)
                print >>sys.stderr, '[train]leaf node level=%d, value=%f' % (level, tn.value)
                continue

            left, right = tn.__split()
            tn.left = left
            tn.right = right

            if left is not None:
                stack.append(left)
            if right is not None:
                stack.append(right)

        self.shrink()


    def initial_residual(self):
        mean_y = sum(self.sample.Y) / float(self.sample.n)
        residual = [y - mean_y for y in self.sample.Y]
        return residual, mean_y


    def next_residual(self):
        residual = copy.copy(self.sample.Y_residual)
        for i in range(0, self.sample.n):
            residual[i] -= self.predict(self.sample.X[i])
        return residual


    def shrink(self):
        if self.param.learning_rate == 1.0:
            return

        def __shrink(tn, learning_rate):
            if tn.leaf:
                tn.value *= learning_rate
            else:
                left = tn.left
                right = tn.right
                if left is not None:
                    __shrink(left, learning_rate)
                if right is not None:
                    __shrink(right, learning_rate)
        __shrink(self, self.param.learning_rate)

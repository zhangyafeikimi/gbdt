#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# decision tree node
#
# author: yafei(zhangyafeikimi@gmail.com)
#


class DTreeNode:
    def __init__(self, sample):
        # training samples
        self.sample = sample

        # node information
        self.level = None
        self.leaf = None
        self.left = None
        self.right = None

        # split information
        self.attr_index = None
        self.partition_value = None

        # trained target value(only for leaf nodes)
        self.value = None


    def dump(self):
        def __dump(tn):
            for i in range(0, tn.level):
                print '   ',

            if tn.leaf:
                print '[leaf]pos/neg=%d/%d, value=%f' % (tn.sample.pos_n, tn.sample.neg_n, tn.value)
            else:
                print '[non-leaf]attr_index=%d, partition_value=%f' % (tn.attr_index, tn.partition_value)
                left = tn.left
                right = tn.right
                if left is not None:
                    __dump(left)
                if right is not None:
                    __dump(right)
        __dump(self)


    def dump_for_dot(self, feature_map = {}):
        def __node_id(tn):
            return 'node' + hex(id(tn))

        def __node_str(tn):
            if tn.leaf:
                return __node_id(tn) + '[label="y=%.3f(%d)"][style=filled];' % (tn.value, tn.sample.n)
            else:
                return __node_id(tn) + '[label=""];'

        def __dump_for_dot(tn, feature_map):
            print '   ', __node_str(tn)
            if not tn.leaf:
                left = tn.left
                right = tn.right
                if left is not None:
                    print '   ', __node_id(tn) + ' -> ' + __node_id(left)
                    if feature_map.has_key(tn.attr_index):
                        print '[label="if (#%s <= %.3f)"]' % (feature_map[tn.attr_index], tn.partition_value)
                    else:
                        print '[label="if (feature No.%d <= %.3f)"]' % (tn.attr_index, tn.partition_value)

                    __dump_for_dot(left, feature_map)
                if right is not None:
                    print '   ', __node_id(tn) + ' -> ' + __node_id(right) + '[label="else"]'
                    __dump_for_dot(right, feature_map)

        print 'digraph G {'
        # 为了输出中文,同时选择了字体
        print '''node [ fontname="FangSong" ];'''
        print '''edge [ fontname="FangSong" ];'''
        __dump_for_dot(self, feature_map)
        print '}'


    def sample_attribute(self, attr_index, max_attr_try_time):
        # sample values of a attribute(by attr_index),
        # these values are used to split the feature space.
        #
        # We only choose the first n feature values.
        # We will try these values to divide the feature space
        # and calculate information gain,
        # finally choose the largest one and its partition value to return.
        try_values = {}
        for i in range(0, min(self.sample.n, max_attr_try_time)):
            value = self.sample.X[i][attr_index]
            try_values[value] = 1
        try_values = try_values.keys()
        try_values.sort()
        return try_values


    def predict(self, x):
        node = self
        while 1:
            if node.leaf is True:
                return node.value
            if x[node.attr_index] <= node.partition_value:
                node = node.left
            else:
                node = node.right

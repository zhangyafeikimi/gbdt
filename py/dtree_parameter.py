#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# decision tree parameter
#
# author: yafei(zhangyafeikimi@gmail.com)
#


class DTreeParameter:
    def __init__(self):
        self.max_level = 5
        self.split_threshold = 0.8# only for DTreeGain
        self.max_attr_try_time = 1000
        self.min_samples_in_leave = 3# only for DTreeLoss and GBDT
        self.tree_number = 800# only for GBDT
        self.learning_rate = 0.5# only for GBDT

#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# 预测房地产价格
#
# author: yafei(zhangyafeikimi@gmail.com)
#
import sys
import codecs
import locale
from dtree_loss import DTreeLoss
from dtree_parameter import DTreeParameter
from dtree_sample import DTreeSample

if __name__ == '__main__':
    param = DTreeParameter()
    sample = DTreeSample()
    sample.load('real-estate.txt')
    dt = DTreeLoss(sample, param)
    dt.train(None)

    feature_map = {
            0: u'结构',
            1: u'装修',
            2: u'周边',
            3: u'地段',
            4: u'绿化',
            5: u'交通',
            6: u'户均车位',
            }
    # 为了输出中文
    locale.setlocale(locale.LC_ALL, '')
    lang, encoding = locale.getdefaultlocale()
    sys.stdout = codecs.getwriter(encoding)(sys.stdout)
    # python demo-real-estate.py | dot -Tpdf > real-estate.pdf
    dt.dump_for_dot(feature_map)

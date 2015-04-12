#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# 预测weibo粉丝是否是僵尸粉
#
# author: yafei(zhangyafeikimi@gmail.com)
#
import sys
import codecs
import locale
from dtree_gain import DTreeGain
from dtree_parameter import DTreeParameter
from dtree_sample import DTreeSample

if __name__ == '__main__':
    param = DTreeParameter()
    param.split_threshold = 0.93
    sample = DTreeSample()
    sample.load('weibo.txt')
    dt = DTreeGain(sample, param)
    dt.train()

    feature_map = {
            0: u'注册天数',
            1: u'加V',
            2: u'关注',
            3: u'粉丝',
            4: u'微博',
            5: u'收藏',
            6: u'互粉',
            7: u'共同好友',
            8: u'tag数',
            9: u'等级',
            }
    # 为了输出中文
    locale.setlocale(locale.LC_ALL, '')
    lang, encoding = locale.getdefaultlocale()
    sys.stdout = codecs.getwriter(encoding)(sys.stdout)
    # python demo-weibo.py | dot -Tpdf > weibo.pdf
    dt.dump_for_dot(feature_map)

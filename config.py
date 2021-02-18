#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Bingyu Jiang, Peixin Lin
LastEditors: Peixin Lin
Date: 2020-08-21 16:20:49
LastEditTime: 2020-08-27 19:37:41
FilePath: /Assignment3-1_solution/config.py
Desciption: 配置文件。
Copyright: 北京贪心科技有限公司版权所有。仅供教学目的使用。
'''

import torch
import os
root_path = os.path.abspath(os.path.dirname(__file__))

train_raw = os.path.join(root_path, 'data/chat.txt')
dev_raw = os.path.join(root_path, 'data/开发集.txt')
test_raw = os.path.join(root_path, 'data/测试集.txt')
ware_path = os.path.join(root_path, 'data/ware.txt')
max_sequence_length=100

sep = '[SEP]'

''' Data '''
# main
train_path = os.path.join(root_path, 'data/train_no_blank.csv')
dev_path = os.path.join(root_path, 'data/dev.csv')
test_path = os.path.join(root_path, 'data/test.csv')
# intention
business_train = os.path.join(root_path, 'data/intention/business.train')
business_test = os.path.join(root_path, 'data/intention/business.test')
keyword_path = os.path.join(root_path, 'data/intention/key_word.txt')


''' Intention '''
# fasttext
ft_path = os.path.join(root_path, "model/intention/fastext")

''' Retrival '''
# Embedding
w2v_path = os.path.join(root_path, "model/generative/word2vec")

# HNSW parameters
ef_construction = 3000  # ef_construction defines a construction time/accuracy trade-off
M = 64  # M defines tha maximum number of outgoing connections in the graph
hnsw_path = os.path.join(root_path, 'model/generative/hnsw_index')

# 通用配置
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')






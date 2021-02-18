#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Bingyu Jiang, Peixin Lin
LastEditors: Peixin Lin
Date: 2020-08-10 15:53:39
LastEditTime: 2020-08-27 19:37:33
FilePath: /Assignment3-1_solution/generative/word2vec.py
Desciption: 训练word2vec model。
Copyright: 北京贪心科技有限公司版权所有。仅供教学目的使用。
'''

import logging
import multiprocessing
import sys
from time import time
import os

import pandas as pd
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser, Phrases

sys.path.append('..')
from config import root_path, train_raw
from preprocessor import clean, read_file
import config


logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                    datefmt="%H:%M:%S",
                    level=logging.INFO)


def read_data(file_path):
    '''
    @description: 读取数据，清洗
    @param {type}
    file_path: 文件所在路径
    @return: Training samples.
    '''
    train = pd.DataFrame(read_file(file_path, True),
                         columns=['session_id', 'role', 'content'])
    train['clean_content'] = train['content'].apply(clean)
    return train


def train_w2v(train, to_file):
    '''
    @description: 训练word2vec model， 并保存
    @param {type}
    train: 数据集 DataFrame
    to_file: 模型保存路径
    @return: None
    '''
    sent = [row.split() for row in train['clean_content']]
    phrases = Phrases(sent, min_count=5, progress_per=10000)
    bigram = Phraser(phrases)
    sentences = bigram[sent]

    cores = multiprocessing.cpu_count()
    w2v_model = Word2Vec(min_count=2,
                         window=2,
                         size=300,
                         sample=6e-5,
                         alpha=0.03,
                         min_alpha=0.0007,
                         negative=15,
                         workers=cores - 1,
                         iter=7)

    t = time()
    w2v_model.build_vocab(sentences)
    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

    t = time()
    w2v_model.train(sentences,
                    total_examples=w2v_model.corpus_count,
                    epochs=15,
                    report_delay=1)
    print('Time to train vocab: {} mins'.format(round((time() - t) / 60, 2)))

    w2v_model.save(to_file)


if __name__ == "__main__":
    train = read_data(config.train_raw)
    train_w2v(train, config.w2v_path)

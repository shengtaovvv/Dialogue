#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Bingyu Jiang, Peixin Lin
LastEditors: Peixin Lin
Date: 2020-08-10 15:53:39
LastEditTime: 2020-08-27 19:37:19
FilePath: /Assignment3-1_solution/generative/hnsw_hnswlib.py
Desciption: 使用hnswlib训练hnsw模型。
Copyright: 北京贪心科技有限公司版权所有。仅供教学目的使用。
'''
import logging
import sys
import os

sys.path.append('..')

import hnswlib
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from retrieval.preprocessor import clean


logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                    datefmt="%H:%M:%S",
                    level=logging.INFO)


def wam(sentence, w2v_model):
    '''
    @description: 通过word average model 生成句向量
    @param {type}
    sentence: 以空格分割的句子
    w2v_model: word2vec模型
    @return: Sentence embeded vector.
    '''
    arr = []
    for s in clean(sentence).split():
        if s not in w2v_model.wv.vocab.keys():
            arr.append(np.random.randn(1, 300))
        else:
            arr.append(w2v_model.wv.get_vector(s))
    return np.mean(np.array(arr), axis=0).reshape(1, -1)


class HNSW(object):
    def __init__(self,
                 w2v_path,
                 data_path=None,
                 ef=ef_construction,
                 M=M,
                 model_path=None):
        self.w2v_model = KeyedVectors.load(w2v_path)

        self.data = self.data_load(data_path)
        if model_path is not None:
            # 加载
            self.hnsw = self.load_hnsw(model_path)
        else:
            # 训练
            self.hnsw = \
                self.build_hnsw(os.path.join(root_path, 'model/hnsw.bin'),
                                ef=ef,
                                m=M)

    def data_load(self, data_path):
        '''
        @description: 读取数据，并生成句向量
        @param {type}
        data_path：问答pair数据所在路径
        @return: 包含句向量的dataframe
        '''
        data = pd.read_csv(
            data_path)
        data['custom_vec'] = data['custom'].apply(
            lambda x: wam(x, self.w2v_model))
        data['custom_vec'] = data['custom_vec'].apply(
            lambda x: x[0][0] if x.shape[1] != 300 else x)
        data = data.dropna()
        return data

    def build_hnsw(self, to_file, ef=2000, m=64):
        '''
        @description: 训练hnsw模型
        @param {type}
        to_file： 模型保存目录
        ef/m 等参数 参考https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
        @return:
        '''
        dim = self.w2v_model.vector_size
        num_elements = self.data['custom'].shape[0]
        hnsw = np.stack(self.data['custom_vec'].values).reshape(-1, 300)

        # Declaring index
        p = hnswlib.Index(space='l2',
                          dim=dim)  # possible options are l2, cosine or ip
        p.init_index(max_elements=num_elements, ef_construction=ef, M=m)
        p.set_ef(10)
        p.set_num_threads(4)
        p.add_items(hnsw)
        logging.info('Start')
        labels, distances = p.knn_query(hnsw, k=1)
        print('labels: ', labels)
        print('distances: ', distances)
        logging.info("Recall:{}".format(
            np.mean(labels.reshape(-1) == np.arange(len(hnsw)))))
        p.save_index(to_file)
        return p

    def load_hnsw(self, model_path):
        '''
        @description: 加载训练好的hnsw模型
        @param {type}
        model_path： 模型保存的目录
        @return: hnsw 模型
        '''
        hnsw = hnswlib.Index(space='l2', dim=self.w2v_model.vector_size)
        hnsw.load_index(model_path)
        return hnsw

    def search(self, text, k=5):
        '''
        @description: 通过hnsw 检索
        @param {type}
        text: 检索句子
        k: 检索返回的数量
        @return:
        '''
        test_vec = wam(clean(text), self.w2v_model)
        q_labels, q_distances = self.hnsw.knn_query(test_vec, k=k)
        return pd.concat(
            (self.data.iloc[q_labels[0]]['custom'].reset_index(),
             self.data.iloc[q_labels[0]]['assistance'].reset_index(drop=True),
             pd.DataFrame(q_distances.reshape(-1, 1), columns=['q_distance'])),
            axis=1)


if __name__ == "__main__":
    hnsw = HNSW(config.w2v_path),
                config.train_path, config.ef_construction, config.M)
    test = '我要转人工'
    print(hnsw.search(test, k=10))

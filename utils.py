#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import json
import random

import numpy as np


def read_data(filename):
    with open(filename, encoding="utf-8") as f:
        data = f.read()
        data = data.split()
        data = '\n'.join(data)
    # data = list(data)
    return data


def index_data(sentences, dictionary):
    shape = sentences.shape
    sentences = sentences.reshape([-1])
    index = np.zeros_like(sentences, dtype=np.int32)
    for i in range(len(sentences)):
        try:
            index[i] = dictionary[sentences[i]]
        except KeyError:
            index[i] = dictionary['UNK']

    return index.reshape(shape)


def get_train_data(vocabulary, batch_size, num_steps):
    ##################
    # Your Code here
    ##################
    train_batches = []
    vocabulary_size = 5000
    data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, vocabulary_size)
    del vocabulary
    '''
        data为文本中一段随机截取的文字，label为data对应的下一个标号的文字。
        以苏轼的江神子（江城子）为例：输入为 “老夫聊发少年”，则对应的label为"夫聊发少年狂"
    '''
    raw_x = data
    raw_y = data[1:] + ['\n']
    data_length = len(raw_x)

    # partition raw data into batches and stack them vertically in a data matrix
    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    for i in range(batch_size):
        data_x[i] = raw_x[batch_partition_length * i: batch_partition_length * (i + 1)]
        data_y[i] = raw_y[batch_partition_length * i: batch_partition_length * (i + 1)]

    # further divide batch partitions into num_steps for truncated backprop
    epoch_size = batch_partition_length // num_steps

    for i in range(epoch_size):
        x = data_x[:, i * num_steps: (i + 1) * num_steps]
        y = data_y[:, i * num_steps: (i + 1) * num_steps]
        yield (x, y)


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    # # 保存dictionary和reversed_dictionary
    # with open('dictionary.json', 'w') as output_dictionary:
    #     json.dump(dictionary, output_dictionary)
    #
    # with open('reverse_dictionary.json', 'w') as output_reversed_dict:
    #     json.dump(reversed_dictionary, output_reversed_dict)
    return data, count, dictionary, reversed_dictionary


if __name__=='__main__':
    filename = 'test.txt'
    words = read_data(filename)
    # batch, labels = get_train_data(vocabulary=words, batch_size=3, num_steps=32)
    print(len(words))
    # print(labels)
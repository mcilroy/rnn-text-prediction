from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import tensorflow as tf
import os
#import six.moves.cPickle as pickle


def to_categorical(y, nb_classes):
    """ to_categorical.
    Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.
    Arguments:
        y: `array`. Class vector to convert.
        nb_classes: `int`. Total number of classes.
    """
    y = np.asarray(y, dtype='int32')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y


def raw_inputs(data_location=".", n_words=100000):
    # f = open(os.path.join(data_location, "imdb.pkl"), 'rb')
    # train_set = pickle.load(f)
    # f.close()
    # class DataSets(object):
    #     pass
    # data_sets = DataSets()
    with open(os.path.join(data_location, "train-data.txt")) as f:
        train_data = f.readlines()
    with open(os.path.join(data_location, "train-labels.txt")) as f:
        train_labels = f.read().splitlines()
    with open(os.path.join(data_location, "val-data.txt")) as f:
        val_data = f.readlines()
    with open(os.path.join(data_location, "val-labels.txt")) as f:
        val_labels = f.read().splitlines()
    with open(os.path.join(data_location, "test-data.txt")) as f:
        test_data = f.readlines()
    with open(os.path.join(data_location, "test-labels.txt")) as f:
        test_labels = f.read().splitlines()

    # vectorize raw text data
    train_data, _ = vectorize_raw_inputs(train_data)
    val_data, _ = vectorize_raw_inputs(val_data)
    test_data, _ = vectorize_raw_inputs(test_data)

    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]

    # remove words if over limit of max words
    train_data = remove_unk(train_data)
    val_data = remove_unk(val_data)
    test_data = remove_unk(test_data)
    train_labels = train_labels
    val_labels = val_labels
    test_labels = test_labels
    return train_data, train_labels, val_data, val_labels, test_data, test_labels


def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", " ").split()


def _build_vocab(data):
    counter = collections.Counter([item for sublist in data for item in sublist])
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def _file_to_word_ids(data, word_to_id):
    word_to_ids = []
    for text in data:
        word_to_ids.append([word_to_id[word] for word in text])
    return word_to_ids


def vectorize_raw_inputs(data):
    data = [text.replace("\n", " ").split() for text in data]
    word_to_id = _build_vocab(data)
    data = _file_to_word_ids(data, word_to_id)
    vocabulary = len(word_to_id)
    return data, vocabulary

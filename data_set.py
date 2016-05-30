from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np


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


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre',
                  truncating='pre', value=0.):
    """ pad_sequences.
    Pad each sequence to the same length: the length of the longest sequence.
    If maxlen is provided, any sequence longer than maxlen is truncated to
    maxlen. Truncation happens off either the beginning (default) or the
    end of the sequence. Supports post-padding and pre-padding (default).
    Arguments:
        sequences: list of lists where each element is a sequence.
        maxlen: int, maximum length.
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.
    Returns:
        x: `numpy array` with dimensions (number_of_sequences, maxlen)
    Credits: From Keras `pad_sequences` function.
    """
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x


def create_datasets(train_data, train_labels, val_data, val_labels, test_data, test_labels):
    class DataSets(object):
        pass
    data_sets = DataSets()
    data_sets.train = DataSet(train_data, train_labels)
    data_sets.validation = DataSet(val_data, val_labels)
    data_sets.testing = DataSet(test_data, test_labels)
    return data_sets


class DataSet(object):
    def __init__(self, x, labels):
        self._x = x
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self.num_examples = len(self._x)

        # Shuffle the data
        perm = np.arange(self.num_examples)
        np.random.shuffle(perm)
        self._x = self._x[perm]
        self._labels = self._labels[perm]

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            self._x = self._x[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self.num_examples
        end = self._index_in_epoch
        return self._x[start:end], self._labels[start:end]

    @property
    def x(self):
        return self._x

    @property
    def labels(self):
        return self._labels

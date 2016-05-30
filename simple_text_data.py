import simple_reader
import data_set


def inputs(data_path=".", vocab_size=10000):
    train_data, train_labels, val_data, val_labels, test_data, test_labels = simple_reader.raw_inputs(data_path,
                                                                                                      vocab_size)
    train_data = data_set.pad_sequences(train_data, maxlen=3, value=0.)
    train_labels = data_set.to_categorical(train_labels, 2)
    val_data = data_set.pad_sequences(val_data, maxlen=3, value=0.)
    val_labels = data_set.to_categorical(val_labels, 2)
    test_data = data_set.pad_sequences(test_data, maxlen=3, value=0.)
    test_labels = data_set.to_categorical(test_labels, 2)
    return data_set.create_datasets(train_data, train_labels, val_data, val_labels, test_data, test_labels)
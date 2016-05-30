import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell


def RNN(_X, keep_prob, _istate, config):
    # Define weights
    _weights = {
        'hidden': tf.Variable(tf.random_normal([config.n_input, config.n_hidden])),  # Hidden layer weights
        'out': tf.Variable(tf.random_normal([config.n_hidden, config.n_classes]))
    }
    _biases = {
        'hidden': tf.Variable(tf.random_normal([config.n_hidden])),
        'out': tf.Variable(tf.random_normal([config.n_classes]))
    }

    #with tf.device("/cpu:0"):
    embedding = tf.get_variable("embedding", [config.vocab_size, config.n_hidden])
    inputs = tf.nn.embedding_lookup(embedding, _X)

    # # input shape: (batch_size, config.n_steps, config.n_input)
    # _X = tf.transpose(_X, [1, 0, 2])  # permute config.n_steps and batch_size
    # # Reshape to prepare input to hidden activation
    # _X = tf.reshape(_X, [-1, config.n_input])  # (config.n_steps*batch_size, config.n_input)
    # # Linear activation
    # _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(config.n_hidden, forget_bias=1.0)
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)

    inputs = tf.nn.dropout(inputs, keep_prob)

    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    #_X = tf.split(0, config.n_steps, _X)  # config.n_steps * (batch_size, config.n_hidden)
    inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, config.n_steps, inputs)]
    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, inputs, initial_state=_istate)

    # Linear activation
    # Get inner loop last output
    return tf.matmul(outputs[-1], _weights['out']) + _biases['out']
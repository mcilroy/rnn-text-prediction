"""
:LSTM model for predicting class of text samples
We will then handle 1 sequences (word) of 3 steps for every sample.
"""
import simple_text_data
import tensorflow as tf
import numpy as np
import simple_text_model
from configs import SimpleConfig
flags = tf.flags
logging = tf.logging

flags.DEFINE_string("model", "simple", "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", "data", "data_path")
FLAGS = flags.FLAGS


def get_config():
    if FLAGS.model == "simple":
        return SimpleConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)


def train():
    config = get_config()
    # Import data
    text_datasets = simple_text_data.inputs(FLAGS.data_path, config.vocab_size)

    # placeholders
    x = tf.placeholder(tf.int32, [config.batch_size, config.n_steps])
    istate = tf.placeholder("float", [config.batch_size, 2*config.n_hidden])  # Tensorflow LSTM cell requires 2x n_hidden length (state & cell)
    y = tf.placeholder(tf.float32, [config.batch_size, config.n_classes])
    keep_prob = tf.placeholder(tf.float32)

    # model
    pred = simple_text_model.RNN(x, keep_prob, istate, config)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))  # Softmax loss
    optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(cost)  # Adam Optimizer

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.initialize_all_variables()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step * config.batch_size < config.training_iters:
            batch_xs, batch_ys = text_datasets.train.next_batch(config.batch_size)
            # Reshape data to get 3 sequences of 1 element each
            #batch_xs = batch_xs.reshape((config.batch_size, config.n_steps, config.n_input))
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: config.keep_prob,
                                           istate: np.zeros((config.batch_size, 2*config.n_hidden))})
            if step % config.display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.,
                                                    istate: np.zeros((config.batch_size, 2*config.n_hidden))})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.,
                                                 istate: np.zeros((config.batch_size, 2*config.n_hidden))})
                print "Iter " + str(step*config.batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + \
                      ", Training Accuracy= " + "{:.5f}".format(acc)
            step += 1
        print "Optimization Finished!"
        # calculate test accuracy
        test_len = config.batch_size
        test_data = text_datasets.testing.x[:test_len]
        #test_data = text_datasets.testing.x[:test_len].reshape((-1, config.n_steps, config.n_input))
        test_label = text_datasets.testing.labels[:test_len]
        print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label, keep_prob: 1.,
                                                                 istate: np.zeros((test_len, 2*config.n_hidden))})


def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()

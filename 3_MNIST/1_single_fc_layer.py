import os
import inspect
import shutil
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

tf.set_random_seed(0)

# Calculate LOG_DIR according to current file
CUR_FILE = inspect.getfile(inspect.currentframe())
LOG_DIR = os.path.join(
os.path.dirname(os.path.abspath(CUR_FILE)), 'logs',
os.path.splitext(os.path.basename(CUR_FILE))[0])

LOG_DIR_TRAIN = os.path.join(LOG_DIR, 'train')
LOG_DIR_VALID = os.path.join(LOG_DIR, 'valid')
LOG_DIR_TEST = os.path.join(LOG_DIR, 'test')

# Check is the LOG_DIR empty. If not ask for clean.
def clean_logs(logdir, ask=True):
    if logdir == None or len(logdir) < 4:
        return
    if os.path.exists(logdir) and len(os.listdir(logdir)) > 0:
        if ask:
            answer = input('Log Folder: ' + logdir + ' is not empty. Clean it? [y/N]')
            if answer in ['Y', 'y']:
                shutil.rmtree(logdir)
        else:
            shutil.rmtree(logdir)

clean_logs(LOG_DIR, False)

print('TensorFlow Version: ' + tf.__version__)

######################################################################
# Main

_DEBUG_GRAPH = False

MNIST_WIDTH = 28
MNIST_HEIGHT = 28
MNIST_CHANNEL = 1
NUM_CLASSES = 10

BATCH_SIZE = 100
EPOCH_NUM = 20
LEARNING_RATE = 5e-3

TRAIN_SIZE = 55000
VALID_SIZE = 5000
TEST_SIZE = 10000

######################################################################
# The Model
#
# · · · · · · · · · ·       (input data, flattened pixels)       X [batch, 784]        # 784 = 28 * 28
# \x/x\x/x\x/x\x/x\x/    -- fully connected layer (softmax)      W [784, 10]     b[10]
#   · · · · · · · ·                                              Y [batch, 10]
#
######################################################################

mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False)

with tf.name_scope('Input'):
    x = tf.placeholder(tf.float32, [None, MNIST_HEIGHT, MNIST_WIDTH, MNIST_CHANNEL], name='X')
    y = tf.placeholder(tf.float32, [None, NUM_CLASSES], name='Y')
    x_flatten = tf.reshape(x, [-1, MNIST_HEIGHT*MNIST_WIDTH], name='X_Flatten')

with tf.name_scope('Output'):
    W = tf.Variable(tf.zeros([MNIST_HEIGHT*MNIST_WIDTH, NUM_CLASSES]), name='Weights')
    b = tf.Variable(tf.zeros([NUM_CLASSES]), name='Biases')
    logits = tf.matmul(x_flatten, W) + b
    pred = tf.nn.softmax(logits)

    tf.summary.histogram('weight', W)
    tf.summary.histogram('bias', b)
    tf.summary.histogram('pred', pred)

with tf.name_scope('Loss'):
    # loss function: cross-entropy = - sum( Y_true * log(Y_pred) )
    xent = -tf.reduce_mean(y * tf.log(pred), name='xent') * 10 * BATCH_SIZE

    tf.summary.scalar('xent', xent)

with tf.name_scope('Accuracy'):
    correct_pred = tf.equal(tf.argmax(pred, axis=1), tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    tf.summary.scalar('accuracy', accuracy)

with tf.name_scope('Optimizer'):
    optimize = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(xent)

with tf.name_scope('Status'):
    global_step = tf.Variable(tf.constant(0), 'step')

with tf.name_scope('Summaries'):
    all_summaries = tf.summary.merge_all()

with tf.name_scope('Global_Ops'):
    init = tf.global_variables_initializer()
    inc_step = global_step.assign_add(1)

if _DEBUG_GRAPH:
    writer = tf.summary.FileWriter(LOG_DIR, tf.get_default_graph())
    writer.flush()
    writer.close()
else:
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(LOG_DIR_TRAIN, sess.graph)
        valid_writer = tf.summary.FileWriter(LOG_DIR_VALID, sess.graph)
        test_writer = tf.summary.FileWriter(LOG_DIR_TEST, sess.graph)
        sess.run(init)

        best_acc = 0
        for epoch in range(EPOCH_NUM + 1):
            for i in range(TRAIN_SIZE//BATCH_SIZE + 1):
                # Train
                x_batch, y_batch = mnist.train.next_batch(BATCH_SIZE)
                step_,  _ = sess.run([inc_step, optimize], feed_dict = {x: x_batch, y: y_batch})

                if step_ % 10 == 0 or step_ == 1:
                    acc_, xent_, summ_ = sess.run([accuracy, xent, all_summaries], feed_dict = {x: x_batch, y: y_batch})
                    print("Train Accuracy: {}, Train loss: {}, step: {}, epoch: {}, iter: {}".format(acc_, xent_, step_, epoch, i))
                    train_writer.add_summary(summ_, global_step=step_)

                if step_ % 100 == 0 or step_ == 1:
                    acc_, xent_, summ_ = sess.run([accuracy, xent, all_summaries], feed_dict = {x: mnist.validation.images, y: mnist.validation.labels})
                    print("Validation Accuracy: {}, Validation loss: {}, step: {}, epoch: {}, iter: {}".format(acc_, xent_, step_, epoch, i))
                    valid_writer.add_summary(summ_, global_step=step_)

                    acc_, xent_, summ_ = sess.run([accuracy, xent, all_summaries], feed_dict = {x: mnist.test.images, y: mnist.test.labels})
                    test_writer.add_summary(summ_, global_step=step_)
                    if acc_ > best_acc:
                        best_acc = acc_
                    print("******** Epoch: {} ********: Test Accuracy: {}, Test Loss: {}".format(epoch, acc_, xent_))

        print("All Done. Best accuracy: {}".format(best_acc))
        train_writer.flush()
        train_writer.close()
        test_writer.flush()
        test_writer.close()
        valid_writer.flush()
        valid_writer.close()


# End of Main
######################################################################

# Automatically show TensorBoard if needed.
def show_tensorboard(auto_show=False):
    cmd = 'tensorboard --logdir=train:' + LOG_DIR_TRAIN + ',validation:' + LOG_DIR_VALID + ',test:' + LOG_DIR_TEST
    if auto_show:
        answer = input('Show tensorboard? [Y/n]')
        if not answer in ['N', 'n']:
            os.system(cmd)
    else:
        print("\nRun this command to see logs:\n" + cmd)
show_tensorboard(True)

import os
import inspect
import shutil
import math
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
EPOCH_NUM = 10
LEARNING_RATE = 5e-3

ROOT_FEATURES = 4
FILTER_SIZE = 4

TRAIN_SIZE = 55000
VALID_SIZE = 5000
TEST_SIZE = 10000

######################################################################
# The Model
# neural network with 5 layers
#
# · · · · · · · · · ·      (input data, 1-deep)                 X [batch, 28, 28, 1]
# @ @ @ @ @ @ @ @ @ @   -- conv. layer 5x5x1=>4 stride 1        W1 [4, 4, 1, 4]        B1 [4]
# ∶ ∶ ∶ ∶ ∶ ∶ ∶ ∶ ∶ ∶ ∶ ∶ ∶                                           Y1 [batch, 28, 28, 4]
#   @ @ @ @ @ @ @ @     -- conv. layer 5x5x4=>8 stride 1        W2 [4, 4, 4, 8]        B2 [8]
#   ∶ ∶ ∶ ∶ ∶ ∶ ∶ ∶ ∶ ∶                                             Y2 [batch, 14, 14, 8]
#     @ @ @ @ @ @       -- conv. layer 4x4x8=>12 stride 1       W3 [4, 4, 8, 12]       B3 [12]
#     ∶ ∶ ∶ ∶ ∶ ∶ ∶ ∶                                               Y3 [batch, 7, 7, 12] => reshaped to YY [batch, 7*7*12]
#      \x/x\x\x/        -- fully connected layer (relu)         W4 [7*7*12, 200]       B4 [200]
#       · · · ·                                                 Y4 [batch, 200]
#       \x/x\x/         -- fully connected layer (softmax)      W5 [200, 10]           B5 [10]
#        · · ·                                                  Y [batch, 10]
#
######################################################################

mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False)

def create_conv_layer(inputs, size_in, size_out, has_pool=True, keep_prob=0.75, stddev=0.1, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([FILTER_SIZE, FILTER_SIZE, size_in, size_out], stddev=stddev), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        conv = tf.nn.conv2d(inputs, w, strides=[1, 1, 1, 1], padding="SAME", name="conv")
        act = tf.nn.relu(conv + b)
        drop = tf.nn.dropout(act, keep_prob)
        ret = drop
        if has_pool:
            pool = tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool")
            ret = pool

        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        return ret

def create_fc_layer(inputs, size_in, size_out, keep_prob=0.75, stddev=0.1, name='fc'):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=stddev), name='Weight')
        b = tf.Variable(tf.ones([size_out])/10, name='Bias')
        act = tf.nn.relu(tf.matmul(inputs, W) + b)
        drop = tf.nn.dropout(act, keep_prob)

        tf.summary.histogram('Weight', W)
        tf.summary.histogram('Bias', b)

        return drop

with tf.name_scope('Input'):
    x = tf.placeholder(tf.float32, [None, MNIST_HEIGHT, MNIST_WIDTH, MNIST_CHANNEL], name='X')
    y = tf.placeholder(tf.float32, [None, NUM_CLASSES], name='Y')

last_output = x

features = MNIST_CHANNEL
for conv_layer in range(3):
    name = "conv_layer_{}".format(conv_layer)
    features = 2**conv_layer * ROOT_FEATURES
    stddev = np.sqrt(2/(FILTER_SIZE**2*features))
    channel_in = features//2
    has_pool = True
    if conv_layer == 0:
        channel_in = MNIST_CHANNEL
        has_pool = False

    last_output = create_conv_layer(last_output, channel_in, features, has_pool=has_pool, stddev=stddev, name=name)

conv_out_size = MNIST_HEIGHT//2//2*MNIST_WIDTH//2//2*features
last_output = tf.reshape(last_output, [-1, conv_out_size])

layer_size = [conv_out_size, 200, 30]
for fc_layer in range(2):
    name = "fc_layer_{}".format(fc_layer)
    last_output = create_fc_layer(last_output, layer_size[fc_layer], layer_size[fc_layer+1], name=name)

with tf.name_scope('Output'):
    W = tf.Variable(tf.truncated_normal([layer_size[-1], NUM_CLASSES], stddev=0.1), name='Weights')
    b = tf.Variable(tf.zeros([NUM_CLASSES]), name='Biases')
    logits = tf.matmul(last_output, W) + b
    y_pred = tf.argmax(logits)

    tf.summary.histogram('weight', W)
    tf.summary.histogram('bias', b)
    tf.summary.histogram('logits', logits)

with tf.name_scope('Loss'):
    # loss function: cross-entropy = - sum( Y_true * log(Y_pred) )
    xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    xent = tf.reduce_mean(xent) * BATCH_SIZE

    tf.summary.scalar('xent', xent)

with tf.name_scope('Accuracy'):
    correct_pred = tf.equal(tf.argmax(logits, axis=1), tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    tf.summary.scalar('accuracy', accuracy)

with tf.name_scope('Optimizer'):
    lr = tf.placeholder(tf.float32, name='lr')
    optimize = tf.train.AdamOptimizer(lr).minimize(xent)

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
        # valid_writer = tf.summary.FileWriter(LOG_DIR_VALID, sess.graph)
        test_writer = tf.summary.FileWriter(LOG_DIR_TEST, sess.graph)
        sess.run(init)

        best_acc = 0
        for epoch in range(EPOCH_NUM + 1):
            for i in range(TRAIN_SIZE//BATCH_SIZE + 1):
                max_lr = 3e-3
                min_lr = 1e-4
                declay_speed = 2000.
                learning_rate = min_lr + (max_lr - min_lr) * math.exp(-i/declay_speed)
                # Train
                x_batch, y_batch = mnist.train.next_batch(BATCH_SIZE)
                step_,  _ = sess.run([inc_step, optimize], feed_dict = {x: x_batch, y: y_batch, lr: learning_rate})

                if step_ % 10 == 0 or step_ == 1:
                    acc_, xent_, summ_ = sess.run([accuracy, xent, all_summaries], feed_dict = {x: x_batch, y: y_batch})
                    print("Train Accuracy: {}, Train loss: {}, step: {}, epoch: {}, iter: {}".format(acc_, xent_, step_, epoch, i))
                    train_writer.add_summary(summ_, global_step=step_)

                if step_ % 500 == 0 or step_ == 1:
                    # acc_, xent_, summ_ = sess.run([accuracy, xent, all_summaries], feed_dict = {x: mnist.validation.images, y: mnist.validation.labels})
                    # print("Validation Accuracy: {}, Validation loss: {}, step: {}, epoch: {}, iter: {}".format(acc_, xent_, step_, epoch, i))
                    # valid_writer.add_summary(summ_, global_step=step_)

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

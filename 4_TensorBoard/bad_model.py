import os
import inspect
import shutil
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

# Calculate LOG_DIR according to current file
CUR_FILE = inspect.getfile(inspect.currentframe())
LOG_DIR = os.path.join(
os.path.dirname(os.path.abspath(CUR_FILE)), 'logs',
os.path.splitext(os.path.basename(CUR_FILE))[0])

# Check is the LOG_DIR empty. If not ask for clean.
def clean_logs(logdir):
    if logdir == None or len(logdir) < 4:
        return
    if os.path.exists(logdir) and len(os.listdir(logdir)) > 0:
        answer = input('Log Folder: ' + logdir + ' is not empty. Clean it? [y/N]')
        if answer in ['Y', 'y']:
            shutil.rmtree(logdir)
clean_logs(LOG_DIR)

print('TensorFlow Version: ' + tf.__version__)

mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False)

def conv_layer(inputs, channels_in, channels_out):
    w = tf.Variable(tf.zeros([5, 5, channels_in, channels_out]))
    b = tf.Variable(tf.zeros([channels_out]))
    conv = tf.nn.conv2d(inputs, w, strides=[1, 1, 1, 1], padding="SAME")
    act = tf.nn.relu(conv + b)
    return act

def fc_layer(inputs, channels_in, channels_out):
    w = tf.Variable(tf.zeros([channels_in, channels_out]))
    b = tf.Variable(tf.zeros([channels_out]))
    act = tf.nn.relu(tf.matmul(inputs, w) + b)
    return act

x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
y = tf.placeholder(tf.float32, shape=[None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

conv1 = conv_layer(x_image, 1, 4)
pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

conv2 = conv_layer(pool1, 4, 8)
pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
flattened = tf.reshape(pool2, [-1, 7*7*8])

fc1 = fc_layer(flattened, 7*7*8, 200)
logits = fc_layer(fc1, 200, 10)

xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

train_step = tf.train.AdamOptimizer(1e-4).minimize(xent)

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

_DEBUG_GRAPH = True
if _DEBUG_GRAPH:
    writer = tf.summary.FileWriter(LOG_DIR, tf.get_default_graph())
    writer.flush()
    writer.close()
else:
    for i in range(1000):
        batch = mnist.train.next_batch(100)
        if i % 200 == 0:
            acc_ = sess.run(accuracy, feed_dict={x: batch[0], y:batch[1]})
            print("step {}, training accuracy: {}".format(i, acc_))

        sess.run(train_step, feed_dict={x: batch[0], y:batch[1]})

# Automatically show TensorBoard if needed.
def show_tensorboard(auto_show=False):
    cmd = 'tensorboard --logdir=' + LOG_DIR
    if auto_show:
        answer = input('Show tensorboard? [Y/n]')
        if not answer in ['N', 'n']:
            os.system(cmd)
    else:
        print("\nRun this command to see logs:\n" + cmd)
show_tensorboard(True)

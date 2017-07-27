import os
import inspect
import shutil
import tensorflow as tf
import numpy as np

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

# Create input data using NumPy.
# y = x * 0.3 + 0.5 + noise
data_x = np.random.rand(100).astype(np.float32)
data_noise = np.random.normal(scale=0.01, size=len(data_x))
data_y = data_x * 0.3 + 0.5 + data_noise


# Build TensorFlow learning graph

# Linear model
# y = w * x + b
w = tf.Variable(tf.random_uniform([1], 0.0, 1.0), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")
y = w * data_x + b

# Training graph
# We need an Error function to evaluate learning mistake
loss = tf.reduce_mean(tf.square(y - data_y))
# We need a way to optimize learning result
learning_rate = 0.5
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# We want a minimized loss
train = optimizer.minimize(loss)

# Define summary
with tf.name_scope('summaries'):
    tf.summary.histogram('weight', w)
    tf.summary.histogram('bias', b)
    tf.summary.scalar('loss', loss)

# Merge all summaries for writer later
merged_summary = tf.summary.merge_all()

# Define Initializer for Variables. Important!!!
init = tf.global_variables_initializer()

with tf.Session() as sess:
    writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
    # Execute Initializer before training. Important!!!
    sess.run(init)

    l_, w_, b_ = sess.run([loss, w, b])
    print('step init, loss: {}, w: {}, b: {}'.format(l_, w_, b_))

    for step in range(201):
        _, summary = sess.run([train, merged_summary])
        writer.add_summary(summary, global_step=step)
        if step % 20 == 0:
            l_, w_, b_ = sess.run([loss, w, b])
            print('step {}, loss: {}, w: {}, b: {}'.format(step, l_, w_, b_))

    print("final results: w: {}, b: {}".format(sess.run(w), sess.run(b)))
    writer.close()

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

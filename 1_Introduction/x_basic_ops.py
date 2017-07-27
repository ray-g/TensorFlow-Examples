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

graph = tf.Graph()

with graph.as_default():
    with tf.name_scope('variables'):
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        total_output = tf.Variable(0.0, dtype=tf.float32, trainable=False, name='total_output')

    with tf.name_scope('transformation'):
        with tf.name_scope('input_layer'):
            input_a = tf.placeholder(tf.float32, shape=[None], name='input_ph_a')
        with tf.name_scope('intermediate_layer'):
            b = tf.reduce_prod(input_a, name='prod_b')
            c = tf.reduce_sum(input_a, name='sum_c')
        with tf.name_scope('output_layer'):
            output = tf.add(b, c, name='output')

    with tf.name_scope('update'):
        update_total = total_output.assign_add(output)
        increment_step = global_step.assign_add(1)

    with tf.name_scope('summaries'):
        avg = tf.div(update_total, tf.cast(increment_step, tf.float32), name='average')

        tf.summary.scalar('Output', output)
        tf.summary.scalar('Sum of output over time', update_total)
        tf.summary.scalar('Average of output over time', avg)

    with tf.name_scope('global_ops'):
        init = tf.global_variables_initializer()
        merged_summaries = tf.summary.merge_all()

sess = tf.Session(graph=graph)
writer = tf.summary.FileWriter(LOG_DIR, graph)
sess.run(init)

def run_graph(input_tensor):
    feed_dict = {input_a: input_tensor}
    _, step, summary = sess.run([output, increment_step, merged_summaries], feed_dict=feed_dict)
    writer.add_summary(summary, global_step=step)

feeds = np.random.rand(10, 2)
for item in feeds:
    run_graph(item)

writer.flush()
writer.close()
sess.close()

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

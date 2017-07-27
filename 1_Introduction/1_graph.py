import tensorflow as tf
import os
import inspect
import shutil

CUR_FILE = inspect.getfile(inspect.currentframe())
LOG_DIR = os.path.join(
os.path.dirname(os.path.abspath(CUR_FILE)), 'logs',
os.path.splitext(os.path.basename(CUR_FILE))[0])

def clean_logs(logdir):
    if logdir == None or len(logdir) < 4:
        return
    if os.path.exists(logdir) and len(os.listdir(logdir)) > 0:
        answer = input('Log Folder: ' + logdir + ' is not empty. Clean it? [y/N]')
        if answer in ['Y', 'y']:
            shutil.rmtree(logdir)
LOG_DIR_DEFAULT = os.path.join(LOG_DIR, 'default_graph')
LOG_DIR_NEW = os.path.join(LOG_DIR, 'new_graph')
clean_logs(LOG_DIR_DEFAULT)
clean_logs(LOG_DIR_NEW)

print('TensorFlow Version: ' + tf.__version__)

# TensorFlow can define multiple graphs at the same time
# But only one graph can be execute in one session
default_graph = tf.get_default_graph()

new_graph = tf.Graph()

default_a = tf.constant(0, name='default_a')
default_b = tf.constant(0, name='default_b')

with default_graph.as_default():
    default_add = tf.add(default_a, default_b, name='default_add')

with new_graph.as_default():
    new_a = tf.constant(0, name='new_a')
    new_b = tf.constant(0, name='new_b')
    new_add = tf.add(new_a, new_b, name='new_add')
    fault_add = tf.add(default_a, default_b, name='fault_add')

default_writer = tf.summary.FileWriter(LOG_DIR_DEFAULT, graph=default_graph)
default_writer.close()

new_writer = tf.summary.FileWriter(LOG_DIR_NEW, graph=new_graph)
new_writer.close()

def show_tensorboard(auto_show=False):
    cmd = 'tensorboard --logdir=default_graph:' + LOG_DIR_DEFAULT + ',new_graph:' + LOG_DIR_NEW
    if auto_show:
        answer = input('Show tensorboard? [Y/n]')
        if not answer in ['N', 'n']:
            os.system(cmd)
    else:
        print("\nRun this command to see logs:\n" + cmd)
show_tensorboard(True)

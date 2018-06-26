#! /usr/bin/env python

import os
import inspect
import shutil
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector

# Calculate LOG_DIR according to current file
CUR_FILE = inspect.getfile(inspect.currentframe())
LOG_DIR = os.path.join(
os.path.dirname(os.path.abspath(CUR_FILE)), 'logs',
os.path.splitext(os.path.basename(CUR_FILE))[0])

USE_10K = True
label_file = os.path.join(os.getcwd(), "labels_1024.tsv")
sprites_file = os.path.join(os.getcwd(), "sprite_1024.png")

if USE_10K:
    label_file = os.path.join(LOG_DIR, 'metadata.tsv')
    sprites_file = os.path.join(os.getcwd(), "mnist_10k_sprite.png")

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


mnist = input_data.read_data_sets('data', False)
images = tf.Variable(mnist.test.images, name='images')

if USE_10K:
    with open(label_file, 'w') as metadata_file:
        for row in mnist.test.labels:
            metadata_file.write('%d\n' % row)

with tf.Session() as sess:
    saver = tf.train.Saver([images])

    sess.run(images.initializer)
    saver.save(sess, os.path.join(LOG_DIR, 'images.ckpt'))

    config = projector.ProjectorConfig()
    # One can add multiple embeddings.
    embedding = config.embeddings.add()
    embedding.tensor_name = images.name
    # Link this tensor to its metadata file
    embedding.metadata_path = label_file
    embedding.sprite.image_path = sprites_file
    embedding.sprite.single_image_dim.extend([28, 28])
    # Save the config to file for TensorBoard
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)

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

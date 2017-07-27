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

# Construct the TensorFlow Compute graph

import tensorflow as tf

print('TensorFlow Version: ' + tf.__version__)

# Define a default Graph
a = tf.constant(3)
b = tf.constant(5)
c = tf.add(a, b)

# Define a customized Graph
graph = tf.Graph()
with graph.as_default():
    aa = tf.constant(33)
    bb = tf.constant(55)
    cc = tf.add(aa, bb)

# With default session, will use default graph
sess = tf.Session()
print('No param in tf.Session(), use default graph' + sess.run(c))

# After use the session, the session must be closed.
sess.close()

# To use non-default graph, need to specify graph parameter
with tf.Session(graph=graph) as sess2:
    # In this case, not necessary to close the session explicity
    print(sess2.run(cc))

import tensorflow as tf

print('TensorFlow Version: ' + tf.__version__)

# Define a default Graph
a = tf.constant(3)
b = tf.constant(5)
c = tf.add(a, b)

# Define a customized Graph
graph1 = tf.Graph()
with graph1.as_default():
    aa = tf.constant(33)
    bb = tf.constant(55)
    cc = tf.add(aa, bb)

# Define another customized Graph
graph2 = tf.Graph()
with graph2.as_default():
    aaa = tf.constant(333)
    bbb = tf.constant(555)
    ccc = tf.add(aaa, bbb)

# With default session, will use default graph
sess = tf.Session()
print("No param in tf.Session(), use default graph, result: {}".format(sess.run(c)))

sess2 = tf.Session(graph=graph2)
print("Another session use graph2, result: {}".format(sess2.run(ccc)))

# After use the session, the session must be closed.
sess.close()
sess2.close()

# To use non-default graph, need to specify graph parameter
with tf.Session(graph=graph1) as sess:
    # In this case, not necessary to close the session explicity
    print("with tf.Session() as sess, result: {}".format(sess.run(cc)))

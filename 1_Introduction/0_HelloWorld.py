import tensorflow as tf

print('TensorFlow Version: ' + tf.__version__)

# Create a Constant Operation
# This Op is added to the default graph
# One Op in TensorFlow Compute Graph is one node

# The value returned by tf.constant() is the node handler
hello = tf.constant('Hello, TensorFlow!')

# After define the Graph, TensorFlow must create a session to execute the Graph
# TensorFlow Python Library is an frontend of TensorFlow Core
# The session is to connect the frontend and the real TensorFlow Core
sess = tf.Session()

# Execute the special node by specific node handler and fetches the output of the node
hi = sess.run(hello)
print(hi)

# After execution, session must be closed
sess.close()

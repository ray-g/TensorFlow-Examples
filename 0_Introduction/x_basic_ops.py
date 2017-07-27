import tensorflow as tf
import numpy as np

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
writer = tf.summary.FileWriter('./logs', graph)
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

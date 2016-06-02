import tensorflow as tf
import numpy as np

x     = tf.get_variable('X', shape=[3,3], regularizer=tf.contrib.layers.l2_regularizer(0.1))
loss  = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
train = tf.train.AdamOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())

  for i in range(10):
    print(sess.run(x))
    print(sess.run(loss))
    sess.run(train)

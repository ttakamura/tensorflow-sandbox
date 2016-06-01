# -*- coding: utf-8 -*-
import tensorflow as tf

feature = tf.placeholder(tf.float32, shape=[None, 2])  # 入力は2次元

W = tf.Variable([[ 1.0 ], [ 1.0 ]], name='W')
b = tf.Variable([ -1.0 ], name='b')
y = tf.nn.relu(tf.matmul(feature, W) + b)

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  print( sess.run(y, feed_dict={ feature: [[0.0, 0.0]] }) )
  print( sess.run(y, feed_dict={ feature: [[1.0, 0.0]] }) )
  print( sess.run(y, feed_dict={ feature: [[0.0, 1.0]] }) )
  print( sess.run(y, feed_dict={ feature: [[1.0, 1.0]] }) )

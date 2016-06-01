# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

#
# AND回路を実現するニューラルネットワーク
#
# 入力： [1 or 0]  2次元
# 出力： [1 or 0]  1次元
#

def and_network(x):
  W = tf.Variable([[ 1.0 ], [ 1.0 ]], name='W')
  b = tf.Variable([ -1.0 ], name='b')
  y = tf.matmul(x, W) + b
  return y

def optimizer(y, t):
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y, t)
  loss          = tf.reduce_mean(cross_entropy, name='loss')
  train         = tf.train.AdamOptimizer(0.001).minimize(loss)
  return train

def accuracy(y, t):
  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(t, 1))
  accuracy           = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  return accuracy

def generate_dummy_data(size):
  xs = []
  ys = []
  for i in range(size):
    x = (np.random.rand(2) > 0.5).astype(np.float32)
    y = float(x.sum() == 2.0)
    xs.append(x)
    ys.append(y)
  return np.array(xs).reshape(size, 2), np.array(ys).reshape(size, 1)

# -----------------------------------------------------------------------------------
feature  = tf.placeholder(tf.float32, shape=[None, 2])  # 入力は2次元
teacher  = tf.placeholder(tf.float32, shape=[None, 1])  # 出力は1次元
predict  = and_network(feature)                         # ニューラルネットワークの出力（predict）
trainer  = optimizer(predict, teacher)                  # ニューラルネットワークを最適化する関数
accuracy = accuracy(predict, teacher)                   # おまけ

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())

  for step in range(30):
    data, label = generate_dummy_data(100)
    # sess.run(trainer, feed_dict={ feature: data, teacher: label })

    acc = sess.run(accuracy, feed_dict={ feature: data, teacher: label })
    print("step %d, accuracy %g" % (step, acc))

  # print(sess.run(W[:,0]))
  # print(sess.run(W[:,1]))

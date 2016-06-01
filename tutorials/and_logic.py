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
  W = tf.Variable(tf.truncated_normal([2, 1], stddev=0.01))
  b = tf.Variable(tf.constant(0.0, shape=[1]))
  y = tf.nn.relu(tf.matmul(x, W) + b)
  return W,b,y

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
    x = (np.random.rand(2) > 0.3).astype(np.float32)
    y = float(x.sum() == 2.0)
    xs.append(x)
    ys.append(y)
  return np.array(xs).reshape(size, 2), np.array(ys).reshape(size, 1)

# -----------------------------------------------------------------------------------
feature  = tf.placeholder(tf.float32, shape=[None, 2])  # 入力は2次元
teacher  = tf.placeholder(tf.float32, shape=[None, 1])  # 出力は1次元
W, b, y  = and_network(feature)                         # ニューラルネットワークの出力（predict）
trainer  = optimizer(y, teacher)                        # ニューラルネットワークを最適化する関数
accuracy = accuracy(y, teacher)                         # おまけ

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())

  for step in range(30):
    data, label = generate_dummy_data(100)
    sess.run(trainer, feed_dict={ feature: data, teacher: label })

    acc = sess.run(accuracy, feed_dict={ feature: data, teacher: label })
    print("step %d, accuracy %g" % (step, acc))

  print(sess.run(W))
  print(sess.run(b))

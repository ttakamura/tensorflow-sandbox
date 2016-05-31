# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

#
# 例：邦楽か洋楽か予測する
#
# 入力： [歌手が日本人, 題名が日本語, 題名が英語, 歌手がアメリカ人]
# 出力： [邦楽, 洋楽]
#

def neural_network(x):
  with tf.variable_scope('Layer1') as scope:
    W  = tf.Variable(tf.truncated_normal([4, 10], stddev=0.01))
    b  = tf.Variable(tf.constant(0.0, shape=[10]))
    h  = tf.nn.relu(tf.matmul(x, W) + b)
  with tf.variable_scope('Layer2') as scope:
    W2 = tf.Variable(tf.truncated_normal([10, 2], stddev=0.01))
    b2 = tf.Variable(tf.constant(0.0, shape=[2]))
    y  = tf.matmul(h, W2) + b2
  return y

def optimizer(y, t):
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y, t)
  loss          = tf.reduce_mean(cross_entropy, name='loss')
  train         = tf.train.AdamOptimizer(0.01).minimize(loss)
  return train

def accuracy(y, t):
  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(t, 1))
  accuracy           = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  return accuracy

def generate_dummy_data(batch_size):
  x_batch = []
  t_batch = []
  for i in range(batch_size):
    japanese = np.random.rand() > 0.5
    if japanese:
      data  = np.random.normal([100, 60, 40, 20]) / 100
      label = [1.0, 0.0]
    else:
      data  = np.random.normal([20, 40, 60, 100]) / 100
      label = [0.0, 1.0]
    x_batch.append(data)
    t_batch.append(label)
  return x_batch, t_batch

# -----------------------------------------------------------------------------------
feature = tf.placeholder(tf.float32, shape=[None, 4]) # 入力は４次元
teacher = tf.placeholder(tf.float32, shape=[None, 2]) # 出力は２次元
predict = neural_network(feature)                  # ニューラルネットワークの出力（predict）
trainer = optimizer(predict, teacher)              # ニューラルネットワークを最適化する関数
accuracy = accuracy(predict, teacher)              # おまけ

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())

  for step in range(30):
    singer, label = generate_dummy_data(100)
    sess.run(trainer, feed_dict={ feature: singer, teacher: label })

    acc = sess.run(accuracy, feed_dict={ feature: singer, teacher: label })
    print("step %d, accuracy %g" % (step, acc))

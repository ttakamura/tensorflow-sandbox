# -*- coding: utf-8 -*-
import tensorflow as tf
import singer_dummy_data

#
# 例：邦楽か洋楽か予測する
#
# 入力： [歌手が日本人, 題名が日本語, 題名が英語, 歌手がアメリカ人]
# 出力： [邦楽, 洋楽]
#

def neural_network(x):
  W = tf.Variable(tf.truncated_normal([4, 2], stddev=0.01))
  b = tf.Variable(tf.constant(0.0, shape=[2]))
  y = tf.matmul(x, W) + b
  return W, b, y

def optimizer(y, t):
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y, t)
  loss          = tf.reduce_mean(cross_entropy, name='loss')
  train         = tf.train.AdamOptimizer(0.001).minimize(loss)
  return train

def accuracy(y, t):
  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(t, 1))
  accuracy           = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  return accuracy

# -----------------------------------------------------------------------------------
feature       = tf.placeholder(tf.float32, shape=[None, 4]) # 入力は４次元
teacher       = tf.placeholder(tf.float32, shape=[None, 2]) # 出力は２次元
W, b, predict = neural_network(feature)                  # ニューラルネットワークの出力（predict）
trainer       = optimizer(predict, teacher)              # ニューラルネットワークを最適化する関数
accuracy      = accuracy(predict, teacher)              # おまけ

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())

  for step in range(30):
    singer, label = singer_dummy_data.generate_dummy_data(100)
    sess.run(trainer, feed_dict={ feature: singer, teacher: label })

    acc = sess.run(accuracy, feed_dict={ feature: singer, teacher: label })
    print("step %d, accuracy %g" % (step, acc))

  print(sess.run(W[:,0]))
  print(sess.run(W[:,1]))

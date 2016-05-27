import numpy as np
import tensorflow as tf
import conf
import model
import reader
from IPython import embed

data_dir     = "data/tab_products/images_ss"
batch_size   = 20   # min-batch size
img_width    = 50   # original image width
img_height   = 63   # original image height
category_dim = 213  # master category nums
learn_rate   = 1e-4

with tf.Session(conf.remote_host_uri()) as sess:
  dropout_ratio = tf.placeholder(tf.float32)

  images = tf.placeholder(tf.float32, shape=[None, img_height, img_width, 1])
  labels = tf.placeholder(tf.int64,   shape=[None, category_dim])
  logits = model.small_model(images, 1, category_dim, dropout_ratio)

  cross_entropy      = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  train_step         = tf.train.AdamOptimizer(learn_rate).minimize(cross_entropy_mean)
  correct_prediction = tf.equal(tf.argmax(logits,1), labels)
  accuracy           = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  sess.run(tf.initialize_all_variables())

  # -------- train --------------------------------------
  train, valid, test = reader.open_data(data_dir, batch_size)
  step = 0
  x    = reader.load_images(data_dir, train[step])
  t    = reader.get_categories(train[step])
  embed()

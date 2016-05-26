import numpy as np
import tensorflow as tf
import conf
import model
import reader
from IPython import embed

with tf.Session(conf.remote_host_uri()) as sess:
  c = tf.constant("Hello, distributed TensorFlow!")
  embed()

  # train, valid, test = reader.open_data("data/tab_products/images_ss", 9)
  # reader.get_categories(train[ 0 ])
  # reader.load_images('data/tab_products/images_ss', train[ 0 ])

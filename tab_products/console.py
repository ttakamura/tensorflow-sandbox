import tensorflow as tf
import conf
import model
from IPython import embed

with tf.Session(conf.remote_host_uri()) as sess:
  c = tf.constant("Hello, distributed TensorFlow!")
  embed()

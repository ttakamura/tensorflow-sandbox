import tensorflow as tf

def weight_variable(shape, wd=None):
  w = tf.Variable(tf.truncated_normal(shape, stddev=0.01), name='W')
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(w), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return w

def bias_variable(shape):
  initial = tf.constant(0.0, shape=shape)
  return tf.Variable(initial, name='b')

def conv2d(x, W, stride):
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

def batch_normalize(x):
  epsilon    = 0.001
  shape      = x.get_shape()[-1:]
  scale      = tf.Variable(tf.ones(shape), name='bn_scale')
  beta       = tf.Variable(tf.zeros(shape), name='bn_beta')
  mean, vari = tf.nn.moments(x, [0])
  return tf.nn.batch_normalization(x, mean, vari, beta, scale, epsilon)

def max_pool_layer(x, ksize, stride):
  return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding='SAME')

def conv_layer(x, ksize, stride, input_channel, output_channel):
  W = weight_variable([ksize, ksize, input_channel, output_channel], wd=None)
  b = bias_variable([output_channel])
  z = conv2d(x, W, stride) + b
  h = tf.nn.relu(batch_normalize(z))
  return W, b, h

def fc_layer(x, input_dim, output_dim):
  W = weight_variable([input_dim, output_dim], wd=0.05)
  b = bias_variable([output_dim])
  h = tf.matmul(x, W) + b
  return W, b, h

def avg_pool_layer(x):
  return tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")

# --------------------------------------------------------------------------------
def small_model(x_image, width, height, input_channel, output_dim, dropout_ratio):
  c1_channel    = 32
  res_channel_a = 32
  res_channel_b = 64
  res_channel_c = 128

  # 48px * 1ch
  with tf.variable_scope('conv1') as scope:
    _, _, h_conv1 = conv_layer(x_image, 5, 1, input_channel, c1_channel)
    h_pool1 = max_pool_layer(h_conv1, 3, 2)

  h_res = h_pool1

  # 24px * 32ch
  with tf.variable_scope('resnet_a') as scope:
    _, _, h_res = conv_layer(h_res, 3, 1, res_channel_a, res_channel_a)
    _, _, h_res = conv_layer(h_res, 3, 2, res_channel_a, res_channel_b)

  # 12px * 64ch
  with tf.variable_scope('resnet_b') as scope:
    _, _, h_res = conv_layer(h_res, 3, 1, res_channel_b, res_channel_b)
    _, _, h_res = conv_layer(h_res, 3, 2, res_channel_b, res_channel_c)

  # 6px * 128ch
  with tf.variable_scope('resnet_c') as scope:
    _, _, h_res = conv_layer(h_res, 3, 1, res_channel_c, res_channel_c)
    _, _, h_res = conv_layer(h_res, 3, 1, res_channel_c, res_channel_c)

  # 6px * 128ch
  h_avg_pool = avg_pool_layer(h_res)
  h_avg_pool_dim = res_channel_c

  # 1px * 128ch
  with tf.variable_scope('fc1') as scope:
    W_fc1, b_fc1, h_fc1 = fc_layer(h_avg_pool, h_avg_pool_dim, output_dim)

  return h_fc1
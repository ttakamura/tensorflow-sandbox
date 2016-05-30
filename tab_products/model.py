import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name='W')

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name='b')

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def conv_and_max_pool_layer(x, input_channel, output_channel):
  W = weight_variable([5, 5, input_channel, output_channel])
  b = bias_variable([output_channel])
  h = tf.nn.relu(conv2d(x, W) + b)
  h_pool = max_pool_2x2(h)
  return W, b, h, h_pool

def fc_layer(x, input_dim, output_dim):
  W = weight_variable([input_dim, output_dim])
  b = bias_variable([output_dim])
  h = tf.matmul(x, W) + b
  return W, b, h

def small_model(x_image, width, height, input_channel, output_dim, dropout_ratio):
  c1_channel = 20
  c1_width   = width / 2
  c1_height  = height / 2
  c2_channel = 30
  c2_width   = c1_width / 2
  c2_height  = c1_height / 2
  fc1_dim    = 100

  with tf.variable_scope('conv1') as scope:
    W_conv1, b_conv1, h_conv1, h_pool1 = conv_and_max_pool_layer(x_image, input_channel, c1_channel)

  with tf.variable_scope('conv2') as scope:
    W_conv2, b_conv2, h_conv2, h_pool2 = conv_and_max_pool_layer(h_pool1, c1_channel, c2_channel)

  h_pool2_dim  = int(c2_width * c2_height * c2_channel)
  h_pool2_flat = tf.reshape(h_pool2, [-1, h_pool2_dim])

  with tf.variable_scope('fc1') as scope:
    W_fc1, b_fc1, h_fc1 = fc_layer(h_pool2_flat, h_pool2_dim, fc1_dim)
    h_fc1_drop = tf.nn.dropout(tf.nn.relu(h_fc1), dropout_ratio)

  with tf.variable_scope('fc2') as scope:
    W_fc2, b_fc2, h_fc2 = fc_layer(h_fc1_drop, fc1_dim, output_dim)

  return h_fc2

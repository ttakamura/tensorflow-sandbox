import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial, name='W')

def bias_variable(shape):
  initial = tf.constant(0.001, shape=shape)
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
  W = weight_variable([ksize, ksize, input_channel, output_channel])
  b = bias_variable([output_channel])
  z = conv2d(x, W, stride) + b
  h = tf.nn.relu(batch_normalize(z))
  return W, b, h

def fc_layer(x, input_dim, output_dim):
  W = weight_variable([input_dim, output_dim])
  b = bias_variable([output_dim])
  h = tf.matmul(x, W) + b
  return W, b, h

def avg_pool_layer(x):
  h = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")
  print(x)
  print(h)
  return h

# --------------------------------------------------------------------------------
def small_model(x_image, width, height, input_channel, output_dim, dropout_ratio):
  c1_channel = 32
  c2_channel = 64

  # 48px * 1ch
  with tf.variable_scope('conv1') as scope:
    _, _, h_conv1 = conv_layer(x_image, 5, 1, input_channel, c1_channel)
    h_pool1 = max_pool_layer(h_conv1, 3, 2)

  # 24px * 32ch
  with tf.variable_scope('conv2') as scope:
    _, _, h_conv2 = conv_layer(h_pool1, 5, 1, c1_channel, c2_channel)
    h_pool2 = max_pool_layer(h_conv2, 3, 2)

  h_res = h_pool2
  res_channel = 64

  # 12px * 64ch
  with tf.variable_scope('resnet_a') as scope:
    for i in range(2):
      with tf.variable_scope('%d_1' % i) as scope:
        _, _, h_res = conv_layer(h_res, 3, 1, res_channel, res_channel)
      with tf.variable_scope('%d_2' % i) as scope:
        _, _, h_res = conv_layer(h_res, 3, 1, res_channel, res_channel)
    with tf.variable_scope('conv') as scope:
      _, _, h_res = conv_layer(h_res, 3, 2, res_channel, int(res_channel*2))

  res_channel = 128

  # 6px * 128ch
  with tf.variable_scope('resnet_b') as scope:
    for i in range(2):
      with tf.variable_scope('%d_1' % i) as scope:
        _, _, h_res = conv_layer(h_res, 3, 1, res_channel, res_channel)
      with tf.variable_scope('%d_2' % i) as scope:
        _, _, h_res = conv_layer(h_res, 3, 1, res_channel, res_channel)
    with tf.variable_scope('conv') as scope:
      _, _, h_res = conv_layer(h_res, 3, 2, res_channel, int(res_channel * 2))

  res_channel = 256

  # 3px * 256ch
  h_avg_pool = avg_pool_layer(h_res)
  h_avg_pool_dim = 256

  with tf.variable_scope('fc1') as scope:
    W_fc1, b_fc1, h_fc1 = fc_layer(h_avg_pool, h_avg_pool_dim, output_dim)

  return h_fc1

import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np

# Parameters
learning_rate  = 0.001
training_iters = 100000
batch_size     = 128
display_step   = 10

# Network Parameters
n_input        = 10  # 0 ~ 9
n_steps        = 28  # timesteps
n_hidden       = 128 # hidden layer num of features
n_classes      = 10  # 0 ~ 9

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
  'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
  'out': tf.Variable(tf.random_normal([n_classes]))
}

def RNN(x, weights, biases):
  # Prepare data shape to match `rnn` function requirements
  # Current data input shape: (batch_size, n_steps,    n_input)
  #                     next: (n_steps,    batch_size, n_input)
  x = tf.transpose(x, [1, 0, 2])

  # Reshaping to (n_steps * batch_size, n_input)
  x = tf.reshape(x, [-1, n_input])

  # Split to get a list of 'n_steps' tensors of shape (batch_size, n_hidden)
  # This input shape is required by `rnn` function
  x = tf.split(0, n_steps, x)

  # Define a lstm cell with tensorflow
  lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

  # Get lstm cell output
  outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

  # Linear activation, using rnn inner loop last output
  return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

cost      = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy     = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Sequence() as sess:
  sess.run(tf.initialize_all_variables())

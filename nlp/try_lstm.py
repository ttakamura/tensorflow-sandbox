import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np

batch_size  = 8
num_steps   = 12
input_size  = 1
n_input     = input_size
hidden_size = 128
output_size = 1

input_data = tf.placeholder(tf.float32, [None, num_steps, input_size])
targets    = tf.placeholder(tf.float32, [None, num_steps, output_size])

lstm_cell  = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0)
state      = tf.zeros([ batch_size, lstm_cell.state_size ])

W = tf.get_variable("W", [hidden_size, output_size])
b = tf.get_variable("b", [output_size])

def convert_seq(x):
  # Prepare data shape to match `rnn` function requirements
  # Current data input shape: (batch_size, n_steps,    n_input)
  #                     next: (n_steps,    batch_size, n_input)
  x = tf.transpose(x, [1, 0, 2])

  # Reshaping to (n_steps * batch_size, n_input)
  x = tf.reshape(x, [-1, n_input])

  # Split to get a list of 'n_steps' tensors of shape (batch_size, n_hidden)
  # This input shape is required by `rnn` function
  x = tf.split(0, num_steps, x)
  return x

def generate_data(batch_size):
  a_batches = []
  b_batches = []
  for batch in range(batch_size):
    a_list = []
    b_list = []
    b = 0.0
    for step in range(num_steps):
      a = np.random.rand()
      b = b + a
      a_list.append([a])
      b_list.append([b])
    a_batches.append(a_list)
    b_batches.append(b_list)
  return np.array(a_batches), np.array(b_batches)

loss = 0.0
outputs = []

x_list = convert_seq(input_data)
y_list = convert_seq(targets)

with tf.variable_scope("RNN"):
  for time_step in range(num_steps):
    if time_step > 0: tf.get_variable_scope().reuse_variables()
    output, state = lstm_cell(x_list[time_step], state)
    output = tf.matmul(output, W) + b
    loss += tf.reduce_mean(tf.square(output - y_list[time_step]))
    outputs.append(output)

tf.scalar_summary('total_loss', loss)

global_step = tf.Variable(0, name='global_step', trainable=False)
optimizer   = tf.train.GradientDescentOptimizer(0.01)
train       = optimizer.minimize(loss, global_step=global_step)
summary_op  = tf.merge_all_summaries()

with tf.Session() as sess:
  summary_writer = tf.train.SummaryWriter('log/try_lstm/main', sess.graph)
  sess.run(tf.initialize_all_variables())

  for step in range(1000):
    a, b = generate_data(batch_size)
    feed_dict={ input_data: a, targets: b }

    sess.run(train, feed_dict=feed_dict)
    print(sess.run(loss, feed_dict=feed_dict))

    summary_str = sess.run(summary_op, feed_dict=feed_dict)
    summary_writer.add_summary(summary_str, step)

    if (step % 10 == 0):
        summary_writer.flush()

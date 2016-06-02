import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell

lstm_size  = 5
batch_size = 8
num_steps  = 12

input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
targets    = tf.placeholder(tf.int32, [batch_size, num_steps])
lstm_cell  = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, forget_bias=1.0)
state      = tf.zeros([ batch_size, lstm_size ])

outputs = []
for time_step in range(num_steps):
  output, state = lstm_cell(input_data[:, time_step], state)
  outputs.append(output)

softmax_w  = tf.get_variable("softmax_w", [lstm_size, vocab_size])
softmax_b  = tf.get_variable("softmax_b", [vocab_size])
logits     = tf.matmul(output, softmax_w) + softmax_b

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())

  print(sess.run(logits))

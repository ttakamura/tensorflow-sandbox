import tensorflow as tf
import numpy as np
from IPython import embed

batch_size = 8
lstm_size  = 10
vocab_size = 255
num_steps  = 20

input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
targets    = tf.placeholder(tf.int32, [batch_size, num_steps])

lstm  = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
state = tf.zeros([ batch_size, lstm_size ])

softmax_w = tf.get_variable("softmax_w", [lstm_size, vocab_size])
softmax_b = tf.get_variable("softmax_b", [vocab_size])

text = "Slightly better results can be obtained with forget gate bia"
text = [ord(i) for i in list(text)].reshape(batch_size,-1)

for chunk in text:
  output, state = lstm(chunk, state)

  logits = tf.matmul(output, softmax_w) + softmax_b
  probabilities = tf.nn.softmax(logits)

  loss = tf.nn.seq2seq.sequence_loss_by_example(
    [logits],
    [chunk],
    [tf.ones([batch_size])])

  tf.add_to_collection('losses', loss)


loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
  sess.run(optimizer)

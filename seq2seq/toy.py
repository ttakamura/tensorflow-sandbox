from numpy.random import *
import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell, seq2seq

seq_length    = 5
batch_size    = 10
vocab_size    = 7
embedding_dim = 50
memory_dim    = 100

x_seq   = [tf.placeholder(tf.int32, shape=(None,), name="x%i" % t) for t in range(seq_length)]
t_seq   = [tf.placeholder(tf.int32, shape=(None,), name="t%i" % t) for t in range(seq_length)]
weights = [tf.ones_like(t_i, dtype=tf.float32) for t_i in t_seq]

# Decoder input: prepend some "GO" token and drop the final token of the encoder input
dec_inp = ([tf.zeros_like(x_seq[0], dtype=np.int32, name="GO")] + x_seq[:-1])

# Initial memory value for recurrence.
prev_mem = tf.zeros((batch_size, memory_dim))

# GRU
cell = rnn_cell.GRUCell(memory_dim)
dec_outputs, dec_memory = seq2seq.embedding_rnn_seq2seq(x_seq, dec_inp, cell, vocab_size, vocab_size)
loss = seq2seq.sequence_loss(dec_outputs, t_seq, weights, vocab_size)
tf.scalar_summary("loss", loss)

magnitude = tf.sqrt(tf.reduce_sum(tf.square(dec_memory[1])))
tf.scalar_summary("magnitude at t=1", magnitude)

summary_op = tf.merge_all_summaries()

learning_rate = 0.05
momentum = 0.9
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
train_op = optimizer.minimize(loss)
logdir = tempfile.mkdtemp()
print logdir

def generate_data():
  X = [np.random.choice(vocab_size, size=(seq_length,), replace=False) for _ in range(batch_size)]
  Y = X[:]
  # Dimshuffle to seq_len * batch_size
  X = np.array(X).T
  Y = np.array(Y).T
  return X, Y

with tf.Session() as sess:
  summary_writer = tf.train.SummaryWriter(logdir, sess.graph_def)
  sess.run(tf.initialize_all_variables())

  for step in range(500):
    X, Y = generate_data()

    feed_dict = {x_seq[t]: X[t] for t in range(seq_length)}
    feed_dict.update({t_seq[t]: Y[t] for t in range(seq_length)})

    _, loss_t, summary = sess.run([train_op, loss, summary_op], feed_dict)
    summary_writer.add_summary(summary, step)

  summary_writer.flush()

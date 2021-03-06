import tensorflow as tf
import numpy as np

filename_queue = tf.train.string_input_producer(["data/file1.csv", "data/file2.csv"])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

col1, col2, col3, col4, col5 = tf.decode_csv(value, record_defaults=[[1], [1], [1], [1], [1]])
features = tf.pack([col1, col2, col3, col4])

with tf.Session() as sess:
  # Start populating the filename queue.
  coord   = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  print("request")
  for i in range(100):
    print(sess.run([key, value]))
    print(sess.run([features, col5]))
  print("done")

  coord.request_stop()
  coord.join(threads)

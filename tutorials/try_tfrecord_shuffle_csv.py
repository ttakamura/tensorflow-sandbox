import tensorflow as tf
import numpy as np

filename_queue = tf.train.string_input_producer(["data/file1.csv", "data/file2.csv"])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

col1, col2, col3, col4, col5 = tf.decode_csv(value, record_defaults=[[1], [1], [1], [1], [1]])
features = tf.pack([col1, col2, col3, col4])

batch_size = 10
min_after_dequeue = 100
capacity = min_after_dequeue + 3 * batch_size

features_batch = tf.train.shuffle_batch([features],
                                        batch_size=batch_size,
                                        capacity=capacity,
                                        min_after_dequeue=min_after_dequeue)

y = features_batch * 10

with tf.Session() as sess:
  # Start populating the filename queue.
  coord   = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(100):
    print("-------- %d ---------" % i)
    print(sess.run(y))

  coord.request_stop()
  coord.join(threads)

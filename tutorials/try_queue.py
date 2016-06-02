import tensorflow as tf
import numpy as np

q = tf.FIFOQueue(3, "float")
x = q.dequeue()
y = x + 10

queue_y = q.enqueue([y])
queue_1 = q.enqueue([1])

with tf.Session() as sess:
  from IPython import embed
  embed()

import tensorflow as tf
import numpy as np
import time

batch_size = 4
counter    = tf.Variable(0.0)
inc_count  = tf.assign(counter, counter + 1.0)
example    = tf.pack([counter])

queue      = tf.RandomShuffleQueue(100, 5, np.float32, shapes=[1])
enqueue_op = queue.enqueue(example)
dequeue_op = queue.dequeue()

inputs     = queue.dequeue_many(batch_size)
train_op   = inputs * 10

q_runner   = tf.train.QueueRunner(queue, [enqueue_op] * 4)

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())

  # Create a coordinator, launch the queue runner threads.
  coord = tf.train.Coordinator()
  enqueue_threads = q_runner.create_threads(sess, coord=coord, start=True)

  # Run the training loop, controlling termination with the coordinator.
  for step in xrange(30):
    # print("Q size %d" % sess.run(queue.size()))

    for i in range(10):
      sess.run(enqueue_op)

    print(sess.run(train_op))
    sess.run(inc_count)

  # When done, ask the threads to stop.
  coord.request_stop()
  # And wait for them to actually do it.
  coord.join(threads)

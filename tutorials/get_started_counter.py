import tensorflow as tf

state = tf.Variable(0, name="counter")

one = tf.constant(1)
next_val = tf.add(state, one)
updater  = tf.assign(state, next_val)

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(state))

    for _ in range(3):
        sess.run(updater)
        print(sess.run(state))

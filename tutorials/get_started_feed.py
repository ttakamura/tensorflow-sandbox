import tensorflow as tf

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
y  = tf.mul(x1, x2)

with tf.Session() as sess:
    res = sess.run(y, feed_dict={ x1:[3.], x2:[10.] })
    print(res)

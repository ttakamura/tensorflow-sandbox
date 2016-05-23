import tensorflow as tf

X = tf.constant([[1.0, 10.0]])

Y = tf.constant([[2.0],
                 [3.0]])

Z1 = tf.matmul(X, Y)
Z2 = tf.matmul(Y, X)

with tf.Session() as sess:
    result = sess.run(Z1)
    print(result)

    result = sess.run(Z2)
    print(result)

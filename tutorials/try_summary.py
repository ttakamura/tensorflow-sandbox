from numpy.random import *
import numpy as np
import tensorflow as tf

#
# Try FizzBuzz by TensorFlow
#

def fizbuz(i):
    if (i % 15 == 0):
        return 1
    if (i % 5 == 0):
        return 2
    if (i % 3 == 0):
        return 3
    return 0

def generate_data(batch_size):
    x_data  = np.zeros((batch_size, 15), dtype=np.float32)
    x_teach = np.zeros((batch_size, 4),  dtype=np.float32)
    for j in xrange(batch_size):
        current_i = randint(15) + 1
        x_data[j,  current_i - 1] = 1.0
        x_teach[j, fizbuz(current_i)] = 1.0
    return (x_data, x_teach)

batch_size = 10

x     = tf.placeholder(tf.float32, shape=[batch_size, 15])
t     = tf.placeholder(tf.float32, shape=[batch_size, 4])
w1    = tf.Variable(tf.truncated_normal([15, 4], stddev=0.1))
b1    = tf.Variable(tf.constant(0.1, shape=[4]))
model = tf.matmul(x, w1) + b1
y     = tf.nn.softmax(model)
loss  = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y), reduction_indices=[1]))
train = tf.train.AdamOptimizer(0.01).minimize(loss)
top_y = tf.argmax(y,1)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for step in xrange(20):
        # ---- train ---------------------------------------
        for i in xrange(5):
            x_data, x_teach = generate_data(batch_size)
            train.run(feed_dict={ x: x_data, t: x_teach })

        # ---- confirm--------------------------------------
        x_data, x_teach = generate_data(batch_size)
        print("--------- %d -------------" % step)
        print(x_teach.argmax(1))
        print(top_y.eval(feed_dict={ x: x_data, t: x_teach }))
        tmp_w = w1.eval()
        tmp_w[tmp_w < 0.0] = 0.0
        for i in xrange(15):
            print(i+1, tmp_w[i,:])
        # print(b1.eval())

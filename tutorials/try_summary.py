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

with tf.Graph().as_default():
    x     = tf.placeholder(tf.float32, shape=[None, 15])
    t     = tf.placeholder(tf.float32, shape=[None, 4])
    w1    = tf.Variable(tf.truncated_normal([15, 4], stddev=0.1))
    b1    = tf.Variable(tf.constant(0.1, shape=[4]))
    model = tf.matmul(x, w1) + b1
    y     = tf.nn.softmax(model)
    top_y = tf.argmax(y,1)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y, t)
    loss          = tf.reduce_mean(cross_entropy, name='loss')
    tf.scalar_summary(loss.op.name, loss)
    global_step   = tf.Variable(0, name='global_step', trainable=False)
    train         = tf.train.AdamOptimizer(0.01).minimize(loss, global_step=global_step)

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(t, 1))
    accuracy           = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    summary_op         = tf.merge_all_summaries()

    with tf.Session() as sess:
        run_name = str(randint(1000))
        summary_writer = tf.train.SummaryWriter('data/fizz_buzz/' + run_name, sess.graph)
        sess.run(tf.initialize_all_variables())

        for step in xrange(200):
            # ---- train ---------------------------------------
            x_data, x_teach = generate_data(batch_size)
            train.run(feed_dict={ x: x_data, t: x_teach })

            # ---- confirm--------------------------------------
            if (step % 10 == 0):
                x_data, x_teach = generate_data(batch_size * 10)
                acc = accuracy.eval(feed_dict={ x: x_data, t: x_teach })
                print("step %d, accuracy %g -------------" % (step, acc))
                # print(x_teach.argmax(1))
                # print(top_y.eval(feed_dict={ x: x_data, t: x_teach }))

                summary_str = sess.run(summary_op, feed_dict={ x: x_data, t: x_teach })
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # if (acc > 0.99):
            #     tmp_w = w1.eval()
            #     tmp_w[tmp_w < 0.0] = 0.0
            #     for i in xrange(15):
            #         print(i+1, tmp_w[i,:])
            #     # print(b1.eval())

from numpy.random import *
import numpy as np
import tensorflow as tf
import time

#
# Try FizzBuzz by TensorFlow
#
def fizbuz_decode(xi, i):
    if (i == 1):
        return 'FizzBuzz'
    if (i == 2):
        return 'Buzz'
    if (i == 3):
        return 'Fizz'
    return str(xi)

def fizbuz_encode(i):
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
        x_teach[j, fizbuz_encode(current_i)] = 1.0
    return (x_data, x_teach)

batch_size = 10
run_name   = str(randint(1000))

with tf.Graph().as_default():
    # ------------------- model -----------------------------------------------
    x     = tf.placeholder(tf.float32, shape=[None, 15], name='X')
    t     = tf.placeholder(tf.float32, shape=[None, 4],  name='T')
    w1    = tf.Variable(tf.truncated_normal([15, 4], stddev=0.1), name='W1')
    b1    = tf.Variable(tf.constant(0.1, shape=[4]), name='b1')
    model = tf.matmul(x, w1) + b1
    y     = tf.nn.softmax(model, name='Y')
    top_y = tf.argmax(y,1)

    # -------------------- train -----------------------------------------------
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y, t)
    loss          = tf.reduce_mean(cross_entropy, name='loss')
    tf.scalar_summary(loss.op.name, loss)
    global_step   = tf.Variable(0, name='global_step', trainable=False)
    train         = tf.train.AdamOptimizer(0.01).minimize(loss, global_step=global_step)

    # -------------------- test -----------------------------------------------
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(t, 1))
    accuracy           = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    summary_op         = tf.merge_all_summaries()

    with tf.Session() as sess:
        summary_writer = tf.train.SummaryWriter('data/fizz_buzz/' + run_name, sess.graph)
        sess.run(tf.initialize_all_variables())

        for step in xrange(300):
            # ---- train ---------------------------------------
            x_data, x_teach = generate_data(batch_size)
            train.run(feed_dict={ x: x_data, t: x_teach })

            # ---- confirm--------------------------------------
            if (step % 10 == 0):
                x_data, x_teach = generate_data(batch_size * 10)
                acc = accuracy.eval(feed_dict={ x: x_data, t: x_teach })

                print("step %d, accuracy %g -------------" % (step, acc))
                tx = x_data.argmax(1) + 1
                ty = top_y.eval(feed_dict={ x: x_data, t: x_teach })
                for i in xrange(batch_size):
                    print("input: %d,\toutput: %s" % (tx[i], fizbuz_decode(tx[i], ty[i])))

                summary_str = sess.run(summary_op, feed_dict={ x: x_data, t: x_teach })
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
                time.sleep(1)

            # if (acc > 0.99):
            #     tmp_w = w1.eval()
            #     tmp_w[tmp_w < 0.0] = 0.0
            #     for i in xrange(15):
            #         print(i+1, tmp_w[i,:])
            #     # print(b1.eval())

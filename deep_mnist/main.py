import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

sess  = tf.InteractiveSession()
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x  = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W  = weight_variable([784, 10])
b  = bias_variable([10])

sess.run(tf.initialize_all_variables())

y = tf.nn.softmax(tf.matmul(x, W) + b)

# --- get numpy array ---
#
# y.eval(feed_dict={ x: batch[0], y_: batch[1] })
#

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step    = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for j in range(10):
    for i in range(100):
        batch = mnist.train.next_batch(50)
        train_step.run(feed_dict={ x: batch[0], y_: batch[1] })
    print("loop %d -----" % (j * 100))
    print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

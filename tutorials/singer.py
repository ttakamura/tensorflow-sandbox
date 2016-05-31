import tensorflow as tf

#
# 例：邦楽か洋楽か予測する
#
# 入力： [歌手が日本人, 題名が日本語, 題名が英語, 歌手がアメリカ人]
# 出力： [邦楽, 洋楽]
#

def neural_network(x):
    W  = tf.Variable(tf.truncated_normal([4, 10], stddev=0.01))
    b  = tf.Variable(tf.constant(0.0, shape=[10]))
    h  = tf.relu(tf.matmul(x, W) + b)
    W2 = tf.Variable(tf.truncated_normal([10, 2], stddev=0.01))
    b2 = tf.Variable(tf.constant(0.0, shape=[10]))
    y  = tf.matmul()
    return y

def optimizer(y, t):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y, t)
    loss          = tf.reduce_mean(cross_entropy, name='loss')
    train         = tf.train.AdamOptimizer(0.01).minimize(loss)
    return train

def accuracy(y, t):
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(t, 1))
    accuracy           = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

feature = tf.placeholder(tf.float32, shape=[4]) # 入力は４次元
teacher = tf.placeholder(tf.float32, shape=[2]) # 出力は２次元
predict = neural_network(feature)               # ニューラルネットワークの出力（predict）
trainer = optimizer(predict, teacher)           # ニューラルネットワークを最適化する関数

# おまけ
accuracy = accuracy(predict, teacher)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for step in range(30):
        input_data, label_data = generate_dummy_data(feature, teacher)

        trainer.run(feed_dict=input_data)

        print("step %d, accuracy %g -------------" % (step, accuracy.run(feed_dict=input_data)))

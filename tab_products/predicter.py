import tensorflow as tf
import reader

def predict(sess, logits, images, labels, data_dir, valid_data, dropout_ratio):
    feed_dict = reader.feed_dict(data_dir, valid_data, 1.0, images, labels, dropout_ratio)
    predict_labels = sess.run(tf.argmax(logits, 1),feed_dict=feed_dict)
    actual_labels  = feed_dict[labels]
    for i in range(predict_labels.shape[0]):
        print("product_id %d - category predicted: %d actual: %d" % (valid_data[i][2], predict_labels[i], actual_labels[i]))

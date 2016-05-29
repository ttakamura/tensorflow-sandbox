import os
import time
import numpy as np
import tensorflow as tf
import conf
import model
import reader
import trainer
from IPython import embed

image_set    = "images_ss"
model_type   = "small_v1"
model_name   = ("%s_%s" % (image_set, model_type))
data_dir     = ("data/tab_products/%s" % image_set)
model_dir    = ("models/tab_products/%s" % (model_name))
log_dir      = ("log/tab_products/%s" % model_name)
batch_size   = 100  # min-batch size
img_width    = 48   # original image width
img_height   = 48   # original image height
img_channel  = 1    # original image channel
category_dim = 213  # master category nums
learn_rate   = 1e-4

def load_model(images, saver, sess):
  logits = model.small_model(images, img_width, img_height, img_channel, category_dim, dropout_ratio)
  ckpt   = tf.train.get_checkpoint_state(model_dir)
  if ckpt:
    last_model = ckpt.model_checkpoint_path
    print("Reading model parameters from '%s'" % last_model)
    saver.restore(sess, last_model)
  else:
    print("Created model with fresh parameters.")
    sess.run(tf.initialize_all_variables())
  return logits

with tf.Session(conf.remote_host_uri()) as sess:
  global_step   = tf.Variable(0, name='global_step', trainable=False)
  dropout_ratio = tf.placeholder(tf.float32, name='dropout_ratio')
  images        = tf.placeholder(tf.float32, shape=[None, img_height, img_width, img_channel], name='images')
  labels        = tf.placeholder(tf.int64,   shape=[None], name='labels')
  saver         = tf.train.Saver(max_to_keep=10)

  logits    = load_model(images, saver, sess)
  train_opt = trainer.optimizer(logits, labels, learn_rate, global_step)
  accuracy  = trainer.evaluater(logits, labels)

  sess.run(tf.initialize_all_variables())

  summary_op     = tf.merge_all_summaries()
  summary_writer = tf.train.SummaryWriter(log_dir, sess.graph)

  training_accuracy_summary   = tf.scalar_summary("training_accuracy", accuracy)
  validation_accuracy_summary = tf.scalar_summary("validation_accuracy", accuracy)

  # -------- train ------------------------------------------
  train, valid, test = reader.open_data(data_dir, batch_size)

  # embed()

  for epoch in xrange(100):
    for i in xrange(len(train)):
      step = tf.train.global_step(sess, global_step)

      train_data = reader.feed_dict(data_dir, train[i], 0.5, images, labels, dropout_ratio)
      sess.run(train_opt, feed_dict=train_data)

      main_summary = sess.run(summary_op, feed_dict=train_data)
      summary_writer.add_summary(main_summary, step)

      if (step % 10 == 0):
        train_data = reader.feed_dict(data_dir, train[i], 1.0, images, labels, dropout_ratio)
        valid_data = reader.feed_dict(data_dir, valid,    1.0, images, labels, dropout_ratio)

        valid_acc_score, valid_acc_summary = sess.run([accuracy, validation_accuracy_summary], feed_dict=valid_data)
        train_acc_score, train_acc_summary = sess.run([accuracy, training_accuracy_summary], feed_dict=train_data)
        print("step %d, valid accuracy %g, train accuracy %g" % (step, valid_acc_score, train_acc_score))

        summary_writer.add_summary(valid_acc_summary, step)
        summary_writer.add_summary(train_acc_summary, step)
        summary_writer.flush()

        checkpoint_path = os.path.join(model_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

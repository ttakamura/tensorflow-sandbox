import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_path', 'models/temp/', 'save and restore')

def main(argv=None):
  x   = tf.Variable(0, name='X')
  inc = tf.assign(x, x+1)

  step = tf.Variable(0, name='global_step')
  next_step = tf.assign(step, step+1)

  saver = tf.train.Saver(max_to_keep=10)

  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_path)

    if ckpt:
      last_model = ckpt.model_checkpoint_path
      print("Reading model from %s" % last_model)
      saver.restore(sess, last_model)
    else:
      print("Initialize")
      sess.run(tf.initialize_all_variables())

    for i in range(3):
      print("step %d, x %d" % (step.eval(), x.eval()))
      inc.eval()
      next_step.eval()

    saver.save(sess, FLAGS.model_path+'model', global_step=step.eval())

if __name__ == '__main__':
  tf.app.run()

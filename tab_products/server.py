#
# export DOCKER_HSOT=tcp://....:4243
# docker run -it -p 8888:8888 gcr.io/tensorflow/tensorflow bash
# python
#
import tensorflow as tf
import os

model_dir = "./models/"

if not os.path.exists(model_dir):
  os.mkdir(model_dir)

cluster = tf.train.ClusterSpec({"tomato": ["localhost:8888"]})
server  = tf.train.Server(cluster, job_name="tomato", task_index=0)
server.join()

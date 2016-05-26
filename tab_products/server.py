#
# export DOCKER_HSOT=tcp://....:4243
# docker run -it -p 8888:8888 gcr.io/tensorflow/tensorflow bash
# python
#
import tensorflow as tf
cluster = tf.train.ClusterSpec({"tomato": ["localhost:8888"]})
server  = tf.train.Server(cluster, job_name="tomato", task_index=0)

import tensorflow as tf
import re
import os

def remote_host_uri():
  host = re.match(r"tcp://(.+):", os.environ['DOCKER_HOST']).group(1)
  return ("grpc://%s:8888" % host)

import tensorflow as tf
import re
import os

def remote_host_uri():
  host = os.environ['TF_HOST']
  return ("grpc://%s:8888" % host)

import numpy as np

def generate_dummy_data(batch_size):
  x_batch = []
  t_batch = []
  for i in range(batch_size):
    japanese = np.random.rand() > 0.5
    if japanese:
      data  = np.random.normal([100, 60, 40, 20]) / 100
      label = [1.0, 0.0]
    else:
      data  = np.random.normal([20, 40, 60, 100]) / 100
      label = [0.0, 1.0]
    x_batch.append(data)
    t_batch.append(label)
  return x_batch, t_batch

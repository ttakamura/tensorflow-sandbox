from PIL import Image
import numpy as np

def open_data(dir_path, batch_size):
    results = []
    with open("%s/data.tsv" % dir_path) as f:
        for row in f:
            cat1, cat2, prod_id = row.rstrip().split("\t")
            results.append([int(cat1), int(cat2), int(prod_id)])
    return split_batches(make_batch(results, batch_size))

def make_batch(data_list, batch_size):
    data = np.array(data_list)
    data = data[0:(len(data_list) // batch_size) * batch_size]
    data = data.reshape((-1, batch_size, 3))
    return data

def split_batches(batches):
    total_size = batches.shape[0]
    train_size = total_size * 80 // 100
    valid_size = total_size * 90 // 100
    test_size  = total_size
    np.random.shuffle(batches)
    train_batches = batches[0:train_size]
    valid_batches = batches[train_size:valid_size].reshape(-1, 3)[0:300]
    test_batches  = batches[valid_size:test_size][0].reshape(-1, 3)[0:300]
    print(train_batches.shape)
    print(valid_batches.shape)
    print(test_batches.shape)
    return train_batches, valid_batches, test_batches

def load_images(dir_path, batch):
    results = []
    for id in batch[:,2]:
        path = "%s/%s.jpg" % (dir_path, id)
        img = np.array(Image.open(path).convert("L"), 'f')
        w, h = img.shape
        results.append(img.reshape((w, h, 1)))
    results = np.array(results) / 255.0
    return results

def get_categories(batch):
    # batch = [second_category, third_category, id]
    return batch[:,0]

def feed_dict(data_dir, batch, dropout_ratio_value, images, labels, dropout_ratio):
  x    = load_images(data_dir, batch)
  t    = get_categories(batch)
  feed = {images: x, labels: t, dropout_ratio: dropout_ratio_value}
  return feed

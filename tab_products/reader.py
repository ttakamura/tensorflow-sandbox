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
    train_size = total_size * 60 // 100
    valid_size = total_size * 80 // 100
    test_size  = total_size
    np.random.shuffle(batches)
    train_batches = batches[0:train_size]
    valid_batches = batches[train_size:valid_size]
    test_batches  = batches[valid_size:test_size]
    return train_batches, valid_batches, test_batches

def load_images(dir_path, batch):
    results = []
    for id in batch[:,2]:
        img = np.array(Image.open("%s/%s.jpg" % (dir_path, id)).convert("L"), 'f')
        results.append(img)
    results = np.array(results) / 255.0
    return results

def get_categories(batch):
    # batch = [second_category, third_category, id]
    return batch[:,0]

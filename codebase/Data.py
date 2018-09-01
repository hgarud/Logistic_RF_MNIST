import numpy as np
import gzip
from sklearn.preprocessing import OneHotEncoder

class MNIST_Data(object):

    def __init__(self, base_dir, img_size):
        self.base_dir = base_dir
        self.img_size = img_size

    def _load_labels(self, file_name):
        file_path = self.base_dir + file_name

        with gzip.open(file_path, 'rb') as f:
                labels = np.frombuffer(f.read(), np.uint8, offset=8)

        return labels

    def _load_imgs(self, file_name):
        file_path = self.base_dir + file_name

        with gzip.open(file_path, 'rb') as f:
                images = np.frombuffer(f.read(), np.uint8, offset=16)
        images = images.reshape(-1, self.img_size)

        return images

if __name__ == '__main__':
    mnist_loader = MNIST_Data(base_dir = "/home/hrishi/1Hrishi/ECE542_Neural_Networks/Homeworks/2/Data/", img_size = 784)
    train_labels = mnist_loader._load_labels("train-labels-idx1-ubyte.gz")
    onehot_encoder = OneHotEncoder(n_values = 10, sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(train_labels.reshape(-1,1))
    print(train_labels)
    print(onehot_encoded)

from Data import MNIST_Data
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

mnist_loader = MNIST_Data(base_dir = "/home/hrishi/1Hrishi/ECE542_Neural_Networks/Homeworks/2/Data/", img_size = 784)

train_images = mnist_loader._load_imgs("train-images-idx3-ubyte.gz")

train_labels = mnist_loader._load_labels("train-labels-idx1-ubyte.gz")
onehot_encoder = OneHotEncoder(n_values = 10, sparse=False)
train_labels = onehot_encoder.fit_transform(train_labels.reshape(-1,1))
# print(train_labels)
# print(onehot_encoded)


X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.33, shuffle = True, random_state=42)

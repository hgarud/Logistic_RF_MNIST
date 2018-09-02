from Data import MNIST_Data
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import csv

mnist_loader = MNIST_Data(base_dir = "/home/hrishi/1Hrishi/ECE542_Neural_Networks/Homeworks/2/Data/", img_size = 784)

X_train = mnist_loader._load_imgs("train-images-idx3-ubyte.gz")
y_train = mnist_loader._load_labels("train-labels-idx1-ubyte.gz")
X_test = mnist_loader._load_imgs("t10k-images-idx3-ubyte.gz")
y_test = mnist_loader._load_labels("t10k-labels-idx1-ubyte.gz")

# np.random.seed(1)  # Reset random state
# np.random.shuffle(X_train)
# np.random.shuffle(y_train)

input = np.append(X_train, y_train[:,None], axis=1)
# print(input.shape)
np.random.shuffle(input)
X_train = input[:,0:784]
y_train = input[:,784]

# X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.33, shuffle = True, random_state=42)

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)

# from sklearn.decomposition import PCA
# pca = PCA(n_components = 256)
# X_train = pca.fit_transform(X_train)
# X_test = pca.fit_transform(X_test)

# l2-sag-ovr = 91.25% acc without standard scaling
# l2-sag-multinomial = 91.91% acc without standard scaling
# l1-saga-ovr = 91.37% acc without standard scaling
# l1-saga-multinomial = 92.29% acc without standard scaling

# logistic_regressor = LogisticRegression(penalty = 'l1', solver = 'saga', tol = 1e-1, multi_class = 'multinomial', verbose = 1, n_jobs = -1)
# logistic_regressor.fit(X_train, y_train)
#
# predictions = logistic_regressor.predict(X_test)
# from sklearn.metrics import accuracy_score
# print(accuracy_score(y_test, predictions))
#
# onehot_encoder = OneHotEncoder(n_values = 10, sparse = False, dtype = np.int8)
# predictions = onehot_encoder.fit_transform(y_train.reshape(-1,1))
# np.savetxt('lr.csv', predictions, delimiter = ',', fmt = '%i')

from sklearn.ensemble import RandomForestClassifier
random_forest_regressor = RandomForestClassifier(criterion = 'entropy', verbose = 1)
random_forest_regressor.fit(X_train, y_train)

predictions = random_forest_regressor.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))

onehot_encoder = OneHotEncoder(n_values = 10, sparse = False, dtype = np.int8)
predictions = onehot_encoder.fit_transform(y_train.reshape(-1,1))
np.savetxt('rf.csv', predictions, delimiter = ',', fmt = '%i')

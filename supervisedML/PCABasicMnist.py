from  sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import seaborn as sb
from sklearn.decomposition import PCA
from scipy.io import loadmat
import matplotlib.pyplot as plt

#Load data
mnist_raw = loadmat("mnist-original.mat")
mnist={
    "data":mnist_raw["data"].T,
    "target":mnist_raw["label"][0]
}

x_train, x_test, y_train,y_test = train_test_split(mnist["data"],mnist["target"],random_state=0)

pca = PCA(.80)
data = pca.fit_transform(x_train)
result = pca.inverse_transform(data)

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(mnist["data"][0].reshape(28,28),cmap=plt.cm.gray,interpolation='nearest')
plt.xlabel("784 Features")
plt.title("Origin")

plt.subplot(1,2,2)
plt.imshow(result[0].reshape(28,28),cmap=plt.cm.gray,interpolation='nearest')
plt.xlabel("154 Features")
plt.title("PCA Images")
plt.show()


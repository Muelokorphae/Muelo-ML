from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read Data
df = pd.read_csv("diabetes.csv")

# Data
x= df.drop("Outcome",axis=1).values

# Outcome Data
y=df['Outcome'].values

# Separate data based on data that already exists
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4)

# Creating model/ find k to model
k_neighbors = np.arange(1,9)

# Empty array
train_score = np.empty(len(k_neighbors))
test_score = np.empty(len(k_neighbors))
for i,k in enumerate(k_neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)

    #Measuring qualities
    train_score[i] = knn.score(x_train,y_train)
    test_score[i] = knn.score(x_test,y_test)

plt.title("Comparing k in model")
plt.plot(k_neighbors,test_score,label="Test Score")
plt.plot(k_neighbors,train_score,label="Train Score")
plt.legend()
plt.xlabel("K Number")
plt.ylabel("Score")
plt.show()









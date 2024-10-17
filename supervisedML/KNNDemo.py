from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,accuracy_score


# iris_datatset = load_iris()
# x_train,x_test,y_train,y_test = train_test_split(iris_datatset['data'],iris_datatset['target'],test_size=0.4,random_state=0)


# # Model
# knn = KNeighborsClassifier(n_neighbors=3)

# # Training
# knn.fit(x_train,y_train)

# # predicttion
# y_pred = knn.predict(x_test)

# print(classification_report(y_test,y_pred,target_names=iris_datatset['target_names']))
# print("Accuracy:",accuracy_score(y_test,y_pred)*100)
dataset_iris = load_iris()
x_train,x_test,y_train,y_test = train_test_split(dataset_iris["data"],dataset_iris["target"],test_size=0.2,random_state=0)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
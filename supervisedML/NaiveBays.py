from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#Loading dataset
iris = load_iris()

#Assign attribute,target 
x= iris['data']
y= iris['target']

#Train and Test
x_train,x_test,y_train,y_test = train_test_split(x,y)

#Model
model =GaussianNB()

#Train
model.fit(x_train,y_train)

#Prediction
y_pred = model.predict(x_test)

#Accuracy
print("Accuracy: ",accuracy_score(y_test,y_pred))
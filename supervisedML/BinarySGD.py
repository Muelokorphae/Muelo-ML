from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
# from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_predict
import itertools

def displayConfusionMatrix(cm,cmap=plt.cm.GnBu):
    classes=["Other Number","Number 5"]
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title("Confusion Matrix")
    plt.colorbar()
    trick_marks=np.arange(len(classes))
    plt.xticks(trick_marks,classes)
    plt.yticks(trick_marks,classes)
    thresh=cm.max()/2
    for i , j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,format(cm[i,j],'d'),
        horizontalalignment='center',
        color='white' if cm[i,j]>thresh else 'black')

    plt.tight_layout()
    plt.ylabel('Actually')
    plt.xlabel('Prediction')
    plt.show()

def displayImage(x):
    plt.imshow(x.reshape(28,28),cmap=plt.cm.binary,interpolation="nearest")
    plt.show()

def displayPredict(clf,actually_y,x):
    print("Actually = ",actually_y)
    print("Prediction =",clf.predict([x])[0])


mnist_raw =loadmat("mnist-original.mat")

mnist ={
    "data":mnist_raw["data"].T,
    "target":mnist_raw["label"][0]
}

x,y = mnist["data"],mnist["target"]

#training , test
x_train, x_test, y_train, y_test = x[:60000],x[60000:],y[:60000],y[60000:]

predict_number = 5500
y_train_5 = (y_train==5)
y_test_5 = (y_test==5)

sgd_clf =SGDClassifier()
sgd_clf.fit(x_train,y_train_5)



# # Showing the result
# displayPredict(sgd_clf,y_test_5[predict_number],x_test[predict_number],score)
# displayImage(x_test[predict_number])

# #Testing Quality of Model by using validation

# score =cross_val_score(sgd_clf,x_train,y_train_5,cv=3,scoring="accuracy")
y_train_pred = cross_val_predict(sgd_clf, x_train, y_train_5, cv=3)
cm=confusion_matrix(y_train_5,y_train_pred)

# # Display confusionMatrix
# plt.figure()
# displayConfusionMatrix(cm)

y_test_pred = sgd_clf.predict(x_test)

classes=['Other number','Number 5']
print(classification_report(y_test_5,y_test_pred,target_names=classes))


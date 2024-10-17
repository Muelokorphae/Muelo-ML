import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Functions
def cleandata(dataset):
    for column in dataset.columns:
        if dataset[column].dtype == type(object):
            le = LabelEncoder()
            dataset[column]=le.fit_transform(dataset[column])
    return dataset

def split_feature_class(dataset,feature):
    features = dataset.drop(feature,axis=1)
    labels = dataset[feature].copy()
    return features,labels


# reading data from csv file
dataset = pd.read_csv('adult.csv')

dataset = cleandata(dataset)

#splitting  Train and Test 
training_set, test_set = train_test_split(dataset,test_size=0.2)
#train
train_features, train_labels =split_feature_class(dataset,"income")
#test
test_features, test_labels = split_feature_class(test_set,"income")

#creating model
model = GaussianNB()
model.fit(train_features,train_labels)

#prediction
clf_pred = model.predict(test_features)

print("Accuracy =",accuracy_score(test_labels,clf_pred))
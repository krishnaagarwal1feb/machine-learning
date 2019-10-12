# mushrooms classification into poisonous(p=1) or edible(e=0)
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

#%% Importing the dataset
dataset = pd.read_csv('mushrooms.csv')
X = dataset.iloc[:, 1:].values
#just checking what will happen if i only consider one feautre 
#afterwards we have to include all the features 1: ...
y = dataset.iloc[:, 0].values

#%% Taking care of missing data
# as i obs here there is no missing data in this dataset
"""from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:])
X[:, 1:3] = imputer.transform(X[:, 1:])
"""
print("Shape of data: {}".format(dataset.shape))
#%% Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
for i in range(0,22,1) :
    X[:,i] = labelencoder_X.fit_transform(X[:,i])

#%% now we have to one hot encode every feature in the matrix X 
onehotencoder = OneHotEncoder(categorical_features = 'all')
X = onehotencoder.fit_transform(X).toarray()

#%% Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#%% Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#%% Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 30, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

#%% Predicting the Test set results
y_pred = classifier.predict(X_test)

#%% Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#%% accuracy of the model 
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
#%% plotting the data and reviewing optimal k value
"""this will guide us to select the optimal number of neighbours 
for our model""" 
from sklearn.model_selection import cross_val_score

# creating odd list of K for KNN
myList = list(range(1,200))

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for k in myList[::2]:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

MSE = [1 - x for x in cv_scores]
# determining best k
optimal_k = myList[::2][MSE.index(min(MSE))]
print ("The optimal number of neighbors is %d" % optimal_k)

#%% plot misclassification error vs k
"""
plt.plot(myList[::2], MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()"""

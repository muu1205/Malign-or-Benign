import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import metrics

data = pd.read_csv("data.csv",header=0)
#data.info()
data.drop("id",axis=1,inplace=True)
print(data.columns)

features_mean= list(data.columns[1:11])
features_se= list(data.columns[11:20])
features_worst=list(data.columns[21:31])
'''print(features_mean)
print("-----------------------------------")
print(features_se)
print("------------------------------------")
print(features_worst)'''

data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})

#print(data.describe())
print()

corr = data[features_mean].corr()
sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},xticklabels= features_mean, yticklabels= features_mean,cmap= 'coolwarm')

prediction_var = ['texture_mean','perimeter_mean','smoothness_mean','compactness_mean','symmetry_mean']

train, test = train_test_split(data, test_size = 0.3)
#print(train.shape)
#print(test.shape)

train_X = train[prediction_var]
train_y=train.diagnosis
test_X= test[prediction_var]
test_y =test.diagnosis

model=RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print("Random Forest Classifier")
print("ACCURACY - ",metrics.accuracy_score(prediction,test_y))

print()

model = svm.SVC()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print("SVM")
print("ACCURACY - ",metrics.accuracy_score(prediction,test_y))
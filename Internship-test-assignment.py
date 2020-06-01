#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


companies = pd.read_excel("Data_Science_Internship_Assignment.xlsx", sheet_name='Data')
companies.head()

companies.dtypes

companies.loc[0:10, ['LAUNCH DATE']]

type(companies['LAUNCH DATE'])
#Creating new date to make launch date of type date!

companies['new date'] = pd.to_datetime(companies['LAUNCH DATE'])

#dropping unnecessary columns

to_drop = ['WEBSITE', 'HQ REGION', 'HQ COUNTRY', 'HQ CITY', 'LINKEDIN', 'GROWTH STAGE', 'LINKEDIN']
companies.drop(to_drop, inplace=True, axis=1)
companies.head()

companies.dtypes

companies.tail(10)

# getting only the year of launch date, because i only need the year 
def test(x):
    if type(x) == str:
        return int(x[:4])
    else:
        return int(x)
companies['new date'] = companies['LAUNCH DATE'].apply(test)
companies.head(10)

companies['year_check'] = companies['new date'].apply(lambda x: 1 if x >= 1900 else 0)
companies.head(1000)

#Visualizing all the companies that are established before 1900 and after 1900 (maybe that can help me with the classification?)

X = companies['new date']
Y = companies['year_check']
plt.scatter(X, Y)
plt.show()

# Creating logisticreg classifier 

logreg_clf = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(companies[['new date']], companies['year_check'], test_size=0.3)

logreg_clf.fit(X_train,y_train)

y_pred=logreg_clf.predict(X_test)

#creating confusion metric to evaluate the model

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix
# printing the accuracy metric of the confusion metric

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# creating randomforset classifier 

clf=RandomForestClassifier(n_estimators=100)

clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# transforming the tags into arrays of word, the whole fit column now is array of arrays 

vectorizer = CountVectorizer(min_df=0, lowercase=False)
companies['fit'] = companies['TAGS'].apply(lambda x: vectorizer.fit(str(x).split(';')).vocabulary_ if x != 0 else []).tolist()
companies.tail(10)


X_train, X_test, y_train, y_test = train_test_split(companies[['fit']], companies['TYPE'], test_size=0.3)

logreg = LogisticRegression()
logreg.fit(X_train,y_train)

# creating tfidvectorizer to transform the tags column from array with strings to arrat with numbers

tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = tfidfconverter.fit_transform([' '.join(x) for x in tags]).toarray()
X

X_train, X_test, y_train, y_test = train_test_split(X, companies['TYPE'], test_size=0.3)

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

# From where i split the data into the array of tags and if i had the types i would try to predict them, but i don't know how to make the data labels.
#clf.fit(X_train,y_train)

#y_pred=clf.predict(X_test)

#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


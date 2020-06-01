#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[3]:


companies = pd.read_excel("Data_Science_Internship_Assignment.xlsx", sheet_name='Data')
companies.head()


# In[4]:


companies.dtypes


# In[5]:


companies.loc[0:10, ['LAUNCH DATE']]


# In[6]:


type(companies['LAUNCH DATE'])


# In[11]:


#Creating new date to make launch date of type date!


# In[7]:


companies['new date'] = pd.to_datetime(companies['LAUNCH DATE'])


# In[12]:


#dropping unnecessary columns


# In[8]:


to_drop = ['WEBSITE', 'HQ REGION', 'HQ COUNTRY', 'HQ CITY', 'LINKEDIN', 'GROWTH STAGE', 'LINKEDIN']
companies.drop(to_drop, inplace=True, axis=1)
companies.head()


# In[13]:


companies.dtypes


# In[14]:


companies.tail(10)


# In[16]:


# getting only the year of launch date, because i only need the year 


# In[20]:


def test(x):
    if type(x) == str:
        return int(x[:4])
    else:
        return int(x)
companies['new date'] = companies['LAUNCH DATE'].apply(test)
companies.head(10)


# In[21]:


companies['year_check'] = companies['new date'].apply(lambda x: 1 if x >= 1900 else 0)
companies.head(1000)


# In[23]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(companies[['new date']], companies['year_check'], test_size=0.3)


# In[24]:


X_test


# In[25]:


X_train


# In[27]:


#Visualizing all the companies that are established before 1900 and after 1900 (maybe that can help me with the classification?)


# In[26]:


import matplotlib.pyplot as plt

X = companies['new date']
Y = companies['year_check']
plt.scatter(X, Y)
plt.show()


# In[35]:


# Creating logisticreg classifier 


# In[30]:


from sklearn.linear_model import LogisticRegression
logreg_clf = LogisticRegression()


# In[31]:


logreg_clf.fit(X_train,y_train)


# In[32]:


y_pred=logreg_clf.predict(X_test)


# In[36]:


#creating confusion metric to evaluate the model


# In[33]:


from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[37]:


# printing the accuracy metric of the confusion metric


# In[34]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[41]:


# creating randomforset classifier 


# In[38]:


from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(n_estimators=100)

clf.fit(X_train,y_train)



# In[39]:


y_pred=clf.predict(X_test)


# In[40]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[45]:


# transforming the tags into arrays of word, the whole fit column now is array of arrays 


# In[44]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=0, lowercase=False)
companies['fit'] = companies['TAGS'].apply(lambda x: vectorizer.fit(str(x).split(';')).vocabulary_ if x != 0 else []).tolist()
companies.tail(10)


# In[46]:


X_train, X_test, y_train, y_test = train_test_split(companies[['fit']], companies['TYPE'], test_size=0.3)


# In[47]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[48]:


logreg.fit(X_train,y_train)


# In[51]:


# trying to frop the null values 


# In[49]:


tags = companies['fit'].dropna()


# In[50]:


tags


# In[55]:


# creating tfidvectorizer to transform the tags column from array with strings to arrat with numbers


# In[54]:


from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = tfidfconverter.fit_transform([' '.join(x) for x in tags]).toarray()
X


# In[61]:


X_train, X_test, y_train, y_test = train_test_split(X, companies['TYPE'], test_size=0.3)


# In[62]:


from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)


# In[64]:


# From where i split the data into the array of tags and if i had the types i would try to predict them, but i don't know how to make the data labels.


# In[63]:


clf.fit(X_train,y_train)


# In[59]:


y_pred=clf.predict(X_test)


# In[60]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


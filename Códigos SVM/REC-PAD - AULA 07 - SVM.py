#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


wine = pd.read_csv('wine.csv')
wine


# # Utilizando SVM

# In[6]:


X = wine.iloc[:,1:]
y = wine.iloc[:, 0]

X_teste = X.iloc[177]
X = X.drop(177)
y_teste = y.iloc[177]
y = y.drop(177)

y_teste


# In[9]:


from sklearn.svm import SVC

svc = SVC(kernel='rbf')
svc.fit(X, y)
svc.predict(np.array(X_teste).reshape(1,-1))


# In[10]:


svc = SVC(kernel='linear')
svc.fit(X, y)
svc.predict(np.array(X_teste).reshape(1,-1))


# In[11]:


svc = SVC(kernel='poly')
svc.fit(X, y)
svc.predict(np.array(X_teste).reshape(1,-1))


# In[12]:


svc = SVC(kernel='sigmoid')
svc.fit(X, y)
svc.predict(np.array(X_teste).reshape(1,-1))


# In[ ]:





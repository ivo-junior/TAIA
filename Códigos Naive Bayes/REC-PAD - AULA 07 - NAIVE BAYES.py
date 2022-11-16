#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


wine = pd.read_csv('wine.csv')
wine


# # Utilizando NB

# In[16]:


X = wine.iloc[:,1:]
y = wine.iloc[:, 0]

X_teste = X.iloc[177]
X = X.drop(177)
y_teste = y.iloc[177]
y = y.drop(177)

y_teste


# In[15]:


from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X, y)
gnb.predict(np.array(X_teste).reshape(1,-1))


# In[17]:


from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()
mnb.fit(X, y)
mnb.predict(np.array(X_teste).reshape(1,-1))


# ## Testando com bases categóricas

# In[19]:


car = pd.read_csv('car.csv')
car


# In[26]:


X = car.iloc[:,:6]
y = car.iloc[:, 6]

X_teste = X.iloc[1727]
X = X.drop(1727)
y_teste = y.iloc[1727]
y = y.drop(1727)


# In[24]:


gnb = GaussianNB()
gnb.fit(X, y)
gnb.predict(np.array(X_teste).reshape(1,-1))


# In[25]:


mnb = MultinomialNB()
mnb.fit(X, y)
mnb.predict(np.array(X_teste).reshape(1,-1))


# In[ ]:





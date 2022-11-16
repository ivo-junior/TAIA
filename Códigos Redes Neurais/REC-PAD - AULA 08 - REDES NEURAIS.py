#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


iris = pd.read_csv('iris.csv')
iris


# # Utilizando MLP

# In[3]:


X = iris.iloc[:,:4]
y = iris.iloc[:, 4]

X_teste = X.iloc[149]
X = X.drop(149)
y_teste = y.iloc[149]
y = y.drop(149)
y_teste


# In[9]:


from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(activation="tanh", hidden_layer_sizes=(10,))
mlp.fit(X, y)
mlp.predict(np.array(X_teste).reshape(1,-1))


# In[ ]:





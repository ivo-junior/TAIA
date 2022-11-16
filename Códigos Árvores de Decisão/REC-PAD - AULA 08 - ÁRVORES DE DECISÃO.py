#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd
import numpy as np


# In[55]:


iris = pd.read_csv('iris.csv')
iris


# # Utilizando Árvore de Decisão

# In[57]:


X = iris.iloc[:,:4]
y = iris.iloc[:, 4]

X_teste = X.iloc[149]
X = X.drop(149)
y_teste = y.iloc[149]
y = y.drop(149)
y_teste


# In[58]:


from sklearn.tree import DecisionTreeClassifier

arvore = DecisionTreeClassifier(criterion="entropy")
arvore.fit(X, y)
arvore.predict(np.array(X_teste).reshape(1,-1))


# In[59]:


from sklearn.tree import export_text
tree_rules = export_text(arvore, feature_names=list(X.columns))


# In[60]:


print(tree_rules)


# In[61]:


import matplotlib.pyplot as plt
from sklearn import tree
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(arvore, 
                   feature_names=X.columns,  
                   class_names=y,
                   filled=True)


# In[ ]:





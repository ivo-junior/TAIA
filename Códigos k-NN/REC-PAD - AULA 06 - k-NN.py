#!/usr/bin/env python
# coding: utf-8

# In[265]:


import pandas as pd
import numpy as np


# In[266]:


iris = pd.read_csv('iris.csv', header=None)
iris


# In[267]:


from sklearn.preprocessing import OrdinalEncoder
import copy

atributos = [4]
enc = OrdinalEncoder()
enc.fit(iris[atributos])
iris[atributos] = enc.transform(iris[atributos])

X = iris.iloc[:,:4]
y = iris.iloc[:, 4]


# In[268]:


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X.values)

print('Projecting %d-dimensional data to 2D' % X.values.shape[1])

plt.figure(figsize=(12,10))
plt.xlim([-4, 4])
plt.ylim([-4, 4])
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y.values, 
            edgecolor='none', alpha=0.7, s=40,
            cmap=plt.cm.get_cmap('nipy_spectral', 3))
plt.colorbar()
plt.title('PCA projection');


# In[269]:


y.iloc[149] = 4.0

X_reduced_2 = pca.fit_transform(X.values)

print('Projecting %d-dimensional data to 2D' % X.values.shape[1])

plt.figure(figsize=(12,10))
plt.xlim([-4, 4])
plt.ylim([-4, 4])
plt.scatter(X_reduced_2[:, 0], X_reduced_2[:, 1], c=y.values, 
            edgecolor='none', alpha=0.7, s=40,
            cmap=plt.cm.get_cmap('rainbow', 5))
plt.colorbar()
plt.title('PCA projection');


# In[270]:


y.iloc[123] = 3.0

X_reduced_3 = pca.fit_transform(X.values)

print('Projecting %d-dimensional data to 2D' % X.values.shape[1])

plt.figure(figsize=(12,10))
plt.xlim([-4, 4])
plt.ylim([-4, 4])
plt.scatter(X_reduced_3[:, 0], X_reduced_3[:, 1], c=y.values, 
            edgecolor='none', alpha=0.7, s=40,
            cmap=plt.cm.get_cmap('rainbow', 5))
plt.colorbar()
plt.title('PCA projection');


# In[271]:


y.iloc[72] = 3.0

X_reduced_3 = pca.fit_transform(X.values)

print('Projecting %d-dimensional data to 2D' % X.values.shape[1])

plt.figure(figsize=(12,10))
plt.xlim([-4, 4])
plt.ylim([-4, 4])
plt.scatter(X_reduced_3[:, 0], X_reduced_3[:, 1], c=y.values, 
            edgecolor='none', alpha=0.7, s=40,
            cmap=plt.cm.get_cmap('rainbow', 5))
plt.colorbar()
plt.title('PCA projection');


# In[272]:


y.iloc[83] = 3.0

X_reduced_3 = pca.fit_transform(X.values)

print('Projecting %d-dimensional data to 2D' % X.values.shape[1])

plt.figure(figsize=(12,10))
plt.xlim([-4, 4])
plt.ylim([-4, 4])
plt.scatter(X_reduced_3[:, 0], X_reduced_3[:, 1], c=y.values, 
            edgecolor='none', alpha=0.7, s=40,
            cmap=plt.cm.get_cmap('rainbow', 5))
plt.colorbar()
plt.title('PCA projection');


# # Utilização do k-NN

# ### Sem PCA

# In[273]:


iris = pd.read_csv('iris.csv', header=None)

X = iris.iloc[:,:4]
y = iris[4]

X_teste = X.iloc[149]
X = X.drop(149)
y_teste = y.iloc[149]
y = y.drop(149)


# In[274]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X, y)
knn.predict(np.array(X_teste).reshape(1,-1))


# In[275]:


dist, vizinhos = knn.kneighbors(np.array(X_teste).reshape(1,-1))
classes = []
for v in vizinhos:
    classes.append(iris.iloc[:-1, 4][v])
classes


# ### Com PCA

# In[276]:


from sklearn.decomposition import PCA

iris = pd.read_csv('iris.csv', header=None)

X = iris.iloc[:,:4]
y = iris[4]

X_pca = pd.DataFrame(pca.fit_transform(X.values))

X_teste = X_pca.iloc[149]
X_pca = X_pca.drop(149)
y_teste = y.iloc[149]
y = y.drop(149)


# In[277]:


knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_pca, y)
knn.predict(np.array(X_teste).reshape(1,-1))


# In[278]:


dist, vizinhos = knn.kneighbors(np.array(X_teste).reshape(1,-1))
classes = []
for v in vizinhos:
    classes.append(iris.iloc[:-1, 4][v])
classes


# In[279]:


y_teste


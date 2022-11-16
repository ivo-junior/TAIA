#!/usr/bin/env python
# coding: utf-8

# In[72]:


import pandas as pd
import numpy as np


# In[73]:


wine = pd.read_csv('wine.csv')
wine


# In[76]:


X = wine.iloc[:,1:]
y = wine.iloc[:,0]
y


# # Holdout

# In[77]:


from sklearn.model_selection import train_test_split

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42)
X_treino


# In[78]:


from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(activation="relu", hidden_layer_sizes=(10,))
mlp.fit(X_treino, y_treino)
mlp.predict(X_teste)


# # K-fold

# In[79]:


from sklearn.model_selection import KFold

resultados = []
kf = KFold(n_splits=10, shuffle=True)
for index_treino, index_teste in kf.split(X):
    #X_treino, X_teste = X[index_treino], X[index_teste]
    #y_treino, y_teste = y[index_treino], y[index_teste]
    X_treino, X_teste = X.iloc[index_treino], X.iloc[index_teste]
    y_treino, y_teste = y.iloc[index_treino], y.iloc[index_teste]
    mlp.fit(X_treino, y_treino)
    resultados.append(mlp.predict(X_teste))
resultados


# # Métricas

# In[80]:


X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42)

mlp = MLPClassifier(activation="relu", hidden_layer_sizes=(10,))
mlp.fit(X_treino, y_treino)
mlp.score(X_teste, y_teste)


# In[81]:


from sklearn.metrics import precision_score, recall_score, f1_score

y_pred = mlp.predict(X_teste)

precision_score(y_teste, y_pred, average='macro')


# In[82]:


recall_score(y_teste, y_pred, average='macro')


# In[83]:


f1_score(y_teste, y_pred, average='macro')


# # Prática para o projeto

# ### Estimação dos parâmetros (kNN, SVM, MLP)

# In[84]:


from sklearn.neighbors import KNeighborsClassifier

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42)

valores_k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
resultados_knn = []

for k in valores_k:
    resultados_k = []
    knn = KNeighborsClassifier(n_neighbors=k)
    kf = KFold(n_splits=10, shuffle=True)
    for index_train, index_valid in kf.split(X_treino):
        X_train, X_validacao = X_treino.iloc[index_train], X_treino.iloc[index_valid]
        y_train, y_validacao = y_treino.iloc[index_train], y_treino.iloc[index_valid]
        knn.fit(X_train, y_train)
        resultados_k.append(knn.score(X_validacao, y_validacao))
    resultados_knn.append(sum(resultados_k)/len(resultados_k))
resultados_knn


# In[85]:


knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_treino, y_treino)
knn.score(X_teste, y_teste)


# In[86]:


from sklearn.metrics import precision_score, recall_score, f1_score
y_pred = knn.predict(X_teste)

precision_score(y_teste, y_pred, average='macro')


# In[87]:


recall_score(y_teste, y_pred, average='macro')


# In[88]:


f1_score(y_teste, y_pred, average='macro')


# In[99]:


from sklearn.svm import SVC
valores_kernel = ["linear", "poly", "rbf", "sigmoid"]
resultados_svm = []

for kernel in valores_kernel:
    resultados_kernel = []
    svc = SVC(kernel=kernel)
    kf = KFold(n_splits=10, shuffle=True)
    for index_train, index_valid in kf.split(X_treino):
        X_train, X_validacao = X_treino.iloc[index_train], X_treino.iloc[index_valid]
        y_train, y_validacao = y_treino.iloc[index_train], y_treino.iloc[index_valid]
        svc.fit(X_train, y_train)
        resultados_kernel.append(svc.score(X_validacao, y_validacao))
    resultados_svm.append(sum(resultados_kernel)/len(resultados_kernel))
resultados_svm


# In[101]:


svc = SVC(kernel="linear")
svc.fit(X_treino, y_treino)
svc.score(X_teste, y_teste)


# In[103]:


from sklearn.neural_network import MLPClassifier
valores_ativacao = ["identity", "logistic", "tanh", "relu"]
resultados_mlp = []

for ativacao in valores_ativacao:
    resultados_ativacao = []
    mlp = MLPClassifier(activation=ativacao)
    kf = KFold(n_splits=10, shuffle=True)
    for index_train, index_valid in kf.split(X_treino):
        X_train, X_validacao = X_treino.iloc[index_train], X_treino.iloc[index_valid]
        y_train, y_validacao = y_treino.iloc[index_train], y_treino.iloc[index_valid]
        mlp.fit(X_train, y_train)
        resultados_ativacao.append(svc.score(X_validacao, y_validacao))
    resultados_mlp.append(sum(resultados_ativacao)/len(resultados_ativacao))
resultados_mlp


# In[111]:


mlp = MLPClassifier(activation="identity")
mlp.fit(X_treino, y_treino)
mlp.score(X_teste, y_teste)


# In[ ]:





# In[ ]:





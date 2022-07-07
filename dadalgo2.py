#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import required libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# Keras specific
import keras
from keras.models import Sequential
from keras.layers import Dense


# In[3]:


df = pd.read_csv("XRPUSDT-8.csv")
df.shape
df.head
df = df.drop('Time',axis=1)
df = df/df.max()
df


# In[4]:


length = 10


# In[5]:


df4 = df[:-length]
df4 = np.array(df4)
for i in range(length-1):
    df2 = df[i+1:i+1-length]
    df2 = np.array(df2)
    df4 = np.concatenate((df4,df2),axis=1)
x = pd.DataFrame(df4)
x


# In[6]:


Y = pd.DataFrame(np.array(df[i:i-length]), columns = list(df.columns) )
Y


# In[7]:


y = Y['High'].values
X = x.values


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01,shuffle = False)
print(X_train.shape)
print(X_test.shape)


# In[9]:


model = Sequential()
model.add(Dense(500, activation='linear', input_dim=50))
model.add(Dense(500, activation='linear'))
model.add(Dense(500, activation='linear'))
model.add(Dense(1, activation='linear'))
# Compile the model
model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])


# In[10]:


model.fit(X_train, y_train, epochs=50)


# In[14]:


pred_train = model.predict(X_train)
scores = model.evaluate(X_train, y_train, verbose=0)
print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))   
 
pred_test= model.predict(X_test)
scores2 = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))    


# In[15]:


import matplotlib.pyplot as plt
plt.plot(pred_test[100:120],color="r")
plt.plot(y_test[100:120])
#plt.plot(pred_test+0.0085,color='r')
#plt.plot(y_test)
plt.show()


# In[23]:


df1 = pd.read_csv("XRPUSDT-8.csv")
df1 = df1.drop('Time',axis=1)
df2 = pd.read_csv("XRPUSDT-1m-2021-09-01.csv")
df2 = df2.drop('Time',axis=1)
df2 = df2/df1.max()
df2


# In[41]:


X_train.shape


# In[50]:


model.predict(np.array([np.concatenate(dfa[:length])]))[0]


# In[53]:


dfa = np.array(df2)


# In[54]:


for i in range(df2.shape[0]-length):
    dfa[i+length][1] = model.predict(np.array([np.concatenate(dfa[i:length+i])]))[0]


# In[71]:


dfk = pd.DataFrame(dfa)
plt.plot(dfk[1][9:50])
plt.plot(df2["High"][9:50],color = "r")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





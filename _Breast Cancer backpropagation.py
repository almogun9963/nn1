#!/usr/bin/env python
# coding: utf-8

# In[1]:


#used code from https://github.com/pulunghendroprastyo/ann_chiSquare


# In[2]:


#imports
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical
from keras.optimizers import adam 
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


# In[3]:


#Dataset
data=pd.read_csv('wpbc.data')
data = data.round(4)
how_much = data.iloc[: , 1].values
temporary = []
for num in how_much:
    if num == "R":
        temporary.append(1)
    else:
        temporary.append(0)
data = data.drop(data.columns[1], axis=1)
temp = pd.DataFrame(data=temporary, columns=['N'])
data = pd.concat([data ,temp], axis=1)

# Delete the lines with "?" in their features
data = data.drop(data.index[5])
data = data.drop(data.index[26])
data = data.drop(data.index[82])
data = data.drop(data.index[192])


# In[4]:


#checking the size of the data
data.shape


# In[5]:


# setting up y label.
y = data[['N']]
y.head(5)


# In[6]:


# setting up x label, aka the features.
X = data.drop(['N'],axis=1)
X.head(5)


# In[7]:


#spliting
x_train, x_test, y_train, y_test = train_test_split(X ,y ,test_size=0.33,random_state=0)
y_train = to_categorical(y_train,num_classes=2)


# In[8]:


# the algorithm backpropagation
def backpropagation(x_train,y_train,epochs=150,batch_size=10):
    start_time = datetime.now()
    inputSize = x_train.shape[1]
    model = Sequential()
    model.add(Dense(units=5, input_dim=inputSize)) 
    model.add(Activation('sigmoid'))
    model.add(Dense(units=2)) 
    model.add(Activation('sigmoid'))
    model.compile(loss='categorical_crossentropy',optimizer=adam(lr=0.001),metrics=['accuracy'])
    
    #train NN
    mlp=model.fit(x_train, y_train,epochs=epochs, batch_size=batch_size)
    end_time = datetime.now()
    result_time  =end_time-start_time
    ans = model.predict_classes(x_test,batch_size=1)
    return result_time, ans

    


# In[9]:


time, ans = backpropagation(x_train,y_train)


# In[10]:


print("Duration:",time)


# In[11]:


#getting the TRUE NEGATIVE,FALSE NEGATIVE,FALSE POSITIVE
CM = confusion_matrix(ans,y_test)
TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]

print ("TRUE NEGATIVE (TN):",TN)
print ("FALSE NEGATIVE (FN):",FN)
print ("TRUE POSITIVE (TP):",TP)
print ("FALSE POSITIVE (FP):",FP)
print(classification_report(ans,y_test))


# In[12]:



# create the model
knn = KNeighborsClassifier(n_neighbors=1)

# Enter the data training to the mode
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
# print(metrics.accuracy_score(y_test, y_pred))

# using cross validation to get the best percentage of success

k_range = list(range(1, 41))
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance', p=1)
    scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)
# graph of the value of k (x) and the cross validation accuracy
plt.plot(k_range, k_scores)
# names of x and y
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross validation Accuracy')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





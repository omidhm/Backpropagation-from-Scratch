#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import operator
import xlsxwriter


# In[2]:


# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')

# X_train = X_train / 255 # 28*28 = 784 neuron
# X_test = X_test / 255
# one hot encode outputs.

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]





# In[3]:


# normalize inputs from 0-255 to 0-1
X_train = X_train / 255 # 28*28 = 784 neuron
X_test = X_test / 255


# In[5]:


W = np.random.rand(10,784)


# In[6]:


#initializition:
η = 1
e = 0
n = 50
W = np.random.rand(10,784)
epoch = 1
err = np.zeros((100000,1))
TrainingErr = np.zeros((100000,1))


# In[7]:


# Temp = np.dot(W, X_train[1,:]) 
# print(Temp)
# print(np.size(Temp))
# print(Temp.shape)

# Temp2 = np.matmul(W, X_train[1,:]) 
# print(Temp2)
# print(np.size(Temp2))
# Temp2.shape


# In[8]:



max(enumerate(np.matrix(v)), key = operator.itemgetter(1))
type(y_train[(i)])


# In[9]:


j = 0
Temp = 1

while  Temp > e:
    
    for i in range(n):                                                   # We will count the misclassification Error
        v = np.dot(W, X_train[i,:])                                     # The v is 10 row and 1 column!

        index, value = max(enumerate(v), key = operator.itemgetter(1))
    #         print(value)
    #         print(np.size(value))
    #         print(index)
    #         print(np.size(index))
        indexy, valuey = max(enumerate(y_train[(i)]), key = operator.itemgetter(1))
    #         print(indexy,valuey)
    #         print(np.size(index))
    #         index, value = max(enumerate(v), key=operator.itemgetter(1))
        if index != indexy:
            err[(epoch)] = err[(epoch)] + 1
        if i == n:
            print(err[(epoch)])


    epoch = epoch + 1
    print(epoch)
        
    for i in range(n):
        indexy, valuey = max(enumerate(y_train[(i)]), key = operator.itemgetter(1))
        d = np.zeros(10)
        d[indexy] = 1
        W = W + η * np.matmul(np.expand_dims((d - np.heaviside(np.dot(W, X_train[i,:]),0)), axis=-1), np.expand_dims(X_train[i,:], axis=0))
      
    TrainingErr[(j)] = err[epoch-1]/n
    Temp = TrainingErr[j]
    j = j + 1
    
    print("Training Error " , Temp)


# In[10]:


np.dot(W, X_train[i,:])


# In[26]:


plt.xlabel('The epoch number')
plt.ylabel('The number of misclassification errors ')
plt.title(' Plot the epoch number vs. the number of\n misclassification errors (including epoch 0).')


for i in range(epoch):
    plt.plot(i,TrainingErr[i]*n, 'bo', (i,i+1),(TrainingErr[i]*n,TrainingErr[i+1]*n), 'k')
    

# axes.plot(x,x**2, marker='o')

plt.xlim([-.1,7])
plt.ylim([-0.1,100])


# In[27]:


errt = np.zeros((100000,1))
epocht = 1
m = 10000
for i in range(m):                                                 
    v = np.dot(W, X_test[i,:])                                     
    index, value = max(enumerate(v), key = operator.itemgetter(1))
    indexy, valuey = max(enumerate(y_test[(i)]), key = operator.itemgetter(1))
    if index != indexy:
        errt[(i)] = errt[(i)] + 1

print(errt)
a,b = np.nonzero(errt)
print(a.shape)

#Missclassification will be:
indexe,valuee = np.nonzero(errt)
ErrPercentage = (np.size(indexe))/m

print("ErrPercentage : ", ErrPercentage)


# Well, The number of training samples are very small, so our trained function fit(somehow overfit) to them. 
# as aresult we had 0 error classification over there. So this model is not a good one for generalization to 
# test data which are also a lot.

# (g)

# In[32]:


#initializition:
η = 1
e = 0
n = 1000
W = np.random.rand(10,784)
epoch = 1
err = np.zeros((100000,1))
TrainingErr = np.zeros((100000,1))

j = 0
Temp = 1

while  Temp > e:
    
    for i in range(n):                                                   # We will count the misclassification Error
        v = np.dot(W, X_train[i,:])                                     # The v is 10 row and 1 column!

        index, value = max(enumerate(v), key = operator.itemgetter(1))
    #         print(value)
    #         print(np.size(value))
    #         print(index)
    #         print(np.size(index))
        indexy, valuey = max(enumerate(y_train[(i)]), key = operator.itemgetter(1))
    #         print(indexy,valuey)
    #         print(np.size(index))
    #         index, value = max(enumerate(v), key=operator.itemgetter(1))
        if index != indexy:
            err[(epoch)] = err[(epoch)] + 1
        if i == n:
            print(err[(epoch)])


    epoch = epoch + 1
    print(epoch)
        
    for i in range(n):
        indexy, valuey = max(enumerate(y_train[(i)]), key = operator.itemgetter(1))
        d = np.zeros(10)
        d[indexy] = 1
        W = W + η * np.matmul(np.expand_dims((d - np.heaviside(np.dot(W, X_train[i,:]),0)), axis=-1), np.expand_dims(X_train[i,:], axis=0))
      
    TrainingErr[(j)] = err[epoch-1]/n
    Temp = TrainingErr[j]
    j = j + 1
    
    print("Training Error " , Temp)
    

j = 0
Temp = 1

while  Temp > e:
    
    for i in range(n):                                                   # We will count the misclassification Error
        v = np.dot(W, X_train[i,:])                                     # The v is 10 row and 1 column!

        index, value = max(enumerate(v), key = operator.itemgetter(1))
    #         print(value)
    #         print(np.size(value))
    #         print(index)
    #         print(np.size(index))
        indexy, valuey = max(enumerate(y_train[(i)]), key = operator.itemgetter(1))
    #         print(indexy,valuey)
    #         print(np.size(index))
    #         index, value = max(enumerate(v), key=operator.itemgetter(1))
        if index != indexy:
            err[(epoch)] = err[(epoch)] + 1
        if i == n:
            print(err[(epoch)])


    epoch = epoch + 1
    print(epoch)
        
    for i in range(n):
        indexy, valuey = max(enumerate(y_train[(i)]), key = operator.itemgetter(1))
        d = np.zeros(10)
        d[indexy] = 1
        W = W + η * np.matmul(np.expand_dims((d - np.heaviside(np.dot(W, X_train[i,:]),0)), axis=-1), np.expand_dims(X_train[i,:], axis=0))
      
    TrainingErr[(j)] = err[epoch-1]/n
    Temp = TrainingErr[j]
    j = j + 1
    
    print("Training Error " , Temp)

plt.xlabel('The epoch number')
plt.ylabel('The number of misclassification errors ')
plt.title(' Plot the epoch number vs. the number of\n misclassification errors (including epoch 0).\n n=1000')


for i in range(epoch):
    plt.plot(i,TrainingErr[i]*n, 'bo', (i,i+1),(TrainingErr[i]*n,TrainingErr[i+1]*n), 'k')
    

# axes.plot(x,x**2, marker='o')

plt.xlim([-.1,50])
plt.ylim([-0.1,200])
    
errt = np.zeros((100000,1))
epocht = 1
m = 10000
for i in range(m):                                                 
    v = np.dot(W, X_test[i,:])                                     
    index, value = max(enumerate(v), key = operator.itemgetter(1))
    indexy, valuey = max(enumerate(y_test[(i)]), key = operator.itemgetter(1))
    if index != indexy:
        errt[(i)] = errt[(i)] + 1

print(errt)
a,b = np.nonzero(errt)
print(a.shape)

#Missclassification will be:
indexe,valuee = np.nonzero(errt)
ErrPercentage = (np.size(indexe))/m

print("ErrPercentage : ", ErrPercentage)


# ErrPercentage is equal to  0.1735 in n=1000. so the model become better for other data which has been not shown. 
# The reason is the high number of data in learning process in comparison to the previous one. Generally when we have
# 10*784 weights, we should have large number of training set, otherwise, our model will overfit and can not be generalaized to other
# new datas.

# (h) Run Step (d) for n = 60000 and ϵ = 0. Make note of (i.e., plot) the errors as the number of
# epochs grow large, and note that the algorithm may not converge. Comment on the results.

# In[34]:


#initializition:
η = 1
e = 0
n = 60000
W = np.random.rand(10,784)
epoch = 1
err = np.zeros((100000,1))
TrainingErr = np.zeros((100000,1))

j = 0
Temp = 1

while  Temp > e:
    
    for i in range(n):                                                   # We will count the misclassification Error
        v = np.dot(W, X_train[i,:])                                     # The v is 10 row and 1 column!

        index, value = max(enumerate(v), key = operator.itemgetter(1))
    #         print(value)
    #         print(np.size(value))
    #         print(index)
    #         print(np.size(index))
        indexy, valuey = max(enumerate(y_train[(i)]), key = operator.itemgetter(1))
    #         print(indexy,valuey)
    #         print(np.size(index))
    #         index, value = max(enumerate(v), key=operator.itemgetter(1))
        if index != indexy:
            err[(epoch)] = err[(epoch)] + 1
        if i == n:
            print(err[(epoch)])


    epoch = epoch + 1
    print(epoch)
        
    for i in range(n):
        indexy, valuey = max(enumerate(y_train[(i)]), key = operator.itemgetter(1))
        d = np.zeros(10)
        d[indexy] = 1
        W = W + η * np.matmul(np.expand_dims((d - np.heaviside(np.dot(W, X_train[i,:]),0)), axis=-1), np.expand_dims(X_train[i,:], axis=0))
      
    TrainingErr[(j)] = err[epoch-1]/n
    Temp = TrainingErr[j]
    j = j + 1
    
    print("Training Error " , Temp)
    

j = 0
Temp = 1

while  Temp > e:
    
    for i in range(n):                                                   # We will count the misclassification Error
        v = np.dot(W, X_train[i,:])                                     # The v is 10 row and 1 column!

        index, value = max(enumerate(v), key = operator.itemgetter(1))
    #         print(value)
    #         print(np.size(value))
    #         print(index)
    #         print(np.size(index))
        indexy, valuey = max(enumerate(y_train[(i)]), key = operator.itemgetter(1))
    #         print(indexy,valuey)
    #         print(np.size(index))
    #         index, value = max(enumerate(v), key=operator.itemgetter(1))
        if index != indexy:
            err[(epoch)] = err[(epoch)] + 1
        if i == n:
            print(err[(epoch)])


    epoch = epoch + 1
    print(epoch)
        
    for i in range(n):
        indexy, valuey = max(enumerate(y_train[(i)]), key = operator.itemgetter(1))
        d = np.zeros(10)
        d[indexy] = 1
        W = W + η * np.matmul(np.expand_dims((d - np.heaviside(np.dot(W, X_train[i,:]),0)), axis=-1), np.expand_dims(X_train[i,:], axis=0))
      
    TrainingErr[(j)] = err[epoch-1]/n
    Temp = TrainingErr[j]
    j = j + 1
    
    print("Training Error " , Temp)

plt.xlabel('The epoch number')
plt.ylabel('The number of misclassification errors ')
plt.title(' Plot the epoch number vs. the number of\n misclassification errors (including epoch 0).\n n=1000')


for i in range(epoch):
    plt.plot(i,TrainingErr[i]*n, 'bo', (i,i+1),(TrainingErr[i]*n,TrainingErr[i+1]*n), 'k')
    

# axes.plot(x,x**2, marker='o')

plt.xlim([-.1,300])
plt.ylim([-0.1,20000])
    
errt = np.zeros((100000,1))
epocht = 1
m = 10000
for i in range(m):                                                 
    v = np.dot(W, X_test[i,:])                                     
    index, value = max(enumerate(v), key = operator.itemgetter(1))
    indexy, valuey = max(enumerate(y_test[(i)]), key = operator.itemgetter(1))
    if index != indexy:
        errt[(i)] = errt[(i)] + 1

print(errt)
a,b = np.nonzero(errt)
print(a.shape)

#Missclassification will be:
indexe,valuee = np.nonzero(errt)
ErrPercentage = (np.size(indexe))/m

print("ErrPercentage : ", ErrPercentage)


# As you may have seen the model will not converge and the error for training set fluctuate around .13,.12 and .15.

# In[43]:



plt.xlabel('The epoch number')
plt.ylabel('The number of misclassification errors ')
plt.title(' Plot the epoch number vs. the number of\n misclassification errors (including epoch 0).\n n=60000')


for i in range(epoch):
    plt.plot(i,TrainingErr[i]*n, 'bo', (i,i+1),(TrainingErr[i]*n,TrainingErr[i+1]*n), 'k')
    

# axes.plot(x,x**2, marker='o')

plt.xlim([-.1,250])
plt.ylim([-0.1,20000])
    
errt = np.zeros((100000,1))
epocht = 1
m = 10000
for i in range(m):                                                 
    v = np.dot(W, X_test[i,:])                                     
    index, value = max(enumerate(v), key = operator.itemgetter(1))
    indexy, valuey = max(enumerate(y_test[(i)]), key = operator.itemgetter(1))
    if index != indexy:
        errt[(i)] = errt[(i)] + 1

print(errt)
a,b = np.nonzero(errt)
print(a.shape)

#Missclassification will be:
indexe,valuee = np.nonzero(errt)
ErrPercentage = (np.size(indexe))/m

print("ErrPercentage : ", ErrPercentage)


# (i) Using your observations in the previous step, pick some appropriate value for ϵ (such that your
# algorithm in (d) will eventually terminate). Repeat the following two subitems three times with
# different initial weights and comment on the results:
# • Run Step (d) for n = 60000, some η of your choice and the ϵ you picked.
# • Run Step (e) to with the W you obtained in the previous step.
# 

# In[59]:


from random import random
#initializition:
η = 10
e = .08
n = 60000
W = np.random.rand(10,784)
epoch = 1
err = np.zeros((100000,1))
TrainingErr = np.zeros((100000,1))


j = 0
Temp = 1

while  Temp > e:
    
    for i in range(n):                                                   # We will count the misclassification Error
        v = np.dot(W, X_train[i,:])                                     # The v is 10 row and 1 column!

        index, value = max(enumerate(v), key = operator.itemgetter(1))
    #         print(value)
    #         print(np.size(value))
    #         print(index)
    #         print(np.size(index))
        indexy, valuey = max(enumerate(y_train[(i)]), key = operator.itemgetter(1))
    #         print(indexy,valuey)
    #         print(np.size(index))
    #         index, value = max(enumerate(v), key=operator.itemgetter(1))
        if index != indexy:
            err[(epoch)] = err[(epoch)] + 1
        if i == n:
            print(err[(epoch)])


    epoch = epoch + 1
    print(epoch)
    
    if Temp > err[epoch-1]/n:
        η = 10 * random()
    if Temp < err[epoch-1]/n:
        η = 200 * random()
    
        
    for i in range(n):
        indexy, valuey = max(enumerate(y_train[(i)]), key = operator.itemgetter(1))
        d = np.zeros(10)
        d[indexy] = 1
        W = W + η * np.matmul(np.expand_dims((d - np.heaviside(np.dot(W, X_train[i,:]),0)), axis=-1), np.expand_dims(X_train[i,:], axis=0))
      
    TrainingErr[(j)] = err[epoch-1]/n
    Temp = TrainingErr[j]
    j = j + 1
    
    print("Training Error " , Temp)


# In[64]:



plt.xlabel('The epoch number')
plt.ylabel('The number of misclassification errors ')
plt.title(' Plot the epoch number vs. the number of\n misclassification errors (including epoch 0).\n n=60000\n ')


for i in range(epoch-2):
    plt.plot(i,TrainingErr[i]*n, 'bo', (i,i+1),(TrainingErr[i]*n,TrainingErr[i+1]*n), 'k')
    

# axes.plot(x,x**2, marker='o')

plt.xlim([-.1,50])
plt.ylim([-0.1,20000])
    
errt = np.zeros((100000,1))
epocht = 1
m = 10000
for i in range(m):                                                 
    v = np.dot(W, X_test[i,:])                                     
    index, value = max(enumerate(v), key = operator.itemgetter(1))
    indexy, valuey = max(enumerate(y_test[(i)]), key = operator.itemgetter(1))
    if index != indexy:
        errt[(i)] = errt[(i)] + 1

print(errt)
a,b = np.nonzero(errt)
print(a.shape)

#Missclassification will be:
indexe,valuee = np.nonzero(errt)
ErrPercentage = (np.size(indexe))/m

print("ErrPercentage : ", ErrPercentage)


# NOTE :     if Temp > err[epoch-1]/n:
#         η = 10 * random()
#     if Temp < err[epoch-1]/n:
#         η = 200 * random()
#         
# As you see, I did not specify η, instead select it based on the flow. If we are reaching good result, η getting smaller
# otherwise, η getting bigger.
# As you can see the trainingerror in 32 epoch become .79 and testing error is .09 
# 

# In[ ]:





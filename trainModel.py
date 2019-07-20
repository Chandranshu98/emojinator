#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from keras.layers import Dense, Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout
from keras.utils import np_utils, print_summary
from keras.models import Sequential 
from keras.callbacks import ModelCheckpoint
import pandas as pd


# In[2]:


data = pd.read_csv("train_foo.csv")
dataset = np.array(data)
np.random.shuffle(dataset)
X=dataset
Y=dataset
X= X[:, 1:2501]
Y=Y[:, 0]


# In[5]:


X_train = X[0:12000, :]
X_train = X_train/255.
X_test = X[12000:13201 , :]
X_test = X_test/255.

Y=Y.reshape(Y.shape[0], 1)
Y_train = Y[0:12000, :]
Y_train = Y_train.T
Y_test = Y[12000:13201 , :]
Y_test = Y_test.T


# In[6]:


print("no of training samples : " + str(X_train.shape[0]))
print("no of test samples : " + str(X_test.shape[0]))
print("X_train shape" + str(X_train.shape))
print("Y_train shape" + str(Y_train.shape))
print("X_test shape" + str(X_test.shape))
print("Y_test shape" + str(Y_test.shape))


# In[8]:


image_x = 50
image_y = 50
train_y =np_utils.to_categorical(Y_train)
test_y = np_utils.to_categorical (Y_test)
train_y = train_y.reshape(train_y.shape[1], train_y.shape[2])
test_y = test_y.reshape(test_y.shape[1], test_y.shape[2])
X_train = X_train.reshape(X_train.shape[0], image_x, image_y, 1)
X_test = X_test.reshape(X_test.shape[0], image_x, image_y, 1)
print("X_train shape" + str(X_train.shape))
print("X_test shape" + str(X_test.shape))
print(" Y train shape = " + str(train_y.shape))


# In[9]:


def keras_model( image_x, image_y):
    num_of_classes = 12
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5,5), input_shape=(image_x, image_y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2), padding='same'))
    model.add(Conv2D(64, (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5,5), strides=(5, 5), padding='same'))
    model.add(Flatten())
    model.add(Dense(1024, activation = 'relu'))
    model.add(Dropout(0.6))
    model.add(Dense( num_of_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath = "face-rec_256.h5"
    checkpoint1 = ModelCheckpoint("_filepath_", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list =[checkpoint1]

    return model, callbacks_list


# In[10]:


model, callbacks_list = keras_model(image_x, image_y)
model.fit(X_train, train_y, validation_data=(X_test, test_y), epochs=5, batch_size=64, callbacks=callbacks_list)
scores  = model.evaluate(X_test, test_y, verbose=0)
print("CNN error: %.2ff%%" % (100-scores[1]*100))
print_summary(model)
model.save('face-rec_256.h5')


# In[ ]:





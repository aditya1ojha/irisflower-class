# Importing the libraries
import numpy as np
import pandas as pd
from keras.models import Model 
from keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical


# Loading the dataset
data = pd.read_csv('iris.data').values


# Initializing the training data
X_train = data[:, :4]
y_train = data[:, 4]


# Creating an empty dictionary
labels={}
cnt=0


# Assigning an integer value to each class (Working on the dictionary)
for i in y_train:
	if i not in labels:
		labels[i]=cnt
		cnt+=1


# Replacing the string values of the classes with integer values (Working on the training data)
for i in range(y_train.shape[0]):
	y_train[i]=labels[y_train[i]]


# Converting the class vector to binary class matrix
y_train = to_categorical(y_train)


# Converting the training data to numpy arrays
X_train = np.array(X_train, dtype="float64")
y_train = np.array(y_train, dtype="float64")


# Training the data
inp = Input(shape=(4))
mid = Dense(64, activation = "relu")(inp)
op = Dense(3, activation = "softmax")(mid)
model = Model(inputs = inp, outputs = op)
model.compile(optimizer = "rmsprop", loss = "categorical_crossentropy", metrics=['acc'])
model.fit(X_train, y_train, epochs = 25)


# Saving the trained data (to be used in the app)
model.save('model.h5')


# Saving the classes (to be used in the app)
arr = []
for k in labels.keys():
	arr.append(k)
	
np.save("labels.npy", np.array(arr))
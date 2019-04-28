import numpy as np
import cv2
from imageai.Prediction.Custom import CustomImagePrediction
import os
import webbrowser
import sys
import csv
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense

dataset = pd.read_csv('/Users/digvijayghotane/Desktop/Projects/mood_based_recommendation_system/Beta/movies.csv')
X = dataset.iloc[:,0].values
y = dataset.iloc[:,1].values
p = X.shape
n = y.shape
#label_encoder = LabelEncoder()
#X = label_encoder.fit_transform(X)
#y =label_encoder.fit_transform(y)
X = X.reshape(-1,1)
y = y.reshape(-1,1)
print('\n')
print(X)
print(y)
onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()
y = onehotencoder.fit_transform(y).toarray()
print('\n')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    #Initializing Neural Network
classifier = Sequential()
    # Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 3))
    # Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    # Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    # Compiling Neural Network
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    # Fitting our model 
#X_train = X_train.reshape(1,)
#y_train = y_train.reshape(1,)
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
y_pred = classifier.predict(X_test)
print(y_pred)
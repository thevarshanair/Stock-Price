# Recurrent Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')

#.iloc[] is primarily integer position based (from 0 to length-1 of the axis),
# but may also be used with a boolean array.
features = ['Open', 'High', 'Low']
training_set = dataset_train.loc[:, features].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler

# MinMaxScaler : Transforms features by scaling each feature to a given range.
# This estimator scales and translates each feature individually
# such that it is in the given range on the training set, e.g. between zero and one.
 
sc = MinMaxScaler(feature_range=(0,1))

#Fit to data, then transform it.
#Fits transformer to X and y with optional parameters fit_params and returns a transformed version of X.
# X : numpy array of shape [n_samples, n_features]: Training set.
# y : numpy array of shape [n_samples] : Target values.

training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 time stamps and 1 output
X_train=[]
Y_train=[]
for i in range(60,1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    Y_train.append(training_set_scaled[i,0])
X_train, Y_train = np.array(X_train), np.array(Y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 60, return_sequences = True, input_shape = (X_train.shape[1],1)))
regressor.add(Dropout(0.8))

# Adding the 2nd LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 60 , return_sequences = True))
regressor.add(Dropout(0.8))

# Adding the 3rd LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 60 , return_sequences = True))
regressor.add(Dropout(0.8))

# Adding the 4th LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 60 ))
regressor.add(Dropout(0.8))

# Adding the 5th LSTM layer and some Dropout regularization
#regressor.add(LSTM(units = 60))
#regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'binary_crossentropy')

# Fitting the RNN to the Training set
regressor.fit(X_train, Y_train, batch_size = 32 , epochs = 100)

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2018(SEPT-OCT)
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
#
real_stock_price = dataset_test.iloc[:,1:2].values

# Getting the predicted stock price of 2018(SEPT-OCT)
dataset_total = pd.concat((dataset_train[features], dataset_test[features]), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test)-60:].values
#np.reshape: Gives a new shape to an array without changing its data.- input : array to be reshapes, new shape
#inputs = inputs.reshape(-1, 1)
#transform: Scaling features of X according to feature_range. 
#Parameters:	X : array-like, shape [n_samples, n_features] Input data that will be transformed.
# Getting the predicted stock price of 2018(SEPT-OCT)

#inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test=[]
for i in range(60,80):
    X_test.append(training_set_scaled[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = np.append(predicted_stock_price, np.zeros((len(dataset_test), len(features) - 1)), axis = 1)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)

predicted_stock_price = np.delete(predicted_stock_price, [1, 2], axis = 1)

r2_score(real_stock_price, predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

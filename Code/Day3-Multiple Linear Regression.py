import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[: , :-1].values
Y = dataset.iloc[: , 4].values


# encoding categorical data
labelencoder = LabelEncoder()
X[: , 3] = labelencoder.fit_transform(X[: , 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# avoiding dummy variable trap
X = X[: , 1:]

# splitting the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# fitting multiple linear regression to the training set
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# predicting the test set
Y_pre = regressor.predict(X_test)

print(Y_pre-Y_test)

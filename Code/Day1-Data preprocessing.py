import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, Imputer, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# dataset = pd.read_csv('D:/Software/Github/Project/100-Days-Of-ML-Code/datasets/Data.csv')
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : , 3].values


# impute the mean value of data entry
imputer = Imputer(missing_values= "NaN", strategy= "mean", axis = 0)
imputer = imputer.fit(X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])


# transform the label values into numeric values for mathematical equations
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])


# creat a dummy variable
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)


#splitting the datasets
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.2, random_state = 0)


sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
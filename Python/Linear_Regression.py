# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

# loading diabetes dataset
diabetes=datasets.load_diabetes()

# print(diabetes.keys())
# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])
# Extracting only feature from data column
diabetes_X=diabetes.data[:,np.newaxis,2]
'''diabetes_X_train=diabetes_X[:-30]
diabetes_X_test=diabetes_X[-30:]'''
#Setting up training and test data for feature
diabetes_X_train=diabetes_X[:-30]
diabetes_X_test=diabetes_X[-30:]

diabetes_Y_train=diabetes.target[:-30]
diabetes_Y_test=diabetes.target[-30:]

# Created linear model
model=linear_model.LinearRegression()

# Training our model against train data
model.fit(diabetes_X_train,diabetes_Y_train)

# Calculating prediction from test data
diabetes_Y_predicted= model.predict(diabetes_X_test)

print("Mean Squared Error : ", mean_squared_error(diabetes_Y_test,diabetes_Y_predicted))
print("Weight : ",model.coef_)
print("Intercept : ",model.intercept_)

# Plotting train and test data
plt.scatter(diabetes_X_test,diabetes_Y_test)
plt.plot(diabetes_X_test,diabetes_Y_predicted)
plt.show()

# Using all values of data columns i.e. proving more features to our model
#Setting up training and test data for feature
diabetes_X_train=diabetes.data[:-30]
diabetes_X_test=diabetes.data[-30:]

diabetes_Y_train=diabetes.target[:-30]
diabetes_Y_test=diabetes.target[-30:]

# Created linear model
model=linear_model.LinearRegression()

# Training our model against train data
model.fit(diabetes_X_train,diabetes_Y_train)

# Calculating prediction from test data
diabetes_Y_predicted= model.predict(diabetes_X_test)

print("Mean Squared Error : ", mean_squared_error(diabetes_Y_test,diabetes_Y_predicted))
print("Weight : ",model.coef_)
print("Intercept : ",model.intercept_)

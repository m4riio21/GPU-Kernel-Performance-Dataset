from warnings import filterwarnings
filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import scipy.stats
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Perceptron, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.model_selection import cross_val_score

#Importem les dades i fem el processament necessari
dataset = pd.read_csv("../BBDD/sgemm_product.csv")
dataset['Run (ms)']=dataset[['Run1 (ms)','Run2 (ms)','Run3 (ms)','Run4 (ms)']].mean(axis=1)
dataset = dataset.drop(columns =['Run1 (ms)','Run2 (ms)','Run3 (ms)','Run4 (ms)'], axis = 1)
Q1=dataset['Run (ms)'].quantile(0.25)
Q2=dataset['Run (ms)'].quantile(0.75)
IQR = Q2 - Q1
LL=Q1-1.5*IQR
UL=Q2+1.5*IQR
dataset2 = dataset[(dataset['Run (ms)']>LL) & (dataset['Run (ms)']<UL)]
reduced_dataset = dataset2[['MWG','NWG','SA','VWN','Run (ms)']]
reduced_dataset.describe()
d = dataset2.values[:] # All atributtes
d2 = reduced_dataset.values[:] # Most correlated parameters

#Scale all attributes
scaler = StandardScaler()
scaler.fit(d)
data = scaler.transform(d)

#Scale most correlated attributes
scaler = StandardScaler()
scaler.fit(d2)
data2 = scaler.transform(d2)

x = data[:,:-1]
y = data[:, -1]

x_filtered = data2[:,:-1]
y_filtered = data2[:,-1]

#Entrenament dels models amb el nombre òptim de variables

millors_variables = ['MWG','NWG', 'SA',  'VWM','MDIMC','SB', 'NDIMC','Run (ms)']
reduced_dataset = dataset2[millors_variables]


data = reduced_dataset.values
scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)

x = data[:,:-1]
y = data[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#Creació i entrenament de models
linear_regression = LinearRegression()
decision_tree = DecisionTreeRegressor()
sgd = SGDRegressor()

linear_regression.fit(x_train,y_train)
sgd.fit(x_train,y_train)
decision_tree.fit(x_train,y_train)

#Regressió Linear
prediction = linear_regression.predict(x_test)
with open('../results/models/linear_regression/metrics_optimal_attributes.txt', 'w') as f:
    f.write("Rendiment del model (regressió linear): "+ str(linear_regression.score(x_test, y_test)) + "\n")
    f.write("MSE: " + str(mean_squared_error(y_test, prediction)))

ax = plt.scatter(x_test[:,0], y_test)
plt.plot(x_test, prediction, 'r')
plt.savefig('../results/models/linear_regression/model_fitting_optimal_attributes.png')
plt.clf()

#Decision Tree
prediction = decision_tree.predict(x_test)
with open('../results/models/decision_tree/metrics_optimal_attributes.txt', 'w') as f:
    f.write("Rendiment del model (decision tree): " + str(decision_tree.score(x_test, y_test)) + "\n")
    f.write("MSE: " + str(mean_squared_error(y_test, prediction)))

ax = plt.scatter(x_test[:,0], y_test)
plt.plot(x_test, prediction, 'r')
plt.savefig('../results/models/decision_tree/model_fitting_optimal_attributes.png')
plt.clf()

#SGD Regressor
prediction = sgd.predict(x_test)
with open('../results/models/sgd_regressor/metrics_optimal_attributes.txt', 'w') as f:
    f.write("Rendiment del model (decision tree): " + str(sgd.score(x_test, y_test)) + "\n")
    f.write("MSE: " + str(mean_squared_error(y_test, prediction)))

ax = plt.scatter(x_test[:,0], y_test)
plt.plot(x_test, prediction, 'r')
plt.savefig('../results/models/sgd_regressor/model_fitting_optimal_attributes.png')
plt.clf()

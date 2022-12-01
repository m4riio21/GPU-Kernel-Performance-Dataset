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

#Cerquem el nombre de variables òptimes per entrenar als models
variables = ['MWG','NWG', 'SA',  'VWM','MDIMC','SB', 'NDIMC','VWN', 'STRM','NDIMB' , 'KWI','MDIMA','KWG', 'STRN']

MSE_values = []

for i in range(1, 16):
    X = dataset2[variables[:i]]
    X = X.to_numpy()
    Y = dataset2['Run (ms)'].to_numpy().ravel()
    X = scaler.fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

    regr = SGDRegressor(alpha=0.0001)
    mse = cross_val_score(regr, X_train, Y_train, cv=5, scoring="neg_mean_squared_error").mean()
    mse = abs(mse)
    MSE_values.append(mse)

plt.plot(range(1, 16), MSE_values)
plt.xlabel('Nombre de variables')
plt.ylabel('MSE')
plt.savefig('../results/models/optimal_number_of_attributes.png')
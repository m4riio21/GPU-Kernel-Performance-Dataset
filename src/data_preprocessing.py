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
import dataframe_image as dfi

dataset = pd.read_csv("../BBDD/sgemm_product.csv")

#Agrupem les 4 variables target (execucions independents) en una sola per simplicitat
dataset['Run (ms)']=dataset[['Run1 (ms)','Run2 (ms)','Run3 (ms)','Run4 (ms)']].mean(axis=1)
dataset = dataset.drop(columns =['Run1 (ms)','Run2 (ms)','Run3 (ms)','Run4 (ms)'], axis = 1)
dataset.head()

#Distribució de valors en l'atribut objectiu
plt.figure(figsize=(10,6))
sns.boxplot(dataset['Run (ms)']);
plt.savefig('../results/data_preprocessing/target_before_processing.png')

#Outlier removal
Q1=dataset['Run (ms)'].quantile(0.25)
Q2=dataset['Run (ms)'].quantile(0.75)
IQR = Q2 - Q1
LL=Q1-1.5*IQR
UL=Q2+1.5*IQR
dataset2 = dataset[(dataset['Run (ms)']>LL) & (dataset['Run (ms)']<UL)]
dataset2.describe()

plt.figure(figsize=(10,6))
sns.boxplot(dataset2['Run (ms)']);
plt.savefig('../results/data_preprocessing/target_after_processing.png')

# Mirem la correlació entre els atributs d'entrada per entendre millor les dades
correlacio = dataset.corr()

plt.figure(figsize=(16,16))
sns.set(font_scale=1)

ax = sns.heatmap(correlacio, annot=True, linewidths=.5)
plt.savefig('../results/correlation/correlation_all.png')

#Mirem la correlació de cada atribut respecte el target
plt.figure(figsize=(8, 12))
heatmap = sns.heatmap(abs(dataset2.corr()[['Run (ms)']]).sort_values(by='Run (ms)', ascending=False), vmin=-1, vmax=1, annot=True)
heatmap.set_title("Correlació d'atributs amb el target", fontdict={'fontsize':18}, pad=16);
plt.savefig('../results/correlation/correlation_to_target.png')

#Agafem els 4 millors atributs
reduced_dataset = dataset2[['MWG','NWG','SA','VWN','Run (ms)']]
reduced_dataset.describe()

#Histograma d'aquest atributs
data = reduced_dataset.values
atributs = ['MWG', 'NWG', 'SA', 'VWN']
for i in range(0, 4):
    ax = plt.subplot(2, 2, i + 1)
    if i > 1:
        plt.xlabel("Histograma de l'atribut " + atributs[i])
    else:
        plt.title("Histograma de l'atribut " + atributs[i])

    ax.hist(data[:, i], bins=15, range=[np.min(data[:, i]), np.max(data[:, i])], histtype="bar", rwidth=0.8)
plt.savefig('../results/histogrames/top4_attributes_histogram.png')

#Escalem les dades per solucionar les diferencies de valors
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

print(y)
# Pràctica Kaggle APC UAB 2022-23
### Author: **Mario González Castaño**
### Dataset: **GPU Kernel Performance Dataset**
### URL: [kaggle](https://www.kaggle.com/datasets/rupals/gpu-runtime)
## Abstract
In this project I will analyze a dataset with samples of the performance of a GPU (Graphics Card) running a 2048 * 2048 matrix multiplication job, using a parametrizable kernel with 241600 possibilities between all the parameter combinations. In the dataset, we have 4 independent execution samples for each parameter combination.

The dataset is composed by 18 atributes, 14 of them corresponding to the parametres given to the kernel for each job. The 4 last columns represent the 4 independent runtimes.

### Objective
In this project we want to predict which will the runtime be given a set of kernel parameters.

## Experiments
We will proceed with the different experiments done in the project to understand a bit better the dataset and perform those predictions.

### Preprocessing
Quines proves hem realitzat que tinguin a veure amb el pre-processat? com han afectat als resultats?

To prepare the data for the prediction task, it is compulsory to analyse the attributes of the dataset and choose the best that will do the best regression. To do that, there are many possibilities: histograms, scatter plots, heatmap (correlation of attributes), apply normal test to see the distribution of all attributes (interesting to choose those who follow normal distribution), and we can normalize data if the attributes have diferent ranges.
In our dataset, for example, cylinders and size of car do not have same ranges.

As for the data preprocessing, it is really important to analyze the dataset attributes and filter them to later do the regression. In our dataset, we have 4 objective attributes for each parameter combinations. We have simplified this by doing the mean of these executions and grouping them in just one attribute:

| Run1 (ms) | Run2 (ms) | Run3 (ms) | Run4 (ms) | 
| -- | -- | -- | -- |
| 115.26 | 115.87 | 118.55 | 115.80 |

This is being transformed to:

| Run (ms) |
| -- |
| 116,37 |

The other transformation that has been applied is **outlier removal** in this objective attribute. This transformation consists in deleting those values in a given attribute that differ a lot from most of the values, meaning that we will have a smaller set of values to predict.


### Model
| Model | Hyperparameters | Score | MSE |
| -- | -- | -- | -- |
| [Decision Tree Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html?highlight=decision+tree) | {'max_depth': 10, 'max_features': None, 'max_leaf_nodes': 500, 'min_weight_fraction_leaf': 0.0, 'splitter': 'random'} | 0.293 | 0.707 |
| [Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html?highlight=linearregre#sklearn.linear_model.LinearRegression) | {'fit_intercept': False} | 0.234 | 0.767 |
| [SGD Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html?highlight=sgdregress#sklearn.linear_model.SGDRegressor) | {'alpha': 0.0001, 'fit_intercept': False, 'loss': 'squared_error', 'penalty': 'l2', 'tol': 0.01} | 0.234 | 0.767 |

As we can see, the best model for this dataset is **Decision Tree Regressor**. However, the model score is pretty low meaning that none of the models will perform a very good prediction.

## Demo
Per tal de fer una prova, es pot fer servir amb la següent comanda
``` python3 demo/demo.py --input here ```
## Conclusions
El millor model que s'ha aconseguit ha estat...
En comparació amb l'estat de l'art i els altres treballs que hem analitzat....
## Idees per treballar en un futur
Crec que seria interesant indagar més en...
## Llicencia
This project is under the GPL-3.0 License - see the [LICENSE](LICENSE) for more details

# Pràctica Kaggle APC UAB 2022-23
### Nom: **Mario González Castaño**
### Dataset: **GPU Kernel Performance Dataset**
### URL: [kaggle](https://www.kaggle.com/datasets/rupals/gpu-runtime)
## Resum
En aquest treball analitzarem un dataset del rendiment d'una GPU (tarjeta gràfica) fent una operació de multiplicació amb matrius 2048 * 2048, fent us d'un kernel parametritzable amb 241600 possibles combinacions de paràmetres. En aquest dataset, tenim 4 execucions de les operacions per cada combinació de paràmetres.

El nostre dataset està composat de 18 atributs, 14 d'ells son els atributs corresponents als paràmetres que pot rebre el kernel el qual executa les operacions amb matrius, i els últims 4 atributs son execucions independents amb els paràmetres corresponents.

### Objectius del dataset
Amb aquest dataset volem aprendre quina es la millor combinació d'atributs la qual optimitza els atributs objectiu de temps d'execució.

## Experiments
A continuació veurem els experiments realitzats a la pràctica amb l'objectiu d'entendre una mica millor el dataset i fer prediccions.

### Preprocessat
Quines proves hem realitzat que tinguin a veure amb el pre-processat? com han afectat als resultats?

En quant al preprocessament de les dades, es un punt important que ens ajudarà a fer una millor regressió després. En aquest dataset, per exemple, al tenir 4 atributs objectiu representant execucions independents, és un problema a l'hora de fer regressió. Per la qual cosa, una de les transformacions que s'han fet al dataset es fer la mitja d'aquests temps d'execució per tenir només un atribut objectiu:

| Run1 (ms) | Run2 (ms) | Run3 (ms) | Run4 (ms) | 
| -- | -- | -- | -- |
| 115.26 | 115.87 | 118.55 | 115.80 |

Això ho transformem en: 

| Run (ms) |
| -- |
| 116,37 |

L'altre transformació que s'ha aplicat es **outlier removal** en l'atribut a predir, aquesta transformació consisteix en l'eliminació d'aquells valors que es trobem molt allunyats de la gran majoria de valors. Això ens donarà un conjunt més petit de valors en l'atribut objectiu la qual cosa ens facilitarà a l'hora de fer les nostres prediccions.

### Model
| Model | Hiperparametres | Score | MSE |
| -- | -- | -- | -- |
| [Decision Tree Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html?highlight=decision+tree) | {'max_depth': 10, 'max_features': None, 'max_leaf_nodes': 500, 'min_weight_fraction_leaf': 0.0, 'splitter': 'random'} | 0.293 | 0.707 |
| [Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html?highlight=linearregre#sklearn.linear_model.LinearRegression) | {'fit_intercept': False} | 0.234 | 0.767 |
| [SGD Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html?highlight=sgdregress#sklearn.linear_model.SGDRegressor) | {'alpha': 0.0001, 'fit_intercept': False, 'loss': 'squared_error', 'penalty': 'l2', 'tol': 0.01} | 0.234 | 0.767 |

## Demo
Per tal de fer una prova, es pot fer servir amb la següent comanda
``` python3 demo/demo.py --input here ```
## Conclusions
El millor model que s'ha aconseguit ha estat...
En comparació amb l'estat de l'art i els altres treballs que hem analitzat....
## Idees per treballar en un futur
Crec que seria interesant indagar més en...
## Llicencia
El projecte s’ha desenvolupat sota llicència ZZZz.

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
### Model
| Model | Hiperparametres | Mètrica | Temps |
| -- | -- | -- | -- |
| [Random Forest](link) | 100 Trees, XX | 57% | 100ms |
| Random Forest | 1000 Trees, XX | 58% | 1000ms |
| SVM | kernel: lineal C:10 | 58% | 200ms |
| -- | -- | -- | -- |
| [model de XXX](link al kaggle) | XXX | 58% | ?ms |
| [model de XXX](link al kaggle) | XXX | 62% | ?ms |
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

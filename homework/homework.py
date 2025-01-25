# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
import pickle

ruta_archivo = "./files/grading/x_train.pkl"

with open(ruta_archivo, "rb") as file:
    contenido1 = pickle.load(file)

ruta_archivo = "./files/grading/y_train.pkl"

with open(ruta_archivo, "rb") as file:
    contenido2 = pickle.load(file)
    
ruta_archivo = "./files/grading/x_test.pkl"

with open(ruta_archivo, "rb") as file:
    contenido3 = pickle.load(file)
    
ruta_archivo = "./files/grading/y_test.pkl"

with open(ruta_archivo, "rb") as file:
    contenido4 = pickle.load(file)
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#

import pandas as pd

data_train = pd.read_csv('./files/input/train_data.csv.zip', index_col = False, compression = "zip")
data_test = pd.read_csv("./files/input/test_data.csv.zip", index_col = False, compression = "zip")

import numpy as np

data_train.rename(columns={"default payment next month": "default"}, inplace=True)
data_test.rename(columns={"default payment next month": "default"}, inplace=True)
data_train.drop(columns=['ID'], inplace=True)
data_test.drop(columns=['ID'], inplace=True)

data_train['EDUCATION'] = data_train['EDUCATION'].apply(lambda x: x if x > 0 else np.nan)
data_test['EDUCATION'] = data_test['EDUCATION'].apply(lambda x: x if x > 0 else np.nan)

data_train['MARRIAGE'] = data_train['MARRIAGE'].apply(lambda x: x if x > 0 else np.nan)
data_test['MARRIAGE'] = data_test['MARRIAGE'].apply(lambda x: x if x > 0 else np.nan)

data_train['EDUCATION'] = data_train['EDUCATION'].apply(lambda x: 4 if x > 4 else x)
data_test['EDUCATION'] = data_test['EDUCATION'].apply(lambda x: 4 if x > 4 else x)

data_train.dropna(inplace=True)
data_test.dropna(inplace=True)

#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#

x_train = data_train.drop(columns=['default'])
y_train = data_train["default"]
x_test = data_test.drop(columns=['default'])
y_test = data_test["default"]
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression

categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
numerical_features = list(set(x_test.columns) - set(categorical_features))

preprocesamiento = ColumnTransformer(
    transformers=[
        ('categoricas', OneHotEncoder(), categorical_features),
        ('numericas', MinMaxScaler(), numerical_features)
    ],
    remainder='passthrough', 
)

select = SelectKBest(score_func = f_classif)

clf = LogisticRegression()


pipeline = Pipeline(steps=[
    ('preprocessor', preprocesamiento),
    ('selector', select),
    ('clf', clf)
])

#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

param_grid = {'selector__k': range(1, len(x_train.columns) + 1)}

modelo = GridSearchCV(pipeline, param_grid, cv=10, scoring='balanced_accuracy', n_jobs=-1)

modelo.fit(x_train, y_train)

#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#

import os
import gzip
import pickle

dir_path = './files/models'

if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    with gzip.open('./files/models/model.pkl.gz', 'wb') as f:
        pickle.dump(modelo, f)
else:
    with gzip.open('./files/models/model.pkl.gz', 'wb') as f:
        pickle.dump(modelo, f)

# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#

from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score
import json

y_train_pred = modelo.predict(x_train)
y_test_pred = modelo.predict(x_test)

train_metrics = {
    "type": "metrics", 
    'dataset': 'train',
    'precision': precision_score(y_train, y_train_pred),
    'balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred),
    'recall': recall_score(y_train, y_train_pred),
    'f1_score': f1_score(y_train, y_train_pred)
}


test_metrics = {
    "type": "metrics", 
    'dataset': 'test',
    'precision': precision_score(y_test, y_test_pred),
    'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
    'recall': recall_score(y_test, y_test_pred),
    'f1_score': f1_score(y_test, y_test_pred)
}

output_path = './files/output/metrics.json'
os.makedirs(os.path.dirname(output_path), exist_ok=True) 

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(train_metrics, f, ensure_ascii=False) 
    f.write('\n')
    json.dump(test_metrics, f, ensure_ascii=False) 
    f.write('\n')
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
from sklearn.metrics import confusion_matrix

train_cm = confusion_matrix(y_train, y_train_pred)
test_cm = confusion_matrix(y_test, y_test_pred)

train_cm_dict = {
    'type': 'cm_matrix',
    'dataset': 'train',
    'true_0': {'predicted_0': int(train_cm[0, 0]), 'predicted_1': int(train_cm[0, 1])},
    'true_1': {'predicted_0': int(train_cm[1, 0]), 'predicted_1': int(train_cm[1, 1])}
}

test_cm_dict = {
    'type': 'cm_matrix',
    'dataset': 'test',
    'true_0': {'predicted_0': int(test_cm[0, 0]), 'predicted_1': int(test_cm[0, 1])},
    'true_1': {'predicted_0': int(test_cm[1, 0]), 'predicted_1': int(test_cm[1, 1])}
}

output_path = '../files/output/metrics.json'

with open(output_path, 'a', encoding='utf-8') as f:
    json.dump(train_cm_dict, f, ensure_ascii=False) 
    f.write('\n')
    json.dump(test_cm_dict, f, ensure_ascii=False)  
    f.write('\n')
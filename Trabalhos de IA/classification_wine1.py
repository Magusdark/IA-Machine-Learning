import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from unicodedata import name

atributos = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
             "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"]

# resultados = ["unacc", "acc", "good", "vgood"]

vinhos = pd.read_csv(
    "winequality-white.csv")

amostras = vinhos.drop(vinhos.columns[11], axis=1)
amostras = amostras.to_numpy()

resultados = vinhos.drop(
    vinhos.columns[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], axis=1)

resultados = resultados.to_numpy().ravel()

X_train, X_test, y_train, y_test = train_test_split(
    amostras, resultados, random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

print(knn.score(X_test, y_test))

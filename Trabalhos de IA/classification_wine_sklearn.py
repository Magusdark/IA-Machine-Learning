import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
import pandas as pd
from unicodedata import name

# atributos = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
#              "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"]

# resultados = ["unacc", "acc", "good", "vgood"]

vinhos = load_wine()

X_train, X_test, y_train, y_test = train_test_split(
    vinhos['data'], vinhos['target'], random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

print(knn.score(X_test, y_test))

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_and_predict():
    """
    Ładuje zbiór Iris, trenuje model i zwraca predykcje oraz prawdziwe etykiety.
    """
    # Wczytanie zbioru Iris (posiada 3 klasy: 0, 1, 2)
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Podział danych na zbiór treningowy (80%) i testowy (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Utworzenie i wytrenowanie modelu klasyfikatora
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Wygenerowanie predykcji
    preds = model.predict(X_test)

    return preds, y_test

def get_accuracy(y_true, y_pred):
    """
    Oblicza i zwraca dokładność (accuracy) modelu.
    """
    return accuracy_score(y_true, y_pred)
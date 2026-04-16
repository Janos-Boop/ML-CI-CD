import numpy as np
from model import train_and_predict, get_accuracy


def test_predictions_not_none():
    """
    Test 1: Sprawdza, czy otrzymujemy jakąkolwiek predykcję.
    """
    preds, _ = train_and_predict()
    assert preds is not None, "Predictions should not be None."


def test_predictions_length():

    preds, y_test = train_and_predict()

    # Sprawdzamy czy lista nie jest pusta
    assert len(preds) > 0, "Lista predykcji jest pusta"
    # Sprawdzamy czy liczba predykcji odpowiada liczbie prawdziwych etykiet testowych
    assert len(preds) == len(y_test), "Długość predykcji nie odpowiada liczbie próbek testowych."


def test_predictions_value_range():
    """
    Sprawdza, czy wartości w predykcjach
    mieszczą się w spodziewanym zakresie: Dla zbioru Iris mamy 3 klasy (0, 1, 2).
    """
    preds, _ = train_and_predict()
    valid_classes = {0, 1, 2}

    # Sprawdzamy, czy każda wygenerowana predykcja znajduje się w dozwolonym zbiorze klas
    assert all(p in valid_classes for p in preds), "Predykcje zawierają wartości spoza dozwolonego zakresu (0, 1, 2)."


def test_model_accuracy():
    """
    Sprawdza, czy model osiąga co najmniej 70% dokładności.
    """
    preds, y_test = train_and_predict()
    accuracy = get_accuracy(y_test, preds)

    # Oczekujemy skuteczności (accuracy) na poziomie min. 0.70
    assert accuracy >= 0.70, f"Dokładność modelu jest za niska: {accuracy * 100}%, wymagane min. 70%."
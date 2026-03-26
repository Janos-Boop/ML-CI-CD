import os
from fastapi import FastAPI, HTTPException
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
from sklearn.linear_model import LinearRegression
import numpy as np

#Inicjalizacja aplikacji
app = FastAPI(
    title="ML API",
    description="Podstawowe API dla modelu uczenia maszynowego",
    version="1.0.0"
)


#Trenujemy banalny model regresji liniowej (funkcja y = 2x) na sztucznych danych
X_train = np.array([[1.0], [2.0], [3.0], [4.0]])
y_train = np.array([2.0, 4.0, 6.0, 8.0])

model = LinearRegression()
model.fit(X_train, y_train)



class PredictionRequest(BaseModel):
    feature: float


#Główny endpoint ("/")
@app.get("/")
def read_root():
    return {"message": "Witaj w moim API dla modelu ML!"}



# Endpoint testujący zmienną środowiskową (Zadanie 5 lab 5)
@app.get("/config")
def get_config():
    # Pobieramy zmienną MY_SECRET_KEY, jeśli jej nie ma, dajemy wartość domyślną
    secret_key = os.getenv("MY_SECRET_KEY", "Brak klucza! Działam na ustawieniach domyślnych.")
    return {
        "status": "success",
        "secret_key": secret_key
    }
#Endpoint predykcji
@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        #Pobieramy zwalidowaną cechę z żądania
        X_new = np.array([[request.feature]])

        #Wykonujemy predykcję
        prediction = model.predict(X_new)

        #Zwracamy wynik
        return {"prediction": float(prediction[0])}

    except Exception as e:
        #Obsługa nieoczekiwanych błędów po stronie modelu
        raise HTTPException(status_code=500, detail=f"Błąd podczas predykcji: {str(e)}")


#Endpoint z informacjami o modelu
@app.get("/info")
def get_info():
    return {
        "model_type": "LinearRegression",
        "features_count": int(model.n_features_in_),
        "description": "Prosty model regresji liniowej wytrenowany lokalnie (y = 2x)."
    }


#Endpoint sprawdzający status serwera
@app.get("/health")
def health_check():
    return {"status": "ok"}
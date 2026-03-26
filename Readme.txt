# Projekt ML API - Regresja Liniowa

Proste API zbudowane w oparciu o **FastAPI**, serwujące model uczenia maszynowego (Regresja Liniowa z biblioteki `scikit-learn`). Projekt jest skonteneryzowany za pomocą Dockera i zawiera środowisko uruchomieniowe z wykorzystaniem Docker Compose (wraz z testową bazą Redis).

## Konfiguracja i wymagania zasobowe
* **Zasoby:** Aplikacja jest lekka. Wymaga minimalnie ok. 256-512 MB pamięci RAM dla głównego kontenera.
* **Porty sieciowe:** * `8000` - Główne API (FastAPI)
  * `6379` - Serwis Redis (uruchamiany przez Docker Compose)
* **Zmienne środowiskowe:** W obecnej wersji aplikacja opiera się na domyślnych ustawieniach i nie wymaga dodatkowej konfiguracji przez `.env`.

## Instrukcja uruchamiania

### 1. Uruchomienie lokalne (bez Dockera)
1. Zainstaluj wymagane biblioteki z pliku konfiguracyjnego:
   ```bash
   pip install -r requirements.txt
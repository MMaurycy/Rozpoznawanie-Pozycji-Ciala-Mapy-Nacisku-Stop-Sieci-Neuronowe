# Rozpoznawanie Pozycji Ciała na Podstawie Map Nacisku Stopy przy Użyciu Sieci Neuronowych

**Autor:** Marcin Przybylski
**Data:** 1 maja 2025

## Opis Projektu

Niniejszy projekt badawczy dotyczy zastosowania sieci neuronowych do klasyfikacji pozycji ciała człowieka – stanie, skłon, przysiad – na podstawie danych z map rozkładu nacisku stóp. Dane wejściowe to obrazy o wymiarach $18 \times 34$ pikseli, reprezentujące rozkład nacisku na podłoże. Głównym celem projektu było porównanie efektywności dwóch architektur sieci neuronowych: w pełni połączonej sieci neuronowej (FCN) oraz konwolucyjnej sieci neuronowej (CNN). Dodatkowo, projekt eksplorował wpływ technik optymalizacyjnych, takich jak normalizacja wsadowa (Batch Normalization), zwiększenie liczby filtrów, augmentacja danych (Data Augmentation) oraz dynamiczna redukcja współczynnika uczenia (Learning Rate Scheduling) na wydajność modelu CNN.

## 📂 Struktura Projektu

Prace zostały zrealizowane w środowisku Google Colaboratory, z wykorzystaniem języka Python i bibliotek TensorFlow/Keras.


## ⚙️ Metodologia

### 1. Wczytywanie i Przygotowanie Danych

Dane wejściowe (mapy nacisku stóp i etykiety pozycji ciała) pobrano ze zdalnego serwera w formacie plików `.npy`. Dostępne były oddzielne zbiory treningowy i testowy.

* **Kształty danych (po transpozycji):**
    * Treningowe (cechy): (5416, 18, 34)
    * Testowe (cechy): (744, 18, 34)
    * Treningowe (etykiety): (5416, 1)
    * Testowe (etykiety): (744, 1)
* **Mapowanie klas:** 0 - Przysiad, 1 - Skłon, 2 - Stanie.
* **Kroki przetwarzania wstępnego:**
    * Transpozycja wymiarów cech do formatu (liczba\_obrazów, wysokość, szerokość).
    * Normalizacja wartości pikseli do zakresu [0, 1].
    * Spłaszczenie obrazów $18 \times 34$ do wektorów 612 cech dla FCN.
    * Zmiana kształtu danych wejściowych do (18, 34, 1) dla CNN.
    * Kodowanie "one-hot" etykiet klas.
    * Podział zbioru treningowego na właściwy zbiór treningowy (85%) i walidacyjny (15%) z stratyfikacją (random\_state=173201). Zbiór testowy pozostał nienaruszony.

### 2. Architektury Modeli

* **Model FCN (W Pełni Połączony):**
    * Warstwa wejściowa dla 612 cech.
    * Dwie ukryte warstwy gęste (Dense) po 7 neuronów z aktywacją ReLU, każda poprzedzona warstwą Dropout (0.1).
    * Warstwa wyjściowa Dense z 3 neuronami (liczba klas) i aktywacją Softmax.
* **Model CNN (Bazowy):**
    * Warstwa Conv2D (4 filtry, kernel 3x3, ReLU, padding 'valid') → MaxPooling2D (2x2).
    * Warstwa Conv2D (4 filtry, kernel 3x3, ReLU, padding 'valid') → MaxPooling2D (2x2).
    * Warstwa Conv2D (4 filtry, kernel 3x3, ReLU, padding 'valid').
    * Warstwa Flatten.
    * Warstwa Dense (16 neuronów, ReLU).
    * Warstwa Dropout (0.2).
    * Warstwa wyjściowa Dense (3 neurony, Softmax).
* **Modele CNN (Optymalizacje):**
    * **CNN v2 (BatchNorm, More Filters):** Dodano Batch Normalization po Conv2D i Dense. Zwiększono liczbę filtrów w Conv2D (8, 16, 32) i neuronów w Dense (32). Dropout zwiększony do 0.3.
    * **CNN v3 (Augmentation):** Bazując na v2, dodano warstwy augmentacji danych (RandomFlip, RandomRotation, RandomZoom).
    * **CNN v4 (LR Scheduling):** Architektura jak v3, z zastosowaniem callbacku `ReduceLROnPlateau` podczas treningu.

### 3. Proces Treningu

* **Kompilacja:** Optymalizator Adam, funkcja straty `categorical_crossentropy`, metryka `accuracy`.
* **Trening FCN i bazowego CNN:**
    1.  Trening wstępny (500 epok) do identyfikacji optymalnej liczby epok na podstawie `val_loss`.
    2.  Trening ostateczny z resetem wag przez wyznaczoną liczbę epok.
* **Trening modeli optymalizacyjnych (v2, v3, v4):** Liczba epok bazująca na optymalnej dla bazowego CNN (v2 tak samo; v3, v4 podwojona liczba dla nowego `random_state`).
* **Rozmiar wsadu (batch\_size):** 32.

### 4. Ewaluacja Modeli

* Ocena na zbiorze testowym.
* Metryki: Dokładność (Accuracy), Strata (Loss).
* Analiza macierzy pomyłek.
* Wizualizacja krzywych uczenia.

## 🛠️ Wykorzystane Narzędzia i Technologie

* **Język programowania:** Python
* **Biblioteki:**
    * TensorFlow/Keras
    * NumPy
    * Scikit-learn (train\_test\_split, metryki oceny)
    * Matplotlib/Seaborn (wizualizacje)
    * Requests (pobieranie danych)
* **Środowisko:** Google Colaboratory

## 📊 Wyniki i Interpretacja (random_state = 173201)

### Model FCN
* Dokładność na zbiorze testowym: 52.55%
* Strata na zbiorze testowym: 1.0120
* Najczęstsze błędy: Mylenie Skłonu ze Staniem (117 przypadków), Przysiadu ze Staniem (62 przypadki).

### Model CNN (Bazowy)
* Dokładność na zbiorze testowym: 49.33%
* Strata na zbiorze testowym: 1.0117
* Niższa dokładność niż FCN w tym przebiegu. Model często mylił klasy Stanie i Skłon oraz Stanie z Przysiadem.

### Optymalizacje CNN

* **CNN v2 (BatchNorm, More Filters):**
    * Dokładność: 44.62%
    * Strata: 3.7451
    * Najgorszy wynik, słaba generalizacja na zbiór testowy mimo dobrej dokładności walidacyjnej.
* **CNN v3 (Augmentation):**
    * Dokładność: 53.90%
    * Strata: 1.0241
    * Najlepsza dokładność w tym przebiegu, nieznacznie przewyższając FCN. Augmentacja okazała się pomocna.
* **CNN v4 (LR Scheduling):**
    * Dokładność: 52.82%
    * Strata: 0.9833
    * Wynik zbliżony do FCN, gorszy od v3. Redukcja LR nie przyniosła poprawy w porównaniu do samej augmentacji.

### Porównanie Modeli (random_state = 173201)

| Model                      | Dokładność na zbiorze testowym (%) |
| -------------------------- | ------------------------------------ |
| FCN (bazowy)               | 52.55                                |
| CNN (bazowy)               | 49.33                                |
| CNN v2 (BN, More Filters)  | 44.62                                |
| CNN v3 (Augmentation)      | 53.90                                |
| CNN v4 (LR Scheduling)     | 52.82                                |


W tym przebiegu model FCN okazał się lepszy od bazowego CNN. Model CNN v3 (Augmentation) osiągnął najlepszy wynik. Augmentacja danych była najbardziej skuteczną techniką optymalizacji. Ogólna dokładność (max 54%) pozostaje stosunkowo niska.

## 🏁 Podsumowanie i Wnioski

1.  **Niejednoznaczna Przewaga CNN nad FCN:** Bazowy CNN wypadł gorzej niż FCN w tym przebiegu. Dopiero augmentacja w CNN v3 dała nieznaczną przewagę.
2.  **Kluczowa Rola Augmentacji Danych:** Augmentacja (model v3) przyniosła najlepszy wynik (53.90%), co sugeruje jej kluczowe znaczenie dla generalizacji.
3.  **Negatywny Wpływ Innych Optymalizacji w Tej Konfiguracji:** Batch Normalization ze zwiększeniem złożoności (v2) drastycznie pogorszyło wynik. LR Scheduling (v4) nie poprawił wyniku względem samej augmentacji.
4.  **Trudność Zadania Klasyfikacyjnego:** Uzyskane dokładności (max ok. 54%) sugerują, że zadanie jest wymagające, a mapy nacisku mogą być trudne do rozróżnienia.
5.  **Znaczenie Walidacji i Wpływ `random_state`:** Losowy podział danych ma duży wpływ na wyniki, co podkreśla potrzebę np. walidacji krzyżowej.

---

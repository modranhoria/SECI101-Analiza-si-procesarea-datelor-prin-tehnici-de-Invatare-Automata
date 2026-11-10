# Proiect 3 – Algoritmul KNN (K-Nearest Neighbors)

## Obiectivul laboratorului

În acest laborator este implementat și evaluat algoritmul **KNN (K-Nearest Neighbors)** pentru clasificarea datelor din setul **Iris**. Sunt prezentate două variante:

1. implementarea algoritmului de la zero, folosind Python și NumPy;
2. implementarea folosind clasificatorul `KNeighborsClassifier` din biblioteca Scikit-learn.

Ambele variante folosesc același set de date și aceeași împărțire în seturi de antrenare și testare, astfel încât rezultatele să poată fi comparate.

## 1. Importarea bibliotecilor

Pentru manipularea datelor și implementarea algoritmului sunt folosite bibliotecile NumPy și Pandas, iar din Scikit-learn sunt utilizate funcțiile pentru încărcarea setului de date, împărțirea acestuia și evaluarea clasificatorului.

```python
import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

## 2. Importarea setului de date Iris

Setul de date Iris este încărcat direct din colecția de seturi de date inclusă în Scikit-learn. Referința setului de date este disponibilă și în [UCI Machine Learning Repository – Iris](https://archive.ics.uci.edu/dataset/53/iris).

```python
dataset = datasets.load_iris()

X = dataset.data
y = dataset.target
```

Variabilele utilizate sunt:

- `X` – matricea caracteristicilor (datele de intrare);
- `y` – vectorul etichetelor/claselor asociate observațiilor.

## 3. Împărțirea setului de date

Datele sunt împărțite în două subseturi:

- **setul de antrenare** (`X_train`, `y_train`) – utilizat pentru identificarea celor mai apropiați vecini;
- **setul de testare** (`X_test`, `y_test`) – utilizat pentru evaluarea predicțiilor.

Parametrul `test_size=0.2` alocă 20% din observații pentru testare și 80% pentru antrenare. Este folosit `random_state=42` pentru ca împărțirea să fie reproductibilă.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)
```

Pentru setul Iris încărcat din Scikit-learn, împărțirea de mai sus produce 120 de observații pentru antrenare și 30 pentru testare.

---

## 4. Implementarea algoritmului KNN de la zero

### 4.1. Stabilirea numărului de vecini

Parametrul `k` reprezintă numărul de vecini cei mai apropiați care participă la votul pentru stabilirea clasei. În exemplul din laborator este utilizată valoarea `k = 3`.

```python
k = 3
```

### 4.2. Calculul distanței euclidiene

Pentru a determina cât de apropiate sunt două observații, se calculează distanța euclidiană dintre vectorii lor de caracteristici:

$$
d(a,b) = \sqrt{\sum_{i=1}^{n}(a_i-b_i)^2}
$$

Funcția primește doi vectori și returnează distanța dintre aceștia.

```python
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))
```

### 4.3. Clasificarea pe baza votului majoritar

După calcularea distanțelor față de toate observațiile din setul de antrenare, perechile distanță–clasă sunt sortate crescător. Funcția `classify()` primește lista sortată și selectează primele `k` clase.

Clasa prezisă este clasa care apare cel mai des printre cei `k` vecini selectați. Această regulă este cunoscută ca **majority vote** (votul majorității).

```python
def classify(distances, k):
    neighbors = []

    for distance, label in distances[:k]:
        neighbors.append(label)

    prediction = max(set(neighbors), key=neighbors.count)
    return prediction
```

### 4.4. Implementarea funcției KNN

Funcția `knn()` parcurge fiecare observație din setul de testare și execută următorii pași:

1. inițializează lista predicțiilor;
2. parcurge fiecare observație din `X_test`;
3. calculează distanța dintre observația curentă și fiecare observație din setul de antrenare;
4. asociază fiecărei distanțe eticheta clasei din setul de antrenare;
5. sortează perechile distanță–clasă în ordinea crescătoare a distanței;
6. determină clasa prin votul majoritar al celor mai apropiați `k` vecini;
7. adaugă clasa prezisă în lista de predicții.

```python
def knn(X_train, y_train, X_test, k):
    predictions = []

    for sample in X_test:
        distances = []

        for train_sample, train_label in zip(X_train, y_train):
            distance = euclidean_distance(sample, train_sample)
            distances.append((distance, train_label))

        distances.sort(key=lambda item: item[0])
        prediction = classify(distances, k)
        predictions.append(prediction)

    return predictions
```

Funcția Python `zip()` combină elementele corespunzătoare din două iterabile. În acest exemplu, `zip(X_train, y_train)` permite parcurgerea simultană a fiecărui vector de caracteristici și a etichetei sale.

### 4.5. Determinarea predicțiilor

După definirea funcțiilor, algoritmul este aplicat asupra observațiilor din setul de testare:

```python
prediction = knn(X_train, y_train, X_test, k)
```

Variabila `prediction` conține clasele estimate pentru observațiile din `X_test`.

### 4.6. Evaluarea modelului

Performanța clasificatorului este evaluată folosind **acuratețea** (*accuracy*). Aceasta reprezintă raportul dintre numărul predicțiilor corecte și numărul total de observații evaluate.

```python
accuracy = accuracy_score(y_test, prediction)
print(accuracy)
```

---

## 5. Implementarea KNN folosind biblioteca Scikit-learn

Scikit-learn pune la dispoziție clasificatorul `KNeighborsClassifier`, care implementează algoritmul KNN. În această parte sunt refolosite seturile de antrenare și testare definite anterior.

### 5.1. Importarea clasificatorului

```python
from sklearn.neighbors import KNeighborsClassifier
```

### 5.2. Inițializarea clasificatorului

Parametrul `n_neighbors` stabilește numărul de vecini utilizați pentru clasificare. Se păstrează aceeași valoare `k = 3` ca în implementarea de la zero.

```python
model = KNeighborsClassifier(n_neighbors=k)
```

### 5.3. Antrenarea clasificatorului

Metoda `fit()` primește caracteristicile și etichetele din setul de antrenare:

```python
model.fit(X_train, y_train)
```

### 5.4. Realizarea predicțiilor

Metoda `predict()` determină clasa estimată pentru fiecare observație din setul de testare:

```python
y_pred = model.predict(X_test)
```

### 5.5. Evaluarea modelului

Acuratețea este calculată folosind aceeași funcție `accuracy_score()`:

```python
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
```

---

## 6. Compararea celor două implementări

Laboratorul urmărește realizarea aceleiași sarcini de clasificare prin două abordări:

| Aspect | Implementare de la zero | Implementare Scikit-learn |
|---|---|---|
| Calculul distanțelor | Funcție definită manual | Gestionat de clasificator |
| Selectarea vecinilor | Sortarea distanțelor și selectarea primilor `k` | Gestionată de `KNeighborsClassifier` |
| Stabilirea clasei | Vot majoritar implementat manual | Gestionată de clasificator |
| Antrenare | Nu există o etapă separată de `fit()`; observațiile de antrenare sunt folosite la calculul distanțelor | Metoda `fit(X_train, y_train)` |
| Predicție | Funcția proprie `knn()` | Metoda `predict(X_test)` |
| Evaluare | `accuracy_score()` | `accuracy_score()` |

Pentru o comparație corectă se păstrează aceleași date de antrenare, aceleași date de testare și aceeași valoare pentru `k`.

# 7. Experimentare cu diferite valori ale lui k
Reluați clasificarea folosind și alte valori pentru numărul de vecini k, nu doar valoarea k = 3 utilizată în exemplul inițial. 
Încercați, de exemplu, valorile k = 1, k = 5 și k = 7.

Pentru fiecare valoare a lui k:
1. rulați implementarea KNN realizată de la zero;
2. rulați clasificatorul KNeighborsClassifier din Scikit-learn;
3. calculați și afișați acuratețea;
4. comparați rezultatele obținute.

### Fișiere necesare
- `README.md` — explicațiile și codul laboratorului
- `KNN.ipynb` — notebook-ul original

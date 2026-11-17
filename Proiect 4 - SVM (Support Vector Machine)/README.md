# Proiect 4 – Algoritmul SVM (Support Vector Machine)

## Obiectivul laboratorului

În acest laborator este studiat algoritmul de învățare automată **SVM (Support Vector Machine)** pentru clasificarea speciilor de pinguini din setul de date **Penguins**, disponibil prin biblioteca Seaborn.

Sunt abordate două tipuri de probleme:

1. **clasificare binară** – se păstrează numai speciile *Adelie* și *Gentoo*;
2. **clasificare multiclasă** – se utilizează toate cele trei specii: *Adelie*, *Chinstrap* și *Gentoo*.

În notebook sunt explorate datele, sunt tratate valorile lipsă, sunt standardizate caracteristicile numerice și sunt codificate caracteristicile categoriale. Ulterior, se antrenează și se evaluează modele SVM folosind biblioteca Scikit-learn.

## 1. Importarea bibliotecilor și a setului de date Penguins

Pentru manipularea și explorarea datelor sunt utilizate NumPy, Pandas, Matplotlib și Seaborn. Din Scikit-learn sunt folosite ulterior funcții pentru preprocesare, împărțirea datelor, antrenarea clasificatorului SVM și evaluarea predicțiilor.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

Setul de date este încărcat direct din Seaborn:

```python
dataset = sns.load_dataset("penguins")
dataset.shape
dataset.head()
```

Fiecare observație descrie un pinguin. Setul include caracteristici numerice și categoriale, iar variabila care trebuie prezisă este `species`.

Caracteristicile utilizate în preprocesare sunt:

- `bill_length_mm` – lungimea ciocului, în milimetri;
- `bill_depth_mm` – adâncimea ciocului, în milimetri;
- `flipper_length_mm` – lungimea aripioarei, în milimetri;
- `body_mass_g` – masa corporală, în grame;
- `island` – insula pe care a fost observat pinguinul;
- `sex` – sexul pinguinului.

Variabila țintă este `species`, cu cele trei clase *Adelie*, *Chinstrap* și *Gentoo*.

## 2. Analiza exploratorie a datelor (EDA)

Înainte de antrenarea unui model este necesară înțelegerea structurii setului de date și verificarea calității acestuia.

### 2.1. Afișarea coloanelor

```python
print("Lista de coloane", dataset.columns.to_list())
```

Această instrucțiune afișează numele atributelor disponibile în setul de date.

### 2.2. Verificarea valorilor lipsă

```python
print("Valori lipsă:")
print(dataset.isna().sum())
```

`isna()` identifică valorile lipsă, iar `sum()` calculează numărul acestora pentru fiecare coloană. Valorile lipsă trebuie tratate înainte de aplicarea algoritmului SVM.

### 2.3. Verificarea tipurilor de date și descrierea statistică

```python
print("Tipuri de date:")
print(dataset.dtypes)

dataset.describe(include="all")
```

`dtypes` indică tipul fiecărei coloane. Metoda `describe(include="all")` oferă statistici descriptive pentru coloanele numerice și informații de frecvență pentru cele categoriale.

### 2.4. Distribuția variabilei țintă

```python
print(dataset["species"].value_counts())
```

Se numără observațiile din fiecare clasă. Distribuția claselor este importantă pentru a înțelege dacă setul de date este echilibrat și pentru interpretarea ulterioară a metricilor de evaluare.

## 3. Preprocesarea datelor

SVM este un algoritm bazat pe distanțe și produse scalare. Din acest motiv, scara caracteristicilor numerice poate influența rezultatul clasificării. În plus, algoritmul nu poate utiliza direct valorile lipsă sau valorile textuale ale atributelor categoriale. În notebook, preprocesarea este realizată separat pentru coloanele numerice și categoriale.

### 3.1. Separarea caracteristicilor numerice și categoriale

```python
numeric_features = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g"
]

categorial_features = ["island", "sex"]
```

Această separare permite aplicarea unor transformări diferite, adecvate fiecărui tip de date.

### 3.2. Înlocuirea valorilor lipsă pentru atributele numerice

```python
from sklearn.impute import SimpleImputer

numericalImputer = SimpleImputer(strategy="mean")
dataset_num = numericalImputer.fit_transform(dataset[numeric_features])
```

`SimpleImputer` completează valorile numerice lipsă cu media coloanei. Media este calculată separat pentru fiecare caracteristică numerică.

### 3.3. Standardizarea atributelor numerice

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
dataset_num = scaler.fit_transform(dataset_num)
```

`StandardScaler` transformă fiecare caracteristică numerică astfel încât valorile să fie exprimate în raport cu media și abaterea standard a caracteristicii:

$$
z = \frac{x - \mu}{\sigma}
$$

unde:

- `x` este valoarea inițială;
- `μ` este media caracteristicii;
- `σ` este abaterea standard (standard deviation);
- `z` este valoarea standardizată.

Standardizarea pune caracteristicile pe scări comparabile. De exemplu, masa corporală este exprimată în grame, iar lungimile în milimetri; fără standardizare, diferențele numerice de scară ar putea influența în mod disproporționat modelul.

### 3.4. Aplicarea transformărilor numerice

```python
dataset[numeric_features] = dataset_num
dataset.describe()
```

Valorile standardizate sunt reintroduse în coloanele numerice ale setului de date.

### 3.5. Înlocuirea valorilor lipsă pentru atributele categoriale

```python
categoricalImputer = SimpleImputer(strategy="most_frequent")
dataset_cat = categoricalImputer.fit_transform(dataset[categorial_features])
dataset[categorial_features] = dataset_cat

print(dataset.isna().sum())
```

Pentru coloanele categoriale se folosește strategia `most_frequent`, care înlocuiește fiecare valoare lipsă cu valoarea cea mai frecventă din coloana respectivă (modul).

### 3.6. Codificarea atributelor categoriale – One-Hot Encoding

```python
dataset = pd.get_dummies(dataset, columns=categorial_features)
dataset.head()
```

Modelele Scikit-learn lucrează cu intrări numerice. `pd.get_dummies()` transformă fiecare categorie într-o coloană binară separată. De exemplu, valorile din `island` și `sex` sunt reprezentate prin coloane-indicator, care pot lua valorile `True`/`False` (echivalente numeric cu 1/0 în acest context).

După această etapă, setul de date conține caracteristici numerice și coloane codificate, iar `species` rămâne variabila țintă.

## 4. Clasificare binară

În acest model sunt selectate numai două dintre cele trei specii, *Adelie* și *Gentoo*. Specia *Chinstrap* nu participă la această etapă.

### 4.1. Selectarea claselor

```python
dataset_binary = dataset[
    dataset["species"].isin(["Adelie", "Gentoo"])
].copy()
```

`isin()` păstrează numai observațiile a căror specie se află în lista indicată.

### 4.2. Definirea matricii de caracteristici și a vectorului țintă

```python
X = dataset_binary.drop("species", axis=1)
y = dataset_binary["species"].map({"Adelie": 0, "Gentoo": 1})

print(X.shape)
print(y.shape)
```

- `X` conține caracteristicile utilizate de model;
- `y` conține clasa asociată fiecărei observații;
- clasa *Adelie* este codificată cu `0`, iar clasa *Gentoo* cu `1`.

Coloana `species` este eliminată din `X` pentru a evita includerea răspunsului corect printre caracteristicile de intrare.

### 4.3. Împărțirea setului de date

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25
)
```

Datele sunt împărțite în:

- **setul de antrenare** (`X_train`, `y_train`) – utilizat pentru antrenarea modelului;
- **setul de testare** (`X_test`, `y_test`) – utilizat pentru evaluarea predicțiilor.

Parametrul `test_size=0.25` alocă 25% dintre observații pentru testare și 75% pentru antrenare.

### 4.4. Definirea și antrenarea modelului SVM

```python
from sklearn.svm import SVC

model = SVC(kernel="linear")
model.fit(X_train, y_train)
```

`SVC` este implementarea din Scikit-learn a unui clasificator Support Vector Machine. În această etapă este ales kernelul liniar (`linear`). Modelul determină un hiperplan care separă cele două clase, încercând să maximizeze distanța dintre hiperplan și cele mai apropiate observații din fiecare clasă.

Observațiile aflate cel mai aproape de limita de separare se numesc **support vectors**. Acestea au un rol esențial în determinarea hiperplanului.

### 4.5. Predicții și evaluarea modelului

```python
y_pred = model.predict(X_test)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

accuracy = accuracy_score(y_test, y_pred)
print("Acuratețe: ", accuracy)

precision = precision_score(y_test, y_pred)
print("Precizie: ", precision)

recall = recall_score(y_test, y_pred)
print("Recall: ", recall)

f1 = f1_score(y_test, y_pred)
print("Scor F1: ", f1)
```

Pentru clasificarea binară, Scikit-learn consideră implicit valoarea `1` drept clasa pozitivă. În acest laborator, clasa pozitivă este *Gentoo*.

Metricile utilizate sunt:

- **Accuracy (acuratețe)** – proporția observațiilor clasificate corect din totalul observațiilor testate;
- **Precision (precizie)** – dintre observațiile prezise ca fiind pozitive (*Gentoo*), proporția celor care sunt într-adevăr pozitive;
- **Recall (rata de identificare)** – dintre toate observațiile pozitive reale, proporția celor identificate corect de model;
- **F1-score** – media armonică dintre precision și recall, utilă când este important un echilibru între cele două metrici.

## 5. Clasificare multiclasă

În aceast exemplu sunt utilizate toate cele trei clase: *Adelie*, *Chinstrap* și *Gentoo*.

### 5.1. Definirea caracteristicilor și codificarea claselor

```python
from sklearn.preprocessing import LabelEncoder

X = dataset.drop(columns="species", axis=1)
encoder = LabelEncoder()
y = encoder.fit_transform(dataset["species"])
```

Matricea `X` conține toate caracteristicile, cu excepția coloanei `species`. `LabelEncoder` transformă etichetele textuale ale speciilor în valori numerice, necesare pentru antrenarea clasificatorului.

### 5.2. Împărțirea datelor

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25
)
```

Se păstrează aceeași proporție de 75% pentru antrenare și 25% pentru testare.

### 5.3. Antrenarea modelului SVM cu kernel polinomial

```python
model = SVC(kernel="poly")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

În această etapă este utilizat kernelul polinomial (`poly`), care permite modelului să construiască limite de decizie neliniare. Acest lucru poate fi util atunci când relația dintre caracteristici și clase nu poate fi descrisă adecvat printr-un singur hiperplan liniar în spațiul inițial al caracteristicilor.

### 5.4. Evaluarea modelului multiclasă

```python
accuracy = accuracy_score(y_test, y_pred)
print("Acuratețe: ", accuracy)

precision = precision_score(y_test, y_pred, average="macro")
print("Precizie: ", precision)

recall = recall_score(y_test, y_pred, average="macro")
print("Recall: ", recall)

f1 = f1_score(y_test, y_pred, average="macro")
print("Scor F1: ", f1)
```

În clasificarea multiclasă, `average="macro"` calculează metrica separat pentru fiecare clasă, apoi face media aritmetică a rezultatelor. Fiecare specie are astfel aceeași pondere în scorul final, indiferent de numărul de observații din clasă.

## 6. SVM: hiperplan, margine și funcții kernel

### 6.1. Limita de decizie (kernel)

Support Vector Machine caută o limită de decizie care separă clasele. Pentru un clasificator liniar, limita de decizie este un **hiperplan**. În două dimensiuni, acesta este o dreaptă; în trei dimensiuni, un plan.

SVM urmărește maximizarea **marginii** (*margin*), adică a distanței dintre hiperplan și cele mai apropiate observații din clasele separate. Aceste observații apropiate de limită sunt *support vectors* și influențează direct poziția hiperplanului.

Pentru date care nu pot fi separate printr-o limită liniară, SVM poate utiliza funcții kernel. Kernelul permite calcularea similarității dintre observații într-un spațiu de caracteristici transformat, fără să fie necesară construirea explicită a transformării.

### 6.1.1. Kernelul liniar – `linear`

Kernelul liniar utilizează produsul scalar dintre două observații:

$$
K(x_i,x_j)=x_i^T x_j
$$

Acesta conduce la o limită de decizie liniară în spațiul caracteristicilor inițiale. Este o alegere potrivită atunci când clasele pot fi separate aproximativ liniar sau când numărul caracteristicilor este mare.

În notebook, kernelul liniar este folosit pentru clasificarea binară:

```python
model = SVC(kernel="linear")
```

### 6.1.2. Kernelul polinomial – `poly`

Kernelul polinomial permite modelarea unor relații neliniare între caracteristici. O formulă uzuală este:

$$
K(x_i,x_j)=(\gamma x_i^T x_j + r)^d
$$

unde:

- `gamma` controlează influența produsului scalar;
- `r` corespunde parametrului `coef0`;
- `d` este gradul polinomului, controlat prin parametrul `degree`.

Un grad mai mare permite frontiere de decizie mai complexe, dar poate crește riscul de supraînvățare (*overfitting*). În notebook, kernelul polinomial este utilizat pentru clasificarea multiclasă:

```python
model = SVC(kernel="poly")
```

În Scikit-learn, valoarea implicită pentru `degree` este `3`, pentru `gamma` este `"scale"`, iar pentru `coef0` este `0.0`.

### 6.1.3. Kernelul RBF (Radial Basis Function) – `rbf`

Kernelul RBF, numit și kernel gaussian, este utilizat frecvent pentru probleme în care limita de decizie este neliniară:

$$
K(x_i,x_j)=\exp\left(-\gamma\lVert x_i-x_j\rVert^2\right)
$$

Kernelul măsoară similaritatea dintre observații în funcție de distanța dintre ele. Parametrul `gamma` controlează cât de departe se extinde influența unei observații:

- `gamma` mic – influența fiecărei observații se extinde pe o regiune mai mare, iar frontiera tinde să fie mai netedă;
- `gamma` mare – influența este mai locală, iar frontiera poate deveni mai complexă.

Kernelul RBF nu este utilizat direct în celulele principale ale notebook-ului, dar este inclus în căutarea de hiperparametri de mai jos.

### 6.2. Parametrii importanți ai clasificatorului `SVC`

Pe lângă alegerea kernelului, performanța SVM depinde de hiperparametrii utilizați. Cei mai importanți pentru exemplele din laborator sunt `kernel` și `C`.

### 6.2.1. `kernel`

Specifică funcția kernel utilizată pentru construirea limitei de decizie. Valorile discutate în acest laborator sunt:

- `"linear"` – limită de decizie liniară;
- `"poly"` – kernel polinomial;
- `"rbf"` – kernel gaussian/RBF.

Mai multe detalii despre cele 3 funcții găsiți aici: https://scikit-learn.org/stable/modules/svm.html#svm-classification

### 6.2.2. `C` – parametrul de regularizare

`C` controlează compromisul dintre o margine cât mai mare și penalizarea observațiilor clasificate greșit sau aflate în interiorul marginii.

- **C mic** – permite mai multe încălcări ale marginii și clasificări greșite în setul de antrenare, favorizând o limită mai regulată; poate reduce supraînvățarea, dar poate conduce la subînvățare (underfitting);
- **C mare** – penalizează mai puternic erorile de antrenare și încearcă să clasifice corect mai multe observații; poate produce o limită mai complexă și poate crește riscul de supraînvățare (overfitting).

În `SVC`, valoarea implicită este `C=1.0`. În notebook nu este indicată explicit, deci este utilizată această valoare implicită.

### 6.2.3.. Alți parametri utili

- `class_weight` – permite acordarea unor ponderi diferite claselor; de exemplu, `class_weight="balanced"` ajustează ponderile în funcție de frecvența claselor;
- `probability` – dacă este `True`, activează estimarea probabilităților, cu un cost suplimentar de calcul; implicit este `False`;
- `tol` – toleranța utilizată pentru criteriul de oprire al optimizării.

## 7. Căutarea hiperparametrilor cu `GridSearchCV`

Pentru optimizarea performanței modelului SVM, se poate adăuga o etapă suplimentară de căutare sistematică a hiperparametrilor folosind `GridSearchCV`. Această etapă reprezintă o extensie opțională a notebook-ului și poate fi realizată după clasificarea multiclasă, pentru a identifica cea mai bună combinație de kernel și parametri ai modelului.

În exemplul de mai jos sunt analizate kernelurile `rbf` și `poly`, fiecare fiind testat pentru trei valori ale parametrului `C`: `0.1`, `1` și `10`.

```python
from sklearn.model_selection import GridSearchCV

params_svm = [
    {"kernel": ["rbf"], "C": [0.1, 1, 10]},
    {"kernel": ["poly"], "C": [0.1, 1, 10]}
]

grid_svm = GridSearchCV(
    SVC(),
    params_svm,
    scoring="accuracy",
    cv=5
)

grid_svm.fit(X_train, y_train)
print("Best SVM parameters:", grid_svm.best_params_)
```

`GridSearchCV` evaluează combinațiile de hiperparametri folosind validare încrucișată. Cu `cv=5`, setul de antrenare este împărțit în cinci subseturi; pe rând, fiecare subset este utilizat pentru validare, iar celelalte pentru antrenare. Scorurile obținute sunt agregate, iar combinația cu cel mai bun scor mediu este disponibilă în `best_params_`.

Modelul final poate fi construit direct din parametrii selectați:

```python
best_svm = SVC(**grid_svm.best_params_)
best_svm.fit(X_train, y_train)
y_pred = best_svm.predict(X_test)
```

Pentru problema multiclasă, metricile pot fi calculate cu media macro:

```python
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
```

Este important ca setul de testare să fie păstrat separat și să nu fie utilizat de `GridSearchCV` pentru alegerea hiperparametrilor. Căutarea se realizează numai pe datele de antrenare, iar testul este folosit pentru evaluarea finală.

## Observații privind experimentele și reproductibilitatea

1. **Rezultatele pot varia între rulări.** În notebook, `train_test_split()` nu primește `random_state`, astfel încât împărțirea datelor poate fi diferită la fiecare rulare. Pentru rezultate reproductibile se poate utiliza, de exemplu, `random_state=42`.
2. **Se poate păstra distribuția claselor la împărțire.** Pentru seturi de date în care proporțiile claselor trebuie menținute în subseturi, se poate adăuga `stratify=y` în apelul `train_test_split()`.
3. **Preprocesarea din notebook este realizată înainte de împărțirea în antrenare și testare.** Prin urmare, valorile folosite de imputare și standardizare sunt calculate pe întregul set de date. Într-un flux experimental riguros, transformările ar trebui învățate numai pe setul de antrenare și apoi aplicate setului de testare. Aceasta evită *data leakage* (scurgerea de informații din test în procesul de antrenare). Un mod uzual de implementare este folosirea `Pipeline` și `ColumnTransformer` din Scikit-learn.
4. **Kernelurile și parametrii trebuie comparați pe aceeași împărțire a datelor.** Pentru a interpreta corect diferențele dintre modele, păstrează același set de antrenare și același set de testare când compari `linear`, `poly` și `rbf` sau valori diferite ale parametrului `C`.
5. **Acuratețea nu este suficientă în toate situațiile.** Precision, recall și F1-score oferă informații suplimentare despre tipurile de erori și despre comportamentul modelului pentru fiecare clasă.

## Fișiere necesare

- `README.md` – explicațiile și codul laboratorului
- `SVM.ipynb` – notebook-ul original

# Proiect 2 — Procesarea datelor

## Pre-procesarea datelor în Python

## Prezentare generală

Preprocesarea datelor este una dintre primele etape ale unui flux de lucru de învățare automată. Datele colectate din surse reale pot conține valori lipsă, înregistrări duplicate, variabile categoriale sau caracteristici măsurate pe scări diferite. Înainte de a antrena un model, aceste probleme trebuie analizate și tratate adecvat.

În acest laborator vom parcurge principalele etape de preprocesare folosind **Python**, **pandas**, **NumPy** și **scikit-learn**. Vom încărca un set de date dintr-un fișier CSV, îi vom analiza structura, vom identifica și elimina înregistrările duplicate, vom separa caracteristicile de variabila țintă, vom trata valorile lipsă, vom transforma variabilele categoriale în reprezentări numerice și vom aplica scalarea caracteristicilor.

Exemplul folosește un set de date demonstrativ care conține următoarele coloane:

| Coloană | Tip / rol | Descriere |
|---|---|---|
| `Country` | Categorială | Țara asociată observației: France, Spain sau Germany. |
| `Age` | Numerică | Vârsta. Unele observații pot avea valori lipsă. |
| `Salary` | Numerică | Salariul. Unele observații pot avea valori lipsă. |
| `Purchased` | Categorială / țintă | Indică dacă a fost efectuată o achiziție (`Yes` sau `No`). |

Setul inițial din notebook are 11 rânduri și 4 coloane. În urma eliminării duplicatelor, rămân 10 observații. Setul este mic și are rol demonstrativ; nu trebuie interpretat ca un eșantion reprezentativ al unei populații reale.

## Obiective de învățare

La finalul laboratorului, vei putea:

1. Încărca un fișier CSV și examina dimensiunea, structura și tipurile coloanelor.
2. Identifica valorile lipsă și înregistrările duplicate.
3. Separa caracteristicile de variabila țintă.
4. Explica mai multe metode de tratare a valorilor lipsă și aplica imputarea cu media.
5. Explica de ce variabilele categoriale trebuie transformate pentru mulți algoritmi de învățare automată.
6. Aplica codificarea One-Hot pentru o caracteristică categorială și codificarea etichetelor pentru variabila țintă.
7. Aplica `MinMaxScaler` și explica rolul scalării caracteristicilor.

## Pregătirea mediului

### Varianta locală: Python și Jupyter

Instalează bibliotecile necesare folosind `pip`:

```bash
python -m pip install pandas numpy scikit-learn jupyter
```

Deschide notebook-ul Jupyter și păstrează fișierul `Dataset.csv` în același director de lucru cu notebook-ul.

### Varianta Google Colab

1. Deschide [Google Colab](https://colab.research.google.com/).
2. Crează un notebook nou în Colab (New Notebook).
3. În panoul **Files / Fișiere**, încarcă `Dataset.csv` înainte de a executa celula care citește fișierul.
4. Rulează celulele în ordinea în care apar.

> **Important:** numele fișierului și calea din `pd.read_csv("Dataset.csv")` trebuie să corespundă fișierului disponibil în sesiunea curentă.

---

## Pasul 1 — Importarea bibliotecilor

Importăm `pandas` pentru manipularea datelor și `numpy` pentru operații numerice. În etapele următoare vom importa, la nevoie, clase din `scikit-learn` pentru codificare și scalare.

```python
import pandas as pd
import numpy as np
```

---

## Pasul 2 — Încărcarea și analiza setului de date

### 2.1. Importarea setului de date

Funcția `pd.read_csv()` încarcă fișierul CSV într-un obiect `DataFrame`, o structură tabelară care permite selectarea coloanelor, calcularea statisticilor și tratarea valorilor lipsă.

```python
dataset = pd.read_csv("Dataset.csv")
```

### 2.2. Analizarea dimensiunii

Atributul `shape` returnează un tuplu cu numărul de rânduri și numărul de coloane. Pentru setul inițial din notebook, dimensiunea este `(11, 4)`.

```python
dataset.shape
```

### 2.3. Inspectarea primelor observații

Metoda `head()` afișează primele cinci rânduri implicit. Pentru a afișa primele zece rânduri, se poate folosi `dataset.head(10)`.

```python
dataset.head()
```

### 2.4. Verificarea tipurilor de date

`dataset.dtypes` arată tipul de date al fiecărei coloane. În acest set, `Country` și `Purchased` sunt coloane categoriale stocate ca șiruri de caractere, iar `Age` și `Salary` sunt numerice.

```python
data_types = dataset.dtypes
print(data_types)
```

### 2.5. Identificarea valorilor lipsă

Valorile lipsă sunt adesea reprezentate de `NaN`. Expresia `dataset.isnull()` produce un tabel de valori booleene, iar `sum()` numără valorile lipsă din fiecare coloană.

```python
isnull = dataset.isnull().sum()
print(isnull)
```

În setul de date folosit de notebook, `Age` și `Salary` au câte o valoare lipsă. Celelalte coloane nu au valori lipsă.

### 2.6. Statistici descriptive

`describe(include="all")` afișează un rezumat atât pentru coloanele numerice, cât și pentru cele categoriale. Pentru coloanele numerice include, printre altele, media, abaterea standard, minimul, quartilele și maximul. Pentru coloanele categoriale afișează numărul de valori, valorile unice, categoria cea mai frecventă și frecvența ei. Statisticile care nu se aplică unui tip de coloană sunt afișate ca `NaN`.

```python
dataset.describe(include="all")
```

### 2.7. Eliminarea rândurilor duplicate

Duplicatele pot influența statisticile și pot face ca anumite observații să fie reprezentate de mai multe ori decât ar trebui. `drop_duplicates()` returnează un tabel din care au fost eliminate rândurile identice. În notebook, rezultatul este atribuit din nou variabilei `dataset`.

```python
dataset = dataset.drop_duplicates()
dataset.describe()
```

După eliminarea duplicatelor, setul de date conține 10 rânduri. Verifică întotdeauna dacă două înregistrări identice sunt într-adevăr duplicate nedorite; în anumite contexte, observații care arată la fel pot fi evenimente distincte.

### 2.8. Separarea caracteristicilor și a variabilei țintă

În acest exemplu:

- `X` conține caracteristicile de intrare (`Country`, `Age` și `Salary`);
- `Y` conține variabila țintă/dependentă (`Purchased`), pe care am putea dori să o prezicem într-o etapă ulterioară.

Este util să separăm caracteristicile de țintă înainte de a aplica transformări diferite asupra lor.

```python
X = dataset[['Country', 'Age', 'Salary']]
Y = dataset['Purchased']
```

### 2.9. Inspectarea caracteristicilor și a țintei

```python
print(X)
```

```python
print(Y)
```

În acest punct, `X` este un `DataFrame` cu trei coloane, iar `Y` este o serie pandas cu etichetele `No` și `Yes`. În fișier pot exista spații suplimentare în unele valori textuale; curățarea spațiilor este o operație uzuală înainte de codificarea categoriilor.

---

## Pasul 3 — Tratarea valorilor lipsă

### De ce trebuie tratate valorile lipsă?

Valorile lipsă apar frecvent în datele colectate prin sondaje, măsurători sau eșantionare. Mulți algoritmi de învățare automată nu acceptă valori `NaN` în caracteristici, astfel că acestea trebuie tratate înainte de antrenarea modelului. Dacă sunt ignorate, datele lipsă pot reduce numărul de observații disponibile, pot introduce distorsiuni sau pot afecta calitatea rezultatelor.

Alegerea metodei potrivite depinde de numărul și distribuția valorilor lipsă, de tipul variabilei și de motivul pentru care informațiile lipsesc.

### Metode uzuale

#### 1. Eliminarea observațiilor cu valori lipsă

Se pot elimina rândurile care conțin cel puțin o valoare lipsă. Metoda poate fi rezonabilă când setul de date este mare și numai câteva observații sunt incomplete. Dacă lipsesc multe valori, eliminarea rândurilor poate produce o pierdere importantă de informație sau poate schimba distribuția eșantionului.

#### 2. Înlocuirea cu media

Pentru o caracteristică numerică, fiecare valoare lipsă poate fi înlocuită cu media valorilor disponibile din acea coloană. Metoda este simplă și păstrează numărul de rânduri. Totuși, poate reduce artificial dispersia și nu păstrează în general relațiile de corelație dintre variabile. Este sensibilă la valori extreme.

#### 3. Imputarea de tip Hot Deck

O valoare lipsă este înlocuită cu o valoare observată într-o altă înregistrare care are caracteristici similare. Valoarea imputată provine astfel din date observate și poate fi aleasă dintre observații comparabile. Metoda cere o regulă justificată pentru identificarea înregistrărilor similare.

### 3.1. Imputarea mediei pentru `Age`

În acest laborator, setul de date este mic, deci eliminarea unei observații ar însemna să pierdem o parte relativ importantă din date. Pentru exemplul demonstrativ, înlocuim valoarea lipsă din `Age` cu media valorilor observate din această coloană.

```python
mean_age = X['Age'].mean()
X.fillna({'Age': mean_age}, inplace=True)
print(X)
```

Media vârstei calculată din observațiile disponibile este aproximativ `38.7778`. Valoarea lipsă este înlocuită cu această medie.

### 3.2. Imputarea mediei pentru `Salary`

Aplicăm același procedeu coloanei `Salary`.

```python
mean_salary = X['Salary'].mean()
X.fillna({'Salary': mean_salary}, inplace=True)
print(X)
```

Media salariului calculată din valorile disponibile este aproximativ `63777.7778`. După această operație, `X` nu mai conține valori lipsă în coloanele numerice.

> **Limitare importantă:** imputarea cu media este folosită aici pentru a ilustra tehnica. Într-un proiect real, trebuie verificat de ce lipsesc datele și dacă media este o alegere rezonabilă. În evaluarea unui model, valorile de imputare trebuie calculate doar pe datele de antrenare și apoi aplicate datelor de validare/test, pentru a evita scurgerea de informație (*data leakage*).

---

## Pasul 4 — Transformarea variabilelor categoriale

Multe metode de învățare automată operează cu intrări numerice. În acest exemplu, `Country` și `Purchased` sunt variabile categoriale, deci trebuie reprezentate numeric înainte de a le folosi cu astfel de algoritmi.

### 4.1. Codificarea One-Hot pentru `Country`

O codificare simplă, de tipul France = 0, Spain = 1 și Germany = 2, ar introduce o ordine numerică artificială între țări. Pentru această caracteristică nominală, nu există o asemenea ordine. **One-Hot Encoding** creează câte o coloană indicator pentru fiecare categorie, cu valori `True`/`False` (sau `1`/`0`, în funcție de reprezentare).

```python
X = pd.get_dummies(X, columns=['Country'])  # One-Hot Encoding
print(X)
```

Rezultatul conține coloane precum `Country_France`, `Country_Germany` și `Country_Spain`. Pentru fiecare observație, coloana corespunzătoare țării sale este activă, iar celelalte sunt inactive. În această reprezentare, categoriile nu mai sunt codificate ca numere ordonate.

Ordinea coloanelor generate de `pd.get_dummies()` este stabilită de pandas, de regulă în ordine alfabetică. De aceea, este mai sigur să interpretezi numele coloanelor decât să presupui că o categorie corespunde unei anumite poziții fără să verifici rezultatul.

### 4.2. Codificarea variabilei țintă `Purchased`

Pentru acest exemplu binar, folosim `LabelEncoder` pentru a transforma etichetele textuale ale variabilei țintă în etichete numerice.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()  # Transformă etichetele în valori numerice
Y = le.fit_transform(Y.astype(str).str.strip())
print(Y)
```

`LabelEncoder` atribuie codurile pe baza ordinii sortate a claselor. În acest set, `No` este codificat ca `0`, iar `Yes` ca `1`. Folosirea `.str.strip()` elimină spațiile suplimentare de la începutul sau sfârșitul valorilor textuale înainte de codificare.

Pentru a verifica asocierea exactă dintre coduri și etichete, rulează:

```python
print("Clasele și ordinea lor:", le.classes_)
```

> **Notă:** `LabelEncoder` este potrivit aici pentru codificarea variabilei țintă `Y`. Pentru caracteristici nominale precum `Country`, One-Hot Encoding evită interpretarea greșită a etichetelor ca o scară numerică ordonată.

---

## Pasul 5 — Scalarea caracteristicilor

Caracteristicile numerice pot avea scări foarte diferite. În exemplul nostru, valorile `Age` sunt în jurul zecilor, în timp ce `Salary` este de ordinul zecilor de mii. Pentru algoritmi sensibili la scară, de exemplu cei care folosesc distanțe sau regularizare, o caracteristică cu valori numeric mari poate domina funcția de învățare.

`MinMaxScaler` transformă o caracteristică astfel încât minimul observat să devină `0`, iar maximul observat să devină `1`. Valorile intermediare sunt mapate în același interval.

### 5.1. Scalarea caracteristicilor din `X` — varianta din notebook

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler() # Scalează valorile într-un interval între 0 și 1
X_all_scaled = scaler.fit_transform(X) 

print(X_all_scaled)

```

După această operație, `X` devine un tablou NumPy, nu un `DataFrame`; numele coloanelor nu mai sunt atașate automat rezultatului. Pentru acest set de date, coloanele One-Hot sunt deja binare (`0`/`1` sau `False`/`True`), astfel că valorile lor rămân la capetele intervalului.

```python
X_scaled = X.copy()
scaler = MinMaxScaler()

X_scaled[['Age', 'Salary']] = scaler.fit_transform(
    X_scaled[['Age', 'Salary']]
)

print(X_scaled)
```

> **Notă:** într-un flux real de machine learning, `fit()` pentru scalare trebuie aplicat doar setului de antrenare; apoi se folosește `transform()` pentru seturile de validare și test. În acest laborator demonstrativ scalarea este aplicată întregului set pentru a ilustra mecanismul.

---

## Rezumatul fluxului de preprocesare

La sfârșitul laboratorului, am parcurs următorii pași:

1. Am importat bibliotecile și am încărcat datele CSV.
2. Am analizat dimensiunea, tipurile de date, valorile lipsă și statisticile descriptive.
3. Am eliminat înregistrările duplicate.
4. Am separat caracteristicile (`X`) de variabila țintă (`Y`).
5. Am înlocuit valorile lipsă din `Age` și `Salary` cu mediile coloanelor.
6. Am transformat `Country` folosind One-Hot Encoding.
7. Am codificat `Purchased` ca valori numerice `0` și `1`.
8. Am aplicat scalarea caracteristicilor.

Preprocesarea nu este o secvență identică pentru toate proiectele. Ordinea și alegerea operațiilor depind de tipul datelor, de algoritmul care va fi folosit și de obiectivul analizei.

## Fișiere

- `README.md` — explicațiile și codul laboratorului
- `Pre_procesarea_datelor.ipynb` — notebook-ul original pentru execuție interactivă.
- `Dataset.csv` — setul de date folosit în exercițiu.

Asigură-te că `Dataset.csv` este disponibil în directorul de lucru al notebook-ului sau în panoul **Files / Fișiere** din Google Colab. Codul din README este prezentat în blocuri separate pentru a putea fi copiat în celule Jupyter în ordinea indicată.

## Documentație suplimentară

- [Pandas — `read_csv`](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)
- [Pandas — valori lipsă](https://pandas.pydata.org/docs/user_guide/missing_data.html)
- [Pandas — `get_dummies`](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html)
- [Scikit-learn — `LabelEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
- [Scikit-learn — `MinMaxScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)

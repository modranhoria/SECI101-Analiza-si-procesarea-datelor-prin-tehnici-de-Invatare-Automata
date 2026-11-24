# Proiect 5 –  Arbori Decizionali (Decision Trees)

## Obiectivul laboratorului

În acest laborator este studiat algoritmul de învățare automată **Arbori Decizionali (Decision Trees)** pentru clasificarea speciilor de pinguini din setul de date **Penguins**, disponibil prin biblioteca Seaborn.

Sunt abordate două tipuri de probleme:

1. **clasificare binară** – se păstrează numai speciile *Adelie* și *Gentoo*;
2. **clasificare multiclasă** – se utilizează toate cele trei specii: *Adelie*, *Chinstrap* și *Gentoo*.

În laborator sunt explorate datele, sunt tratate valorile lipsă și sunt codificate caracteristicile categoriale. Ulterior, se antrenează și se evaluează modele de tip Decision Tree folosind biblioteca Scikit-learn. Sunt analizate structura arborelui, criteriile de divizare, importanța caracteristicilor și influența principalilor hiperparametri asupra complexității modelului.

Spre deosebire de SVM, arborii decizionali **nu necesită standardizarea caracteristicilor numerice**, deoarece deciziile sunt luate prin praguri aplicate individual fiecărei caracteristici, nu prin calculul distanțelor dintre observații.

## 1. Importarea bibliotecilor și a setului de date Penguins

Pentru manipularea și explorarea datelor sunt utilizate NumPy, Pandas, Matplotlib și Seaborn. Din Scikit-learn sunt folosite funcții pentru preprocesare, împărțirea datelor, antrenarea arborelui decizional și evaluarea predicțiilor.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
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
print("Lista de coloane:", dataset.columns.to_list())
```

Această instrucțiune afișează numele atributelor disponibile în setul de date.

### 2.2. Verificarea valorilor lipsă

```python
print("Valori lipsă:")
print(dataset.isna().sum())
```

`isna()` identifică valorile lipsă, iar `sum()` calculează numărul acestora pentru fiecare coloană. Valorile lipsă trebuie tratate înainte de antrenarea clasificatorului.

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

sns.countplot(data=dataset, x="species")
plt.title("Distribuția speciilor de pinguini")
plt.xlabel("Specie")
plt.ylabel("Număr de observații")
plt.show()
```

Se numără observațiile din fiecare clasă și se reprezintă grafic distribuția lor. Distribuția claselor este importantă pentru alegerea strategiei de împărțire a datelor și pentru interpretarea metricilor de evaluare.

## 3. Preprocesarea datelor

Înainte de aplicarea algoritmului, valorile lipsă trebuie tratate, iar caracteristicile categoriale trebuie transformate în valori numerice. Spre deosebire de SVM, **arborii decizionali nu necesită standardizarea caracteristicilor numerice**. Ei construiesc reguli de forma `caracteristică <= prag`, astfel încât schimbarea unității de măsură nu impune standardizarea valorilor.

În această etapă, preprocesarea este realizată separat pentru coloanele numerice și categoriale, urmând aceeași succesiune de pași ca în exemplul SVM.

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

### 3.3. Aplicarea transformărilor numerice

```python
dataset[numeric_features] = dataset_num
dataset.describe()
```

Valorile completate sunt reintroduse în coloanele numerice ale setului de date.

**Observație:** nu se aplică `StandardScaler` sau `MinMaxScaler`. Arborii decizionali nu compară distanțele dintre observații, ci caută praguri de separare pentru fiecare caracteristică. Prin urmare, standardizarea nu este necesară pentru acest algoritm.

### 3.4. Înlocuirea valorilor lipsă pentru atributele categoriale

```python
categoricalImputer = SimpleImputer(strategy="most_frequent")
dataset_cat = categoricalImputer.fit_transform(dataset[categorial_features])
dataset[categorial_features] = dataset_cat

print(dataset.isna().sum())
```

Pentru coloanele categoriale se folosește strategia `most_frequent`, care înlocuiește fiecare valoare lipsă cu valoarea cea mai frecventă din coloana respectivă (modul).

### 3.5. Codificarea atributelor categoriale – One-Hot Encoding

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
    test_size=0.25,
    random_state=42,
    stratify=y
)
```

Datele sunt împărțite în:

- **setul de antrenare** (`X_train`, `y_train`) – utilizat pentru construirea arborelui;
- **setul de testare** (`X_test`, `y_test`) – utilizat pentru evaluarea predicțiilor.

Parametrul `test_size=0.25` alocă 25% dintre observații pentru testare și 75% pentru antrenare. `random_state=42` asigură reproductibilitatea împărțirii, iar `stratify=y` păstrează proporția claselor în cele două subseturi.

### 4.4. Definirea și antrenarea modelului Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)

model.fit(X_train, y_train)
```

`DecisionTreeClassifier` este implementarea din Scikit-learn a algoritmului de clasificare bazat pe arbori decizionali. Arborele identifică succesiv caracteristicile și pragurile care împart cel mai bine observațiile în clase cât mai omogene.

În acest exemplu este utilizat criteriul `gini`, care măsoară impuritatea claselor din fiecare nod. Parametrul `random_state=42` permite reproducerea rezultatelor.

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

precision = precision_score(y_test, y_pred, zero_division=0)
print("Precizie: ", precision)

recall = recall_score(y_test, y_pred, zero_division=0)
print("Recall: ", recall)

f1 = f1_score(y_test, y_pred, zero_division=0)
print("Scor F1: ", f1)
```

Pentru clasificarea binară, Scikit-learn consideră implicit valoarea `1` drept clasa pozitivă. În acest laborator, clasa pozitivă este *Gentoo*.

Metricile utilizate sunt:

- **Accuracy (acuratețe)** – proporția observațiilor clasificate corect din totalul observațiilor testate;
- **Precision (precizie)** – dintre observațiile prezise ca fiind pozitive (*Gentoo*), proporția celor care sunt într-adevăr pozitive;
- **Recall (rata de identificare)** – dintre toate observațiile pozitive reale, proporția celor identificate corect de model;
- **F1-score** – media armonică dintre precision și recall, utilă când este important un echilibru între cele două metrici.

Matricea de confuzie oferă detalii despre numărul de observații clasificate corect și despre tipurile de erori:

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Adelie", "Gentoo"],
    yticklabels=["Adelie", "Gentoo"]
)
plt.xlabel("Clasa prezisă")
plt.ylabel("Clasa reală")
plt.title("Matricea de confuzie – clasificare binară")
plt.show()
```

## 5. Clasificare multiclasă

În această etapă sunt utilizate toate cele trei clase: *Adelie*, *Chinstrap* și *Gentoo*.

### 5.1. Definirea caracteristicilor și codificarea claselor

```python
from sklearn.preprocessing import LabelEncoder

X = dataset.drop("species", axis=1)
encoder = LabelEncoder()
y = encoder.fit_transform(dataset["species"])

print("Clase:", encoder.classes_)
```

Matricea `X` conține toate caracteristicile, cu excepția coloanei `species`. `LabelEncoder` transformă etichetele textuale ale speciilor în valori numerice, necesare pentru antrenarea și evaluarea clasificatorului.

### 5.2. Împărțirea datelor

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)
```

Se păstrează 75% dintre observații pentru antrenare și 25% pentru testare. Prin `stratify=y`, distribuția celor trei specii este menținută aproximativ în ambele subseturi.

### 5.3. Antrenarea modelului pentru clasificare multiclasă

```python
model_multi = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)

model_multi.fit(X_train, y_train)
y_pred = model_multi.predict(X_test)
```

`DecisionTreeClassifier` gestionează direct clasificarea multiclasă. La fiecare nod, algoritmul caută o împărțire care să reducă impuritatea claselor, indiferent dacă problema are două sau mai multe clase.

### 5.4. Evaluarea modelului multiclasă

```python
accuracy = accuracy_score(y_test, y_pred)
print("Acuratețe: ", accuracy)

precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
print("Precizie: ", precision)

recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
print("Recall: ", recall)

f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
print("Scor F1: ", f1)
```

În clasificarea multiclasă, `average="macro"` calculează metrica separat pentru fiecare clasă, apoi face media aritmetică a rezultatelor. Fiecare specie are astfel aceeași pondere în scorul final, indiferent de numărul de observații din clasă.

Matricea de confuzie poate fi reprezentată astfel:

```python
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_
)
plt.xlabel("Clasa prezisă")
plt.ylabel("Clasa reală")
plt.title("Matricea de confuzie – clasificare multiclasă")
plt.show()
```

## 6. Înțelegerea și vizualizarea arborelui decizional

### 6.1. Structura unui arbore decizional

Un arbore decizional clasifică observațiile printr-o succesiune de reguli. Fiecare nod intern verifică o condiție asupra unei caracteristici, ramurile reprezintă rezultatele condiției, iar frunzele indică predicția finală.

Elementele principale sunt:

- **root (rădăcină)** – primul nod al arborelui, de la care începe clasificarea;
- **decision node (nod de decizie)** – nod în care se verifică o condiție de forma `caracteristică <= prag`;
- **branch (ramură)** – traseul urmat în funcție de rezultatul condiției;
- **leaf (frunză)** – nod terminal care furnizează clasa prezisă;
- **depth (adâncime)** – numărul de niveluri parcurse de la rădăcină până la un nod.

La fiecare pas, algoritmul alege o caracteristică și un prag care împart observațiile în grupuri cât mai omogene. Procesul continuă până când sunt îndeplinite condițiile de oprire stabilite prin hiperparametri sau până când nodurile devin suficient de pure.

### 6.2. Criterii de divizare: Gini și Entropy

Pentru a decide cât de bună este o împărțire, algoritmul măsoară impuritatea claselor dintr-un nod.

**Indicele Gini** estimează probabilitatea unei clasificări greșite dacă eticheta unei observații ar fi aleasă aleatoriu conform distribuției claselor din nod:

$$
Gini = 1 - \sum_{k=1}^{K} p_k^2
$$

unde `p_k` reprezintă proporția observațiilor din clasa `k`, iar `K` este numărul de clase. Valoarea Gini este zero atunci când toate observațiile din nod aparțin aceleiași clase.

**Entropia** măsoară incertitudinea distribuției claselor:

$$
Entropy = -\sum_{k=1}^{K} p_k \log_2(p_k)
$$

Entropia este zero pentru un nod pur și crește atunci când clasele sunt distribuite mai uniform.

În Scikit-learn, parametrul `criterion` permite alegerea criteriului folosit pentru selectarea divizărilor:

- `"gini"` – utilizează impuritatea Gini;
- `"entropy"` – utilizează entropia informațională;
- `"log_loss"` – utilizează reducerea log-loss.

În multe aplicații, Gini și Entropy pot conduce la rezultate apropiate. Alegerea potrivită depinde de setul de date și poate fi testată experimental.

### 6.3. Reprezentarea grafică a arborelui

Arborele antrenat poate fi reprezentat cu funcția `plot_tree()`. Pentru exemplul multiclasă:

```python
tree_model = model_multi
feature_names = X_train.columns

plt.figure(figsize=(22, 12))
plot_tree(
    tree_model,
    feature_names=list(feature_names),
    class_names=encoder.classes_,
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=9
)
plt.title("Arbore decizional – primele niveluri")
plt.show()
```

`feature_names` furnizează numele caracteristicilor după preprocesare. `filled=True` colorează nodurile în funcție de clasa predominantă, iar `max_depth=3` limitează reprezentarea la primele niveluri pentru a păstra diagrama lizibilă. Această limitare afectează numai desenul, nu și modelul deja antrenat.

### 6.4. Regulile de clasificare

Regulile învățate de arbore pot fi afișate sub formă de text cu `export_text()`:

```python
rules = export_text(
    tree_model,
    feature_names=list(feature_names),
    max_depth=4
)
print(rules)
```

Regulile permit interpretarea traseului parcurs de o observație și identificarea condițiilor care conduc la o anumită clasă.

### 6.5. Importanța caracteristicilor

Un arbore decizional estimează importanța caracteristicilor în funcție de contribuția acestora la reducerea impurității în nodurile în care sunt utilizate.

```python
feature_importance = pd.Series(
    tree_model.feature_importances_,
    index=list(feature_names)
).sort_values(ascending=False)

print(feature_importance)

feature_importance.sort_values().plot(
    kind="barh",
    figsize=(9, 5),
    title="Importanța caracteristicilor"
)
plt.xlabel("Importanță")
plt.show()
```

Valorile mai mari indică o contribuție mai importantă la divizările arborelui. Importanța caracteristicilor trebuie interpretată cu atenție: ea descrie utilizarea caracteristicilor în arborele antrenat și nu demonstrează, de una singură, existența unei relații cauzale.

### 6.6. Parametrii importanți ai `DecisionTreeClassifier`

Performanța și complexitatea unui arbore decizional depind de hiperparametrii utilizați. Un arbore prea complex poate memora observațiile din setul de antrenare, fenomen numit **supraînvățare (overfitting)**. Un arbore prea simplu poate să nu surprindă suficient structura datelor, fenomen numit **subînvățare (underfitting)**.

#### 6.6.1. `criterion` – criteriul de divizare

Stabilește măsura utilizată pentru evaluarea calității unei divizări:

- `"gini"` – impuritatea Gini;
- `"entropy"` – entropia informațională;
- `"log_loss"` – criteriul bazat pe log-loss.

#### 6.6.2. `max_depth` – adâncimea maximă

Limitează numărul nivelurilor arborelui:

- `None` – arborele crește până când sunt îndeplinite alte condiții de oprire;
- valori mici (de exemplu `3`, `5`) – produc arbori mai simpli, mai ușor de interpretat;
- valori mari – permit reguli mai detaliate, dar pot crește riscul de overfitting.

#### 6.6.3. `min_samples_split` – numărul minim de observații pentru divizarea unui nod

Stabilește numărul minim de observații pe care trebuie să le conțină un nod pentru a putea fi împărțit:

- valoare mică – permite mai multe divizări și un arbore mai complex;
- valoare mare – limitează divizările și poate reduce overfitting-ul.

#### 6.6.4. `min_samples_leaf` – numărul minim de observații într-o frunză

Stabilește numărul minim de observații care trebuie să rămână într-un nod terminal:

- `1` permite frunze care conțin o singură observație;
- valori mai mari produc frunze cu mai multe observații și reguli, de regulă, mai puțin specifice.

## 7. Optimizarea hiperparametrilor cu `GridSearchCV`

Pentru identificarea unei configurații potrivite se poate utiliza `GridSearchCV`. Acesta testează sistematic combinații de hiperparametri și estimează performanța fiecărei combinații prin validare încrucișată.

În exemplul de mai jos sunt comparate criteriile `gini` și `entropy`, mai multe valori pentru `max_depth`, `min_samples_split` și `min_samples_leaf`.

```python
params_tree = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 3, 5, 7],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid_tree = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=params_tree,
    scoring="accuracy",
    cv=5,
    n_jobs=-1
)

grid_tree.fit(X_train, y_train)

print("Cei mai buni parametri:", grid_tree.best_params_)
print("Cea mai bună acuratețe medie CV:", grid_tree.best_score_)
```

Deoarece `GridSearchCV` primește direct un obiect `DecisionTreeClassifier`, parametrii sunt indicați fără prefix. De exemplu, `max_depth` se referă direct la adâncimea maximă a arborelui.

Cu `cv=5`, setul de antrenare este împărțit în cinci subseturi. Pe rând, fiecare subset este utilizat pentru validare, iar celelalte patru pentru antrenare. Scorurile sunt agregate, iar combinația cu cel mai bun scor mediu este disponibilă în `best_params_` și `best_score_`.

### Evaluarea celui mai bun model

```python
best_tree = grid_tree.best_estimator_
y_pred_best = best_tree.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_best)
precision = precision_score(y_test, y_pred_best, average="macro", zero_division=0)
recall = recall_score(y_test, y_pred_best, average="macro", zero_division=0)
f1 = f1_score(y_test, y_pred_best, average="macro", zero_division=0)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
```

Modelul selectat prin validare încrucișată este evaluat o singură dată pe setul de testare, păstrat separat de procesul de optimizare. În acest exemplu se utilizează mediile macro pentru Precision, Recall și F1-score, deoarece problema este multiclasă. Preprocesarea a fost efectuată anterior pe setul complet, conform pașilor laboratorului; pentru evaluări experimentale riguroase, aceasta trebuie inclusă într-un `Pipeline` și recalculată în interiorul fiecărui fold.

## Observații privind experimentele și reproductibilitatea

1. **Rezultatele depind de împărțirea datelor.** În exemple este utilizat `random_state=42`, astfel încât împărțirea în seturi de antrenare și testare să poată fi reprodusă.
2. **Distribuția claselor este păstrată.** Parametrul `stratify=y` din `train_test_split()` menține aproximativ proporțiile claselor în cele două subseturi.
3. **Ordinea preprocesării trebuie avută în vedere.** În exemplul de laborator, imputarea și codificarea sunt prezentate înainte de împărțirea în seturi, pentru a păstra succesiunea didactică a pașilor. Într-un flux experimental riguros, parametrii de imputare și codificare ar trebui învățați numai pe setul de antrenare și apoi aplicați setului de testare. Acest lucru se poate realiza cu `Pipeline` și `ColumnTransformer` și reduce riscul de *data leakage*.
4. **Arborii decizionali nu necesită standardizare.** Spre deosebire de SVM sau KNN, arborele folosește praguri pentru fiecare caracteristică și nu calculează distanțe între observații.
5. **Un scor foarte bun pe setul de antrenare nu garantează generalizarea.** Compară rezultatele de antrenare, validare și testare și urmărește efectul parametrilor `max_depth` și `min_samples_leaf`.
6. **Acuratețea nu este suficientă în toate situațiile.** Precision, Recall, F1-score și matricea de confuzie oferă informații suplimentare despre erorile modelului și performanța pentru fiecare clasă.
7. **Importanța caracteristicilor nu indică o relație cauzală.** Ea reflectă modul în care arborele a utilizat caracteristicile pentru realizarea divizărilor.

## Fișiere necesare

- `README.md` – explicațiile și codul laboratorului;
- `Decision_Trees.ipynb` – notebook-ul cu implementarea și experimentele laboratorului.

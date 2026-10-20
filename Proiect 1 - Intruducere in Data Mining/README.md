# Introducere în Data Mining

## Pregătirea mediului

Dacă lucrezi local, instalează bibliotecile necesare cu:

```bash
python -m pip install numpy scipy pandas matplotlib jupyter
```

Pentru partea despre fișiere CSV, păstrează `Dataset.csv` în același director de lucru cu notebook-ul/scriptul. 

În Google Colab (https://colab.research.google.com/) , încarcă fișierul `Dataset.csv` în panoul **Files/Fișiere** înainte de a rula secțiunea respectivă. Fișierul CSV este inclus în arhiva pachetului laboratorului.

---

## Prezentare generală

Acest laborator prezintă tehnici de bază pentru explorarea unui set de date înainte de aplicarea unor metode mai avansate de data mining. Vei calcula statistici descriptive, vei genera și vizualiza un eșantion aleatoriu simplu, vei construi o diagramă de dispersie, vei încărca un fișier CSV și vei inspecta coloanele numerice și categoriale.

Materialul este conceput pentru lucru practic. Execută celulele în ordine, citește fiecare explicație, examinează rezultatele și rezolvă sarcinile scurte din notebook.

## Obiective de învățare

La finalul acestui laborator, vei putea:

1. Explica ce reprezintă o serie de date unidimensională și calcula statistici descriptive uzuale.
2. Deosebi media, mediana și mode și descrie situațiile în care fiecare este util.
3. Explica varianța și abaterea standard ca măsuri ale dispersiei datelor.
4. Calcula percentile și interpreta percentila 50 ca mediană.
5. Genera date aleatorii reproductibile și explora o distribuție cu ajutorul unei histograme.
6. Crea și interpreta o diagramă de dispersie care prezintă relația dintre două variabile.
7. Încărca un set de date CSV cu pandas și inspecta structura, statisticile descriptive și valorile lipsă.

## Parcurgerea laboratorului

1. Execută fiecare celulă de cod de sus în jos.
2. Citește rezultatul înainte de a continua.
3. Modifică valorile sau parametrii acolo unde ți se solicită și compară rezultatele.

**Important:** Fișierul CSV `Dataset.csv` trebuie să fie disponibil în același director de lucru cu notebook-ul. Dacă folosești Google Colab, încarcă ambele fișiere în sesiune folosind panoul **Files/Fișiere** sau descarcă-le din repository.

---

## Partea 1 — Statistici descriptive

O **serie** este o colecție unidimensională de valori. În primul exemplu, `speed` este o listă Python care reprezintă zece observații numerice. Valorile sunt ilustrative; nu este specificată o unitate de măsură sau o sursă din lumea reală.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from scipy import stats
from pathlib import Path

speed = [33, 66, 99, 132, 61, 53, 80, 49, 139, 80]

print("Număr de observații:", len(speed))
print("Observații:", speed)
```

### 1. Media aritmetică

**Media aritmetică** este suma tuturor observațiilor împărțită la numărul acestora. Folosește fiecare valoare, astfel că valorile neobișnuit de mari sau de mici o pot influența semnificativ.

Formula:

\[
\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i
\]

Execută celula și notează rezultatul. Apoi gândește-te cum s-ar modifica media dacă una dintre observații ar fi mult mai mare decât celelalte.

```python
mean_value = np.mean(speed)
print("Media aritmetică:", mean_value)
```

**Experiment:** Adaugă într-o listă nouă o copie a celei mai mari valori înmulțită cu zece și compară media acestei liste cu media inițială. Nu suprascrie lista originală `speed`; păstrează comparația reproductibilă.

```python
# Experiment opțional: compară media inițială cu media afectată de o valoare extremă.
speed_with_extreme_value = speed + [max(speed) * 10]
print("Media inițială:", np.mean(speed))
print("Media după adăugarea unei valori extreme:", np.mean(speed_with_extreme_value))
```

### 2. Mediana

**Mediana** este valoarea din mijloc după sortarea observațiilor. Dacă numărul valorilor este par, mediana este media aritmetică a celor două valori centrale. În general, mediana este mai puțin sensibilă la valorile extreme decât media.

```python
median_value = np.median(speed)
print("Mediana:", median_value)
```

**Experiment:** În experimentul anterior, mediana s-a modificat la fel de mult ca media? De ce este de preferat mediana în locul mediei pentru un set de date care conține valori extreme?

### 3. Mode

**Mode** este valoarea care apare cel mai frecvent. Un set de date poate avea un singur mod, mai multe moduri sau nicio valoare repetată. În acest exemplu, `80` apare de două ori, iar toate celelalte valori apar o singură dată.

`scipy.stats.mode` returnează atât moda, cât și frecvența sa. Argumentul `keepdims=True` păstrează valorile returnate în tablouri, ceea ce face indexarea explicită și compatibilă cu versiunile actuale SciPy.

```python
mode_result = stats.mode(speed, keepdims=True)
mode_value = mode_result.mode[0]
mode_count = mode_result.count[0]
print("Mode:", mode_value)
print("Frecvența mode-ului:", mode_count)
```

### 4. Varianța

**Varianța** măsoară cât de răspândite sunt valorile în jurul mediei. Pentru a calcula varianța unui șir, scade media din fiecare observație, ridică diferențele la pătrat și calculează media lor:

\[
\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2
\]

În mod implicit, `np.var()` folosește formula pentru populație și împarte la `n` (`ddof=0`). Dacă valorile sunt tratate ca un eșantion folosit pentru estimarea unei populații mai mari, o alternativă uzuală este varianța eșantionului, care împarte la `n - 1` (`ddof=1`). Alegerea corectă depinde de ceea ce reprezintă datele.

#### Exemplu pas cu pas: calculul manual al varianței

Considerăm următoarele șapte valori: **32, 111, 138, 28, 59, 77, 97**. În acest exemplu calculăm **varianța**, deci la final împărțim suma pătratelor diferențelor la numărul total de valori, `n = 7`.

**Pasul 1 — Calculăm media aritmetică.** Adunăm toate valorile și împărțim suma la numărul lor:

\[
\bar{x} = \frac{32+111+138+28+59+77+97}{7} = \frac{542}{7} \approx 77.4
\]

**Pasul 2 — Scădem media din fiecare valoare.** Aceste diferențe arată cât de departe este fiecare observație de media calculată:

```text
 32 - 77.4 = -45.4
111 - 77.4 =  33.6
138 - 77.4 =  60.6
 28 - 77.4 = -49.4
 59 - 77.4 = -18.4
 77 - 77.4 =  -0.4
 97 - 77.4 =  19.6
```

**Pasul 3 — Ridicăm fiecare diferență la pătrat.** Pătratul elimină semnul diferenței, astfel încât abaterile pozitive și negative să nu se anuleze reciproc:

```text
(-45.4)² = 2061.16
 (33.6)² = 1128.96
 (60.6)² = 3672.36
(-49.4)² = 2440.36
(-18.4)² =  338.56
 (-0.4)² =    0.16
 (19.6)² =  384.16
```

**Pasul 4 — Calculăm media pătratelor diferențelor.** Adunăm pătratele și împărțim la `7`:

\[
\sigma^2 = \frac{2061.16+1128.96+3672.36+2440.36+338.56+0.16+384.16}{7}
\]

\[
\sigma^2 = \frac{10025.72}{7} \approx 1432.2
\]

Prin urmare, **varianța este aproximativ `1432.2`**. Rezultatul este aproximativ deoarece media a fost rotunjită la `77.4` în pașii intermediari. Dacă păstrăm mai multe zecimale pe parcurs, rezultatul final rotunjit la o zecimală rămâne `1432.2`.

> **Atenție:** pentru varianța unui eșantion se împarte la `n - 1 = 6`, nu la `n = 7`. În exemplul de mai sus calculăm explicit varianța populației.

Codul următor calculează varianța și varianța eșantionului pentru lista `speed` folosită în exercițiul principal:

```python
population_variance = np.var(speed)
sample_variance = np.var(speed, ddof=1)
print("Varianța populației (ddof=0):", population_variance)
print("Varianța eșantionului (ddof=1):", sample_variance)
```

### 5. Abaterea standard

**Abaterea standard** este rădăcina pătrată a varianței. Descrie dispersia folosind aceeași unitate de măsură ca observațiile inițiale, ceea ce o face adesea mai ușor de interpretat decât varianța. Ca și în cazul varianței, NumPy folosește implicit varianta pentru populație; setează `ddof=1` pentru estimarea bazată pe eșantion.

```python
population_std = np.std(speed)
sample_std = np.std(speed, ddof=1)
print("Abaterea standard a populației:", population_std)
print("Abaterea standard a eșantionului:", sample_std)
```

### 6. Percentile

O **percentilă** indică o poziție în distribuție: percentila 25 este o valoare sub care se află aproximativ 25% dintre observații, conform metodei folosite pentru calculul percentilei. Percentila 50 este mediana.

Metoda implicită din NumPy interpolează între observații atunci când este necesar. Pentru acest set de date, compară percentilele 25, 50 și 75.

```python
p25, p50, p75 = np.percentile(speed, [25, 50, 75])
print("Percentila 25:", p25)
print("Percentila 50 (mediana):", p50)
print("Percentila 75:", p75)
```

### Compararea statisticilor descriptive

Tabelul de mai jos reunește principalele statistici. Folosește-l pentru a compara măsurile tendinței centrale (media, mediana, moda) și măsurile dispersiei (varianța și abaterea standard). Reține că varianța este exprimată în unități pătrate, în timp ce abaterea standard este exprimată în unitățile originale.

```python
summary = pd.DataFrame({
    "Statistică": ["Media", "Mediana", "Moda", "Varianța populației", "Abaterea standard a populației", "Percentila 25", "Percentila 50", "Percentila 75"],
    "Valoare": [mean_value, median_value, mode_value, population_variance, population_std, p25, p50, p75],
})
display(summary)
```

---

## Partea 2 — Generarea și vizualizarea unei distribuții

O **distribuție uniformă** pe un interval atribuie aceeași densitate de probabilitate valorilor din acel interval. Celula de mai jos generează 250 de numere pseudoaleatoare între 0 și 5.

O sămânță aleatoare fixă face ca secvența generată să poată fi reprodusă în același mediu. Valoarea `42` este o convenție, nu un număr special din punct de vedere matematic.

```python
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)
uniform_data = rng.uniform(low=0.0, high=5.0, size=250)

print("Număr de valori generate:", len(uniform_data))
print("Minim:", uniform_data.min())
print("Maxim:", uniform_data.max())
print("Media:", uniform_data.mean())
```

### 7. Histogramă

O **histogramă** grupează valorile numerice în intervale numite clase (*bins*). Înălțimea fiecărei bare arată câte observații se află în intervalul respectiv. O histogramă este utilă pentru explorarea formei, dispersiei și eventualei asimetrii a unei distribuții, deși aspectul ei depinde de numărul și lățimea claselor.

```python
plt.figure(figsize=(8, 4))
plt.hist(uniform_data, bins=5, edgecolor="black")
plt.title("Histogramă pentru date generate aleatoriu, cu distribuție uniformă")
plt.xlabel("Valoare")
plt.ylabel("Frecvență")
plt.show()
```

**Experiment:** Modifică `bins=5` în `bins=10` și execută din nou celula histogramei. Ce se schimbă în reprezentare?

---

## Partea 3 — Diagrame de dispersie

O **diagramă de dispersie** reprezintă observații pereche pentru două variabile numerice. Fiecare punct reprezintă o pereche de valori. O diagramă de dispersie poate evidenția relații, grupuri, tendințe sau observații neobișnuite, dar o asociere vizuală nu demonstrează, de una singură, o relație de cauzalitate.

Următorul exemplu folosește `speed` și o a doua listă, `y_values`. Aceste valori sunt perechi ilustrative create pentru reprezentare grafică; nu au o semnificație documentată în lumea reală.

```python
y_values = [3, 1, 5, 9, 11, 19, 6, 4, 9, 3]

plt.figure(figsize=(7, 4))
plt.scatter(speed, y_values)
plt.title("Diagramă de dispersie pentru două variabile ilustrative")
plt.xlabel("Valori speed")
plt.ylabel("Valori Y")
plt.show()
```

**Sarcină de interpretare:** Descrie dacă graficul sugerează o relație pozitivă clară, o relație negativă sau lipsa unui tipar liniar evident.

---

## Partea 4 — Încărcarea și explorarea unui set de date CSV

Fișierul `Dataset.csv` conține un set de date ilustrativ, de dimensiuni mici, cu patru coloane:

- `Țară`: variabilă categorială;
- `Vârstă`: variabilă numerică, cu o valoare lipsă;
- `Salariu`: variabilă numerică, cu o valoare lipsă;
- `Achiziționat`: rezultat categorial (`Da` sau `Nu`).

Setul de date este intenționat mic, astfel încât primii pași de explorare să fie ușor de urmărit. Nu este destinat formulării unor concluzii despre populații reale sau comportamentul consumatorilor.

### 8. Deschiderea fișierului

`pandas.read_csv()` citește un fișier CSV într-un `DataFrame`. Calea fișierului este relativă la directorul de lucru curent al notebook-ului. Dacă fișierul nu este găsit, verifică dacă `Dataset.csv` a fost încărcat sau plasat lângă notebook.

```python
dataset = pd.read_csv("Dataset.csv")

display(dataset.head())
```

### 9. Inspectarea structurii și a tipurilor de date

Înainte de a analiza un set de date, verifică numărul de rânduri și coloane, numele coloanelor, tipurile de date și numărul valorilor care nu lipsesc. Acest lucru ajută la identificarea unor probleme precum text neașteptat în coloanele numerice sau valori lipsă.

```python
print("Dimensiunea setului de date (rânduri, coloane):", dataset.shape)
print("Numele coloanelor:", dataset.columns.tolist())
dataset.info()
```

### 10. Statistici descriptive pentru toate coloanele

`DataFrame.describe()` rezumă implicit coloanele numerice. Cu `include="all"`, pandas include și coloanele categoriale: de exemplu, numărul valorilor unice, valoarea cea mai frecventă și frecvența acesteia. Unele statistici nu au sens pentru toate tipurile de date, astfel că pandas afișează `NaN` atunci când o statistică nu se aplică.

```python
display(dataset.describe(include="all"))
```

### 11. Vizualizarea vârstei și a salariului

O diagramă de dispersie poate arăta cum variază împreună două variabile numerice. În acest grafic sunt reprezentate variabilele `Vârstă` și `Salariu`.

```python
plt.scatter(dataset['Salary'], dataset['Age'])
plt.show()
```

---

## Resurse suplimentare

- Documentația NumPy: https://numpy.org/doc/
- Statistici SciPy: https://docs.scipy.org/doc/scipy/reference/stats.html
- Documentația Pandas: https://pandas.pydata.org/docs/
- Documentația Matplotlib: https://matplotlib.org/stable/users/index.html

---

### Fișiere necesare

- `README.md` — explicațiile și codul laboratorului, afișate direct pe GitHub.
- `Dataset.csv` — fișierul folosit în partea de explorare a datelor.

**Notă:** GitHub afișează blocurile de cod, dar nu păstrează starea variabilelor între blocuri și nu execută codul. Pentru o rulare completă, copiază blocurile `python` în celule Colab/Jupyter în ordinea lor.

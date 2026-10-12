# Car Price Prediction

Machine learning projekat za predviđanje tržišne cene polovnih automobila na osnovu njihovih karakteristika.

## Autor

Igor Jovanovic

## Cilj projekta

Cilj projekta je razvoj regresionog modela koji predviđa cenu polovnog automobila izraženu u američkim dolarima.

Ciljna promenljiva je:

```text
priceUSD
```

Model koristi podatke kao što su:

- marka automobila
- model
- godina proizvodnje
- kilometraža
- stanje vozila
- tip goriva
- zapremina motora
- boja
- menjač
- pogon
- segment vozila

## Dataset

Projekat koristi dataset:

```text
cars.csv
```

Originalni dataset sadrži:

```text
56244 reda
12 kolona
```

## Struktura projekta

```text
Task4_CarPricePrediction/
│
├── cars.csv
├── cars_clean.csv
├── cars_features.csv
├── car_price_analysis.ipynb
│
├── data_cleaning.py
├── feature_engineering.py
├── data_preprocessing.py
├── model_training.py
├── model_evaluation.py
├── model_comparison.py
│
├── car_price_model.joblib
├── model_comparison_results.csv
│
├── requirements.txt
└── README.md
```

## EDA analiza

Tokom istraživanja podataka analizirani su:

- broj redova i kolona
- tipovi podataka
- nedostajuće vrednosti
- distribucija ciljne promenljive `priceUSD`
- ekstremne vrednosti
- numeričke i kategorijske karakteristike
- duplirani redovi
- distribucija kategorijskih vrednosti

Utvrđeno je da dataset sadrži 87 potpuno dupliranih redova.

Nedostajuće vrednosti su pronađene uglavnom u kolonama:

```text
volume(cm3)
drive_unit
segment
```

Takođe su pronađene sumnjivo visoke vrednosti kilometraže i zapremine motora.

## Čišćenje podataka

Čišćenje se obavlja u fajlu:

```text
data_cleaning.py
```

Primijenjeni su sledeći koraci:

- uklanjanje potpuno dupliranih redova
- kilometraže veće od 1.000.000 km tretiraju se kao nevalidne
- zapremine motora veće od 8000 cm3 tretiraju se kao nevalidne
- nevalidne vrednosti se postavljaju na `NaN`
- nedostajuće vrednosti se kasnije obrađuju u preprocessing pipeline-u

Nakon uklanjanja duplikata ostaje:

```text
56157 redova
```

Ekstremno visoke cene nisu automatski uklonjene, jer je analizom utvrđeno da mnoge pripadaju realnim luksuznim automobilima kao što su Bentley, Porsche, Mercedes-Benz i drugi.

## Feature engineering

Feature engineering se nalazi u fajlu:

```text
feature_engineering.py
```

Kreirane su četiri dodatne karakteristike.

### car_age

Starost automobila u odnosu na referentnu godinu.

Najnovija godina proizvodnje u datasetu je 2019, pa je kao reproduktivna referentna godina korišćena:

```text
2020
```

### mileage_per_year

Prosečna kilometraža automobila po godini starosti.

### engine_volume_liters

Zapremina motora izražena u litrima umesto u cm3.

### brand_model

Kombinacija marke i modela automobila.

Primer:

```text
volkswagen_golf
```

Dataset sadrži:

```text
1071 jedinstvenu brand_model kombinaciju
```

## Preprocessing

Preprocessing se nalazi u fajlu:

```text
data_preprocessing.py
```

Numeričke karakteristike su:

```text
car_age
mileage(kilometers)
mileage_per_year
engine_volume_liters
```

Kategorijske karakteristike su:

```text
make
model
brand_model
condition
fuel_type
color
transmission
drive_unit
segment
```

Za numeričke podatke koriste se:

- `SimpleImputer(strategy="median")`
- `StandardScaler`

Za kategorijske podatke koriste se:

- `SimpleImputer(strategy="most_frequent")`
- `OneHotEncoder(handle_unknown="ignore")`

Svi koraci su povezani pomoću `ColumnTransformer` i `Pipeline`.

## Train/Test podela

Za sve modele korišćen je isti train/test split:

```text
80% trening
20% test
random_state = 42
```

Broj primera:

```text
Training set: 44925
Test set:     11232
```

Isti test skup korišćen je za sve modele kako bi poređenje bilo fer.

## Testirani modeli

Testirana su četiri glavna regresiona algoritma:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

U notebook-u je dodatno testirana i varijanta Random Forest modela sa log transformacijom ciljne promenljive.

## Rezultati modela

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Random Forest | **1108.31** | **8100920.74** | **2846.21** | **0.8793** |
| Decision Tree | 1325.65 | 11340819.26 | 3367.61 | 0.8310 |
| Gradient Boosting | 1538.81 | 9041896.28 | 3006.97 | 0.8653 |
| Linear Regression | 2086.22 | 18713903.34 | 4325.96 | 0.7212 |

## Finalni model

Kao finalni model izabran je:

```text
Random Forest Regressor
```

Korišćeni parametri:

```python
RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    max_depth=20,
    min_samples_leaf=2
)
```

Finalni rezultat na test skupu:

```text
MAE:  1108.31 USD
RMSE: 2846.21 USD
R²:   0.8793
```

MAE od približno 1108 USD znači da model u proseku promašuje stvarnu cenu automobila za oko 1108 dolara.

R² od 0.8793 znači da model objašnjava približno 87.9% varijacije cena u test skupu.

## Random Forest sa log transformacijom

Tokom eksperimentisanja testiran je i Random Forest sa `log1p` transformacijom ciljne promenljive.

Rezultat:

```text
MAE:  1106.70 USD
RMSE: 2944.98 USD
R²:   0.8708
```

Iako je MAE bio neznatno bolji, razlika je bila samo oko 1.6 USD, dok su RMSE i R² bili lošiji.

Zbog boljeg ukupnog balansa metrika standardni Random Forest izabran je kao finalni model.

## Analiza grešaka

Model dobro predviđa većinu automobila iz uobičajenog cenovnog opsega.

Najveće greške primećene su kod veoma skupih i retkih luksuznih automobila.

Primeri takvih vozila uključuju:

- Bentley
- Porsche 911
- Mercedes-AMG modele
- BMW serije 7

Razlog je što se takva vozila pojavljuju znatno ređe u datasetu, pa model ima manje sličnih primera iz kojih može da uči.

## Instalacija

Potrebne biblioteke nalaze se u:

```text
requirements.txt
```

Instalacija:

```bash
pip install -r requirements.txt
```

## Pokretanje projekta

### 1. Čišćenje podataka

```bash
python data_cleaning.py
```

Kreira:

```text
cars_clean.csv
```

### 2. Feature engineering

```bash
python feature_engineering.py
```

Kreira:

```text
cars_features.csv
```

### 3. Provera preprocessing-a

```bash
python data_preprocessing.py
```

### 4. Treniranje finalnog modela

```bash
python model_training.py
```

Kreira:

```text
car_price_model.joblib
```

### 5. Evaluacija finalnog modela

```bash
python model_evaluation.py
```

Prikazuje:

- MAE
- MSE
- RMSE
- R²
- primere stvarnih i predviđenih cena

### 6. Poređenje modela

```bash
python model_comparison.py
```

Kreira:

```text
model_comparison_results.csv
```

i prikazuje rezultate svih testiranih modela.

## Jupyter notebook

Kompletna analiza i razvoj rešenja nalaze se u:

```text
car_price_analysis.ipynb
```

Notebook sadrži:

- EDA analizu
- analizu missing vrednosti
- analizu outlier-a
- analizu ciljne promenljive
- čišćenje podataka
- feature engineering
- preprocessing
- treniranje više modela
- poređenje modela
- regresione metrike
- primere predikcija
- grafik stvarne i predviđene cene
- analizu najvećih grešaka
- izbor finalnog modela

## Sačuvani model

Finalni trenirani pipeline sačuvan je kao:

```text
car_price_model.joblib
```

Fajl sadrži preprocessing i Random Forest model, tako da se isti preprocessing automatski primenjuje prilikom korišćenja modela.

## Zaključak

Random Forest Regressor pokazao je najbolji ukupni rezultat među testiranim modelima.

Finalni model ostvaruje prosečnu apsolutnu grešku od približno 1108 USD i objašnjava oko 87.9% varijacije cena automobila.

Rezultati pokazuju da kombinovanje čišćenja podataka, dodatnih karakteristika i nelinearnog ensemble modela može pouzdano procenjivati cenu većine polovnih automobila u datasetu.
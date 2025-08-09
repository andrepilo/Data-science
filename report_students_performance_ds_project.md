# Progetto di Data Science — Students Performance

**Titolo:** Predizione del punteggio di matematica (regressione)

**Autore:** [Tu Nome]

**Dataset:** *Students Performance in Exams* (Kaggle) — [https://www.kaggle.com/datasets/spscientist/students-performance-in-exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

---

## 1. Abstract

Breve progetto di Data Science volto a prevedere il punteggio di matematica (`math score`) degli studenti a partire da variabili socio‑demografiche e dagli altri punteggi di esame (`reading score`, `writing score`). Il lavoro comprende: esplorazione del dataset, preprocessing, addestramento di due modelli di regressione (Linear Regression e Random Forest Regressor), valutazione quantitativa e analisi critica dei risultati.

---

## 2. Descrizione del dataset

**Fonte:** Kaggle — *Students Performance in Exams*.

**Caratteristiche principali:**

- Numero di righe: \~1000 (variabile a seconda della versione del dataset)
- Colonne principali:
  - `gender` (categorical)
  - `race/ethnicity` (categorical)
  - `parental level of education` (categorical)
  - `lunch` (categorical)
  - `test preparation course` (categorical)
  - `reading score` (numeric)
  - `writing score` (numeric)
  - `math score` (numeric) — target

**Note:** il dataset è di dimensione contenuta, ben adatto per esercitazioni e modelli interpretabili. I punteggi sono in genere su scala 0-100.

---

## 3. Obiettivo

Prevedere `math score` (regressione continua) usando le informazioni disponibili al momento dell’esame.

Motivazione: capire quali fattori (es. preparazione al test, livello d’istruzione dei genitori, performance in reading/writing) contribuiscono maggiormente al risultato in matematica.

---

## 4. Metodologia

**Ambiente di lavoro:** Python, pandas, scikit-learn, matplotlib.

**Passaggi:**

1. Caricamento del CSV locale (`StudentsPerformance.csv`).
2. Analisi esplorativa (ispezione, statistiche descrittive, distribuzioni, correlazioni).
3. Preprocessing:
   - Rinominazione colonne per coerenza
   - Gestione dei valori mancanti (se presenti)
   - Codifica delle variabili categoriche (One‑Hot Encoding, `drop_first=True` per evitare multicollinearità)
   - Separazione X / y
4. Divisione train/test (80% train, 20% test, `random_state=42`).
5. Modelli:
   - **Linear Regression** (baseline, facilmente interpretabile)
   - **Random Forest Regressor** (modello non lineare, robusto)
6. Valutazione con metriche: MAE, RMSE, R². Visualizzazioni: distribuzioni, importanza delle feature, scatter y\_true vs y\_pred.

---

## 5. Codice utilizzato

Il notebook allegato (`students_performance_analysis.ipynb`) contiene il codice completo. I passi salienti sono questi (estratto):

```python
# Caricamento
df = pd.read_csv("StudentsPerformance.csv")
# Preprocessing
df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
df_encoded = pd.get_dummies(df, drop_first=True)
X = df_encoded.drop('math_score', axis=1)
y = df_encoded['math_score']
# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Modelli
lin_reg = LinearRegression().fit(X_train, y_train)
rf_reg = RandomForestRegressor(random_state=42).fit(X_train, y_train)
# Predizioni e metriche
```

---

## 6. Risultati (da eseguire nel notebook)

> **ATTENZIONE:** qui vanno inseriti i risultati numerici ottenuti eseguendo il notebook sul tuo computer. Di seguito trovi la tabella modello/metriche che puoi copiare e compilare.

| Modello                 | MAE        | RMSE       | R²         |
| ----------------------- | ---------- | ---------- | ---------- |
| Linear Regression       | *inserire* | *inserire* | *inserire* |
| Random Forest Regressor | *inserire* | *inserire* | *inserire* |

**Grafici da includere nel report:**

- Istogrammi di `math_score`, `reading_score`, `writing_score`.
- Matrice di correlazione (heatmap) per le variabili numeriche.
- Feature importance da Random Forest (bar chart).
- Scatter plot: valore reale vs predetto per i due modelli sulla stessa figura.

---

## 7. Analisi critica dei risultati

Esempi di punti da analizzare e inserire nel report, dopo aver ottenuto i risultati effettivi:

- **Quali feature risultano più importanti?**

  - È molto probabile che `reading score` e `writing score` siano tra le feature più influenti. Questo va riportato e discusso: potrebbe indicare che abilità linguistiche e logico‑pratiche correlate supportano la performance in matematica.

- **Confronto modelli:**

  - Se Random Forest ottiene R² più alto e RMSE più basso rispetto alla regressione lineare, discutere il trade‑off interpretabilità vs performance.

- **Overfitting / underfitting:**

  - Confrontare errori su train vs test. Se il modello complesso ha errore molto più basso su train che su test, sospettare overfitting.

- **Limitazioni del dataset:**

  - Dimensione limitata; mancanza di variabili potenzialmente rilevanti (es. ore di studio settimanali, qualità della scuola); possibili bias (distribuzione dell’etnia, genere, background familiare).

- **Bias informativi:**

  - Inclusion of reading/writing scores as predictors may trivially improve performance ma ridurre il valore predittivo in scenari realistici dove questi dati non sono immediatamente disponibili.

- **Ulteriori passi suggeriti:**

  - Prova di modelli regolarizzati (Ridge/Lasso), ricerca iperparametri (GridSearchCV), cross‑validation (K‑fold), e test su dati esterni.

---

## 8. Conclusioni

Riassumere i risultati principali (includere i numeri calcolati). Discutere l’utilità del modello: ad es. se si riesce a prevedere il punteggio con buona accuratezza, strumenti simili potrebbero aiutare a identificare studenti a rischio e intervenire tempestivamente — sempre tenendo conto di etica e privacy.

---

## 9. Allegati e riferimenti

- Notebook: `students_performance_analysis.ipynb` (incluso)
- Dataset su Kaggle: [https://www.kaggle.com/datasets/spscientist/students-performance-in-exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- Librerie principali: pandas, numpy, scikit-learn, matplotlib

---

## 10. Script suggerito per il video (max 8 minuti)

- **0:00–0:45** — Introduzione: obiettivo e dataset.
- **0:45–2:00** — Panoramica dataset (colonne, dimensioni, qualche statistica chiave).
- **2:00–3:30** — Preprocessing e motivazioni (encoding, split).
- **3:30–5:00** — Modelli implementati e motivazione della scelta.
- **5:00–6:30** — Risultati principali (mostra le metriche e i grafici principali).
- **6:30–7:30** — Discussione critica e limiti.
- **7:30–8:00** — Conclusione e possibili sviluppi.

---

### Note finali — istruzioni per l'uso

1. Scarica `StudentsPerformance.csv` da Kaggle e mettilo nella stessa cartella del notebook.
2. Esegui il notebook; copia i risultati (metriche e grafici) nella sezione **6. Risultati** del report.
3. Esporta il report in PDF (max 8 pagine).
4. Registra il video seguendo lo script suggerito.

Buon lavoro! Se vuoi, preparo anche il file `.ipynb` pronto nello spazio canvas o ti fornisco il file .ipynb da scaricare.


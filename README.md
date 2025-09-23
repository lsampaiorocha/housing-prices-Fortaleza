# Housing Prices in Fortaleza/Brazil – Data Science & ML Project

This repository explores **real estate transactions in Fortaleza, Brazil** (ITBI dataset).  
The goal is to perform **data cleaning, exploratory analysis, statistical testing, and predictive modeling** of property values using modern **data science and machine learning techniques**.  

---

## About Fortaleza

**Fortaleza** is the capital of the state of Ceará, located in Northeast Brazil, and one of the country’s largest urban centers with more than **2.6 million inhabitants**.  
As a coastal city with strong economic, touristic, and cultural activity, Fortaleza has a highly dynamic **real estate market**.  

The dataset used here reflects **property transactions subject to ITBI** (Imposto sobre Transmissão de Bens Imóveis), a municipal tax applied when property ownership is transferred.  
Analyzing this dataset provides insights into **housing prices, urban development, and investment opportunities** in Fortaleza.

---

## Repository Structure

project-name/
│── data/ # Raw or processed datasets (not included in repo)
│── models/ # Trained ML models (to be added)
│── notebooks/
│ └── analysis.ipynb # Main notebook with step-by-step exploration
│── results/ # Generated plots, tables, and reports
│── src/ # Source code (data processing, modeling, utilities)
│── README.md # Project overview (this file)


At the moment, the main work is concentrated in the notebook:

- **`notebooks/analysis.ipynb`** → Contains the end-to-end exploratory data analysis and first experiments.

---

## Objectives

- Explore property transaction data (ITBI Fortaleza).
- Clean and preprocess the dataset (dates, values, missing data, outliers).
- Perform exploratory data analysis (EDA):
  - Property values per neighborhood.
  - Temporal trends in transactions.
  - Usage types and construction patterns.
- Apply statistical testing (normality, variance homogeneity, ANOVA/Welch).
- Build regression and classification models to predict housing prices.

---

## Dataset

The dataset is publicly available through the **Prefeitura de Fortaleza Open Data portal**:  
[ITBI Real Estate Transactions Dataset](https://dados.fortaleza.ce.gov.br/dataset/dados_abertos_itbi_transacoes_imobiliarias/resource/46324d49-9809-4d32-8e5c-b4b570f067ce)

---

## Next Steps

- Move reusable code (preprocessing, modeling) into `src/`.
- Save trained models under `models/`.
- Add results (figures, metrics) in `results/`.
- Extend the notebook into a full ML pipeline.

---

## Author

**Leonardo Sampaio Rocha**  
PhD in Computer Science | Data Scientist & ML Engineer  

---

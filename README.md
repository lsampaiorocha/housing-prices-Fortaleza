# Housing Prices in Fortaleza, Brazil — Data Science, Machine Learning, and Model Serving

This repository explores **real estate transactions in Fortaleza, Brazil**, using the **ITBI dataset** to analyze, model, and deploy predictive systems for estimating housing prices.  

The project covers the complete lifecycle of a machine learning solution:  
from **data cleaning** and **exploratory analysis (EDA)**, through **training and evaluating models**, to **exporting models to ONNX format** and **serving them with NVIDIA Triton Inference Server** inside a Dockerized environment.

---

## Author
**Leonardo Sampaio Rocha**  
PhD in Computer Science | AI/ML Engineer  

---

## Project Overview

Fortaleza, the capital of Ceará (Brazil), is a large coastal city with a dynamic real estate market.  
The dataset used in this project — the **ITBI Real Estate Transactions Dataset** — includes property transactions subject to the municipal ITBI tax, offering rich information on market behavior, property characteristics, and transaction values.  

This project investigates how machine learning models can learn patterns from this dataset to estimate property prices and automate valuation processes.

---

## Repository Structure

```
housing-prices-Fortaleza/
│
├── data/
│   ├── raw/                     # Original dataset (ITBI public data)
│   │   └── dados_abertos_itbi_transacoes_imobiliarias.csv
│   └── processed/               # Cleaned and transformed data
│       └── processed_data.csv
│
├── notebooks/
│   └── analysis.ipynb           # EDA and model training steps
│
├── models/
│   ├── preprocessor.pkl         # Trained preprocessing pipeline
│   └── decision_tree/
│       ├── config.pbtxt         # Triton model configuration
│       └── 1/
│           └── model.onnx       # Decision Tree model in ONNX format
│
├── results/                     # Generated plots and figures
│   ├── Decision_Tree.png
│   ├── Linear Regression.png
│   ├── Pairwise correlations.png
│   ├── Dunns post hoc test.png
│   ├── Numeric features versus target.png
│   └── XGBoost.png
│
├── test/
│   └── client_test.py           # Python script for inference testing
│
├── Dockerfile                   # Image build for Triton model serving
├── docker-compose.yml           # Service orchestration
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## Pipeline Description

### 1. Data Cleaning and Transformation
The raw ITBI dataset was cleaned and standardized through:
- Conversion of date fields and normalization of numeric attributes.  
- Removal of inconsistent or missing records.  
- Detection and treatment of outliers in property values.  
- Feature encoding for categorical variables such as *bairro* (neighborhood).

The cleaned dataset is stored under `data/processed/processed_data.csv`.

---

### 2. Exploratory Data Analysis (EDA)
Performed using **Pandas**, **Matplotlib**, and **Seaborn** to identify:
- Distribution of transactions per neighborhood and construction type.  
- Correlation between variables and sale values.  

---

### 3. Model Training and Evaluation
Multiple supervised learning models were trained and compared, including:
- **Linear Regression**
- **Decision Tree Regressor**
- **XGBoost Regressor**

Each model was evaluated using:
- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**
- **R² Score**

The Decision Tree Regressor was selected for deployment based on interpretability and performance balance.

---

### 4. Exporting the Model to ONNX
The trained model was converted to the **ONNX format** for production deployment using the ONNX Runtime.  
Only the trained regressor was exported; the preprocessing pipeline (`preprocessor.pkl`) is loaded separately during inference.

ONNX file location:
```
models/decision_tree/1/model.onnx
```

---

### 5. Deployment with NVIDIA Triton Inference Server
The project includes a full **Docker-based inference environment**.  
Triton automatically loads the exported model and exposes HTTP and gRPC endpoints for inference.

#### Key Files
- **`Dockerfile`**: Builds a custom image based on Triton and installs required Python packages.  
- **`docker-compose.yml`**: Defines the Triton service with exposed ports.  
- **`config.pbtxt`**: Model configuration specifying input/output tensors and dimensions.

---

## Technologies Used

- **Python 3.11**
- **Pandas, NumPy, Scikit-learn, XGBoost**
- **Matplotlib, Seaborn** (EDA and visualization)
- **ONNX & ONNX Runtime**
- **NVIDIA Triton Inference Server**
- **Docker / Docker Compose**

---

## Build and Run Instructions

### 1. Build the Docker image
```bash
docker compose build
```

### 2. Start the Triton Inference Server
```bash
docker compose up
```

This will:
- Launch the Triton container.
- Automatically load the model from `/models/decision_tree`.
- Expose the following ports:
  - `8000`: HTTP endpoint  
  - `8001`: gRPC endpoint  
  - `8002`: Metrics endpoint  

### 3. Verify the model is loaded
```bash
curl -X POST localhost:8000/v2/repository/index
```

Expected output:
```json
[{"name":"decision_tree","version":"1","state":"READY"}]
```

### 4. Run inference (example)
```bash
curl -X POST localhost:8000/v2/models/decision_tree/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "input",
        "shape": [1, 3],
        "datatype": "FP32",
        "data": [[120.0, 15.0, 1.0]]
      }
    ],
    "outputs": [{"name": "variable"}]
  }'
```

Or use the provided test script:
```bash
python test/client_test.py
```

---

This setup demonstrates an **end-to-end data science workflow**, from **data preparation and modeling** to **ONNX-based deployment** with **Triton Inference Server** for production-grade model serving.
# ☀️ Solar Power Generation Forecasting

A machine learning regression project to predict solar power output using environmental and operational data, deployed as an interactive Streamlit web application.

---

## 🔍 Project Overview

This project builds and evaluates multiple regression models to forecast solar power generation based on environmental conditions such as temperature, irradiance, and humidity. The trained XGBoost model is deployed via a Streamlit web app that allows users to input environmental parameters and get instant power output predictions.

---

## 🚀 Features

- 📊 Exploratory Data Analysis (EDA) on solar generation data
- 🤖 Multiple regression models built and compared
- 🏆 XGBoost selected as the best performing model
- 📉 Model evaluation using MAE, RMSE, and R² metrics
- 🌐 Interactive Streamlit web app for real-time predictions
- 🔧 Feature selection to identify most impactful environmental variables

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Machine Learning | Scikit-learn, XGBoost |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |

---

## 📁 Project Structure

```
Solar_Power_Generation_Project/
│
├── P640 Regression.ipynb       # Full ML pipeline — EDA, modeling, evaluation
├── P640_app.py                 # Streamlit web application
├── xgboost_solar_model.pkl     # Trained XGBoost model (saved)
└── README.md
```

---

## 📦 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Tilna/Solar_Power_Generation_Project.git
cd Solar_Power_Generation_Project
```

### 2. Install dependencies

```bash
pip install streamlit scikit-learn xgboost pandas numpy matplotlib seaborn
```

### 3. Run the app

```bash
streamlit run P640_app.py
```

---

## 🧠 ML Pipeline

1. **Data Loading & EDA** — explored distributions, correlations, and outliers
2. **Feature Selection** — identified key environmental variables affecting power output
3. **Preprocessing** — handled missing values, scaled features
4. **Model Training** — trained and compared multiple regression models
5. **Evaluation** — compared models using MAE, RMSE, and R² metrics
6. **Deployment** — best model (XGBoost) saved and deployed via Streamlit

---

## 📊 Model Evaluation Metrics

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error — average prediction error |
| RMSE | Root Mean Squared Error — penalizes large errors |
| R² | R-squared — how well the model explains variance |

---

## 📌 How the App Works

1. User inputs environmental parameters (temperature, irradiance, etc.)
2. The app feeds inputs into the trained XGBoost model
3. Predicted solar power output is displayed instantly

---

## 🙋 Author

**Tilna**
- GitHub: [@Tilna](https://github.com/Tilna)

---

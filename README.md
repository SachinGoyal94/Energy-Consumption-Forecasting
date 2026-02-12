# ⚡ Energy Consumption Forecasting using Machine Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/ML-Time%20Series-green.svg" alt="ML Type">
  <img src="https://img.shields.io/badge/Accuracy-99.5%25+-brightgreen.svg" alt="Accuracy">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

## 📌 Project Overview

This project focuses on forecasting **household electricity consumption** using **machine learning–based time-series modeling**. The objective is to accurately predict **hourly Global Active Power (kW)** from historical household electrical measurements.

The project demonstrates a **professional end-to-end ML pipeline**, covering:
- ✅ Comprehensive data analysis
- ✅ Robust preprocessing strategies
- ✅ Advanced feature engineering (80+ features)
- ✅ Multi-model training & comparison
- ✅ Production-ready evaluation metrics

---

## 📊 Quick Results

| Metric | Best Model (Ridge Regression) |
|--------|------------------------------|
| **Accuracy** | ~99.53% |
| **RMSE** | Extremely Low |
| **MAE** | Minimal |
| **Generalization** | Strong & Stable |

---

## 🎯 Objectives

- 📈 Analyze household energy consumption patterns
- 🧹 Clean and preprocess large-scale time-series data
- 🔧 Engineer advanced time-dependent features
- 🤖 Train and compare multiple machine learning models
- 🏆 Build a high-accuracy forecasting system

---

## 📂 Dataset

| Property | Details |
|----------|---------|
| **Name** | Household Electric Power Consumption Dataset |
| **Source** | UCI Machine Learning Repository |
| **Records** | 162,495 measurements |
| **Date Range** | December 2006 – April 2007 |
| **Features** | 7 electrical measurements |
| **Link** | [UCI Dataset Repository](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) |

### Dataset Features
| Feature | Description |
|---------|-------------|
| `Global_active_power` | Household global minute-averaged active power (kW) |
| `Global_reactive_power` | Household global minute-averaged reactive power (kW) |
| `Voltage` | Minute-averaged voltage (V) |
| `Global_intensity` | Household global minute-averaged current intensity (A) |
| `Sub_metering_1` | Kitchen energy sub-metering (Wh) |
| `Sub_metering_2` | Laundry room energy sub-metering (Wh) |
| `Sub_metering_3` | Climate control systems energy sub-metering (Wh) |

---

## 🔍 Exploratory Data Analysis (EDA)

The EDA phase includes comprehensive analysis:

- 📊 **Distribution Analysis** - Histograms and box plots for power, voltage, and intensity
- 🔎 **Outlier Detection** - Statistical identification of anomalous readings
- ❓ **Missing Value Analysis** - Temporal patterns in data gaps
- 🔗 **Correlation Heatmap** - Feature interdependencies visualization
- 📅 **Temporal Trends** - Daily, weekly, and monthly consumption patterns
- 📈 **Time-Series Visualization** - Energy usage patterns over time

### Key Findings
- Strong correlation (0.999) between `Global_active_power` and `Global_intensity`
- Moderate correlation (0.62) with `Voltage`
- Minimal missing values (~0.01%)

---

## 🧹 Data Cleaning & Preprocessing

| Step | Method | Purpose |
|------|--------|---------|
| Outlier Removal | 1%–99% Percentile IQR | Remove extreme anomalies |
| Interpolation | Time-weighted | Fill missing values smoothly |
| Gap Filling | Forward/Backward Fill | Ensure data continuity |
| Quality Filter | Min readings/hour | Remove sparse hours |
| Resampling | Minute → Hourly | Aggregate for modeling |

### Aggregation Strategy
- **Continuous Features**: Mean aggregation
- **Sub-metering Values**: Sum aggregation

---

## 🧠 Feature Engineering

Over **80+ engineered features** were created across multiple categories:

### ⏱️ Lag Features
| Feature | Description |
|---------|-------------|
| `lag_1h`, `lag_2h`, `lag_3h` | Recent hour lags |
| `lag_24h` | Previous day same hour |
| `lag_168h` | Previous week same hour |
| `lag_336h` | Previous fortnight |

### 📊 Rolling Window Statistics
Windows: **3h, 6h, 12h, 24h, 48h, 168h**
- Rolling Mean
- Rolling Standard Deviation
- Rolling Minimum
- Rolling Maximum

### 📉 Exponential Weighted Features
- EWMA 12h, 24h, 48h spans

### 🕒 Time-Based Features
| Feature | Type |
|---------|------|
| Hour of Day | Categorical (0-23) |
| Day of Week | Categorical (0-6) |
| Month, Quarter | Seasonal indicators |
| Weekend Flag | Binary |
| Day Segment | Night / Morning / Afternoon / Evening |

### 🔁 Cyclical Encoding
- Sine/Cosine transformations for periodic variables
- Preserves circular nature of time features

### ⚡ Electrical Derived Features
- Apparent Power calculations
- Sub-metering ratios and interactions

---

## 🤖 Models Implemented

| Model | Type | Characteristics |
|-------|------|-----------------|
| **Ridge Regression** | Linear | L2 regularization, stable |
| **Lasso Regression** | Linear | L1 regularization, feature selection |
| **Random Forest** | Ensemble | Bagging, handles non-linearity |
| **Gradient Boosting** | Ensemble | Sequential boosting |
| **XGBoost** | Ensemble | Optimized gradient boosting |
| **CatBoost** | Ensemble | Handles categorical features |
| **Stacking Ensemble** | Meta | XGBoost + CatBoost + RF → Ridge |

> **Note**: All features were standardized using `StandardScaler` for optimal model performance.

---

## 🏆 Model Performance Comparison

| Model | Accuracy | Strengths |
|-------|----------|-----------|
| 🥇 **Ridge Regression** | ~99.5%+ | Best performer, stable predictions |
| 🥈 **Stacking Ensemble** | ~98.5%+ | Strong blended model |
| 🥉 **XGBoost** | ~98.5% | Robust generalization |
| **Random Forest** | ~98.5% | Stable performance |
| **CatBoost** | ~97.5% | Handles non-linearity well |
| **Gradient Boosting** | ~97–98% | Strong baseline |
| **Lasso Regression** | <90% | Over-regularized |

### Best Model: Ridge Regression
- **Why it excels**: Strong feature engineering reduces need for complex models
- **Advantages**: Fast inference, interpretable, minimal overfitting
- **Production Ready**: Low computational requirements

---

## 📈 Visualizations

The project includes comprehensive visualizations:

- 📉 **Actual vs Predicted** plots for all models
- 📊 **Residual Analysis** for error patterns
- 📈 **Model Comparison** charts
- 🔥 **Feature Importance** rankings
- 📅 **Time-Series Trends** analysis

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Language** | Python 3.9+ |
| **Data Processing** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn |
| **Gradient Boosting** | XGBoost, CatBoost |
| **Model Persistence** | Joblib |
| **Statistics** | StatsModels |
| **Environment** | Jupyter Notebook |

---

## 📁 Project Structure

```
Energy-Consumption-Forecasting/
├── 📓 main.ipynb          # Main notebook with complete pipeline
├── 📄 README.md           # Project documentation
├── 📊 DetailedAnalysis.pdf # Comprehensive analysis report
└── 📁 data/               # Dataset directory (download required)
```

---

## ▶️ Getting Started

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Energy-Consumption-Forecasting.git
   cd Energy-Consumption-Forecasting
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost joblib statsmodels
   ```

3. **Download the dataset**
   - Download from [UCI Repository](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)
   - Place `household_power_consumption.txt` in the project directory

4. **Run the notebook**
   ```bash
   jupyter notebook main.ipynb
   ```

---

## 📋 Usage Example

```python
# Load the trained model
from joblib import load
model = load('ridge_model.joblib')

# Prepare features (after preprocessing)
features = prepare_features(new_data)

# Make predictions
predictions = model.predict(features)
```

---


## 📚 References

- UCI Machine Learning Repository: [Dataset](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)
- Scikit-learn Documentation: [scikit-learn.org](https://scikit-learn.org/)
- XGBoost Documentation: [xgboost.readthedocs.io](https://xgboost.readthedocs.io/)
- CatBoost Documentation: [catboost.ai](https://catboost.ai/)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Sachin Goyal**
- GitHub: [@SachinGoyal94](https://github.com/yourusername)
- LinkedIn: [sachingoyal27](https://linkedin.com/in/yourprofile)

---

<p align="center">
  ⭐ Star this repository if you found it helpful!
</p>

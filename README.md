# Machine Learning Projects

Two machine learning projects from my Diploma in Applied AI and Analytics at Singapore Polytechnic, covering classification, regression, clustering, anomaly detection and time series forecasting in Python.

| Project | Task | Notebook |
|---|---|---|
| Machine Fault Detection | Classification | [CA1 Part A](CA1/p2510893_CA1_PartA_Cayden.ipynb) |
| Housing Price Prediction | Regression | [CA1 Part B](CA1/p2510893_CA1_PartB_Cayden.ipynb) |
| Customer Segmentation | Clustering and anomaly detection | [CA2 Part A](CA2/p2510893_CA2_PartA_Cayden.ipynb) |
| Energy Consumption Forecasting | Time series forecasting | [CA2 Part B](CA2/p2510893_CA2_PartB_Cayden.ipynb) |

---

## Machine Fault Detection

**Goal:** Predict whether a factory machine will fail, using sensor readings.

- **Data:** 20,000 machine records (temperature, rotation speed, torque, tool wear and more), with only about 3.4% faulty, so the classes are heavily imbalanced.
- **Approach:** Compared KNN and Random Forest against a dummy baseline, judging models on recall and F1 for the faulty class rather than overall accuracy, since accuracy is misleading when most machines are healthy. Tuned with GridSearchCV.
- **Result:** Random Forest caught **81% of faults (faulty-class F1: 0.89)**, compared with 46% for KNN.

## Housing Price Prediction

**Goal:** Predict house prices from property features (545 houses: area, bedrooms, toilets, stories, renovation status).

- **Approach:** Built and compared regression models, with feature encoding and hyperparameter tuning.

## Customer Segmentation

**Goal:** Group 200 customers by age, income and spending to support targeted marketing.

- **Approach:** Used PCA for dimensionality reduction, Isolation Forest to detect outliers, and compared three clustering models: KMeans, Agglomerative and DBSCAN.
- **Result:** Identified a low-income segment spending **3.42× its income**, a group worth flagging for a business.

## Energy Consumption Forecasting

**Goal:** Forecast monthly gas, electricity and water consumption.

- **Data:** 33 years of monthly data (1990–2023).
- **Approach:** Compared ARIMA, SARIMA, Auto-ARIMA and Holt-Winters models.
- **Result:** The tuned models overfit, so I kept the simpler baseline models, which generalised better to unseen data.

---

## Repository contents

```
CA1/   Fault detection and housing notebooks, datasets and slides
CA2/   Segmentation and forecasting notebooks and datasets
```

## Tools

Python, pandas, NumPy, scikit-learn, statsmodels, pmdarima, Matplotlib, Seaborn

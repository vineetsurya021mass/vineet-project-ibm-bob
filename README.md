# Sales Prediction Project

End-to-end sales prediction system powered by machine learning.
Covers sales amount prediction, profit prediction, and revenue forecasting
on 18 000+ real transactions (2022–2025).

---
##Dataset## https://www.kaggle.com/datasets/danielsowah123/retail-sales-dataset?resource=download 
## Quick Start

### 1. Install dependencies
```bash
cd sales_prediction
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```
Opens at <http://localhost:8501>

> On first launch, click **"🚀 Train Models Now"** in the sidebar.
> Training runs once and saves models to `models/` for all future sessions.

Alternatively, pre-train from the command line:
```bash
python app.py train
```

---

## Project Structure

```
sales_prediction/
├── app.py                    # Everything: preprocessing + ML + Streamlit UI
├── data/
│   └── Sales_transactions_2022_2025.csv
├── models/                   # Auto-created after first training
│   ├── sales_model.pkl
│   ├── profit_model.pkl
│   ├── feature_names.json
│   └── metrics.json
├── requirements.txt
└── README.md
```

---

## Features

| Page | Description |
|------|-------------|
| **Overview Dashboard** | KPIs, monthly revenue, channel & category breakdown, model metrics |
| **Predict Sales** | Interactive form → live ML prediction of sales amount, profit & margin |
| **Product & Region** | Top products, regional revenue, monthly order volume |
| **Forecasting** | Linear trend forecast with configurable horizon (3–24 months) |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Predict sales & profit |
| GET | `/metrics` | Model evaluation metrics |
| GET | `/analytics/kpis` | Aggregate KPIs |
| GET | `/analytics/monthly-revenue` | Monthly revenue series |
| GET | `/analytics/top-products` | Top N products by revenue |
| GET | `/analytics/sales-by-region` | Revenue by region |
| GET | `/analytics/sales-by-category` | Revenue by category |
| GET | `/analytics/channel-split` | Revenue by sales channel |

---

## Models

| Target | Algorithm | Notes |
|--------|-----------|-------|
| `Sales_Amount` | Gradient Boosting Regressor | 300 trees, LR=0.08, depth=5 |
| `Profit` | Random Forest Regressor | 200 trees, depth=10 |
| Forecast | Linear Regression | On monthly aggregates |

Features used: `Quantity`, `Unit_Price`, `Discount_Percentage`,
`Customer_Age`, `Delivery_Days`, temporal features, and
one-hot encoded categorical variables (channel, category, segment, etc.)

"""
Sales Prediction — single app.py
Everything in one file: preprocessing, model training, and Streamlit UI.

Run:
    streamlit run app.py
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════

_HERE = Path(__file__).parent
DATA_PATH = _HERE / "data" / "Sales_transactions_2022_2025.csv"
MODELS_DIR = _HERE / "models"

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Preprocessing
# ══════════════════════════════════════════════════════════════════════════════

CATEGORICAL_COLS = [
    "Sales_Channel", "Product_Category", "Product_Subcategory",
    "Customer_Segment", "Customer_Gender", "Payment_Method",
    "Shipping_Method", "Country", "Region",
]

TARGET_SALES = "Sales_Amount"
TARGET_PROFIT = "Profit"


def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, low_memory=False)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
    df["Order_Time"] = pd.to_datetime(df["Order_Time"], format="%H:%M:%S", errors="coerce")
    df["Order_Month"] = df["Order_Date"].dt.month
    df["Order_DayOfWeek"] = df["Order_Date"].dt.dayofweek
    df["Order_Hour"] = df["Order_Time"].dt.hour

    num_cols = [
        "Customer_Age", "Discount_Percentage", "Delivery_Days",
        "Unit_Price", "Quantity", "Sales_Amount", "Cost_Amount",
        "Profit", "Inventory_Level", "Customer_Rating",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    if "Payment_Method" in df.columns:
        df["Payment_Method"] = df["Payment_Method"].str.strip().str.title()

    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df.copy(), columns=CATEGORICAL_COLS, drop_first=True)


def get_feature_matrix(df: pd.DataFrame):
    df = encode(df)
    base_numeric = [c for c in [
        "Quantity", "Unit_Price", "Discount_Percentage",
        "Customer_Age", "Delivery_Days",
        "Order_Month", "Order_DayOfWeek", "Order_Hour",
    ] if c in df.columns]
    encoded_cats = [
        c for c in df.columns
        if any(c.startswith(cat + "_") for cat in CATEGORICAL_COLS)
    ]
    all_features = base_numeric + encoded_cats
    X = df[all_features].astype(float)
    y_sales = df[TARGET_SALES].astype(float)
    y_profit = df[TARGET_PROFIT].astype(float)
    return X, y_sales, y_profit, all_features


def monthly_revenue(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["YearMonth"] = df["Order_Date"].dt.to_period("M")
    monthly = (
        df.groupby("YearMonth")
        .agg(
            Revenue=("Sales_Amount", "sum"),
            Orders=("Transaction_ID", "count"),
            AvgOrderValue=("Sales_Amount", "mean"),
        )
        .reset_index()
    )
    monthly["YearMonth"] = monthly["YearMonth"].astype(str)
    return monthly


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Model Training
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    return {
        "MAE": float(mean_absolute_error(y_test, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "R2": float(r2_score(y_test, preds)),
    }


def _build_sales_pipeline(X_train, y_train):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.08,
            max_depth=5, subsample=0.8, random_state=42,
        )),
    ])
    model.fit(X_train, y_train)
    return model


def _build_profit_pipeline(X_train, y_train):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1,
        )),
    ])
    model.fit(X_train, y_train)
    return model


def train_and_save(status_fn=print) -> dict:
    """Train both models and persist to MODELS_DIR. Returns metrics."""
    MODELS_DIR.mkdir(exist_ok=True)

    status_fn("Loading and cleaning data…")
    df = clean(load_raw())

    status_fn("Building feature matrix…")
    X, y_sales, y_profit, feature_names = get_feature_matrix(df)

    X_tr, X_te, ys_tr, ys_te, yp_tr, yp_te = train_test_split(
        X, y_sales, y_profit, test_size=0.2, random_state=42
    )

    status_fn("Training Sales Amount model (Gradient Boosting)…")
    sales_model = _build_sales_pipeline(X_tr, ys_tr)
    sales_metrics = _evaluate(sales_model, X_te, ys_te)

    status_fn("Training Profit model (Random Forest)…")
    profit_model = _build_profit_pipeline(X_tr, yp_tr)
    profit_metrics = _evaluate(profit_model, X_te, yp_te)

    with open(MODELS_DIR / "sales_model.pkl", "wb") as f:
        pickle.dump(sales_model, f)
    with open(MODELS_DIR / "profit_model.pkl", "wb") as f:
        pickle.dump(profit_model, f)
    with open(MODELS_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f)

    metrics = {"sales": sales_metrics, "profit": profit_metrics}
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    status_fn("✓ Models saved.")
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Model Loading & Prediction Helpers
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_models():
    """Load trained models from disk. Cached for the session lifetime."""
    sales_pkl = MODELS_DIR / "sales_model.pkl"
    if not sales_pkl.exists():
        return None, None, [], None

    with open(sales_pkl, "rb") as f:
        sales_model = pickle.load(f)
    with open(MODELS_DIR / "profit_model.pkl", "rb") as f:
        profit_model = pickle.load(f)
    with open(MODELS_DIR / "feature_names.json") as f:
        feature_names = json.load(f)

    df = clean(load_raw())
    return sales_model, profit_model, feature_names, df


@st.cache_data(show_spinner=False)
def load_metrics() -> Optional[dict]:
    metrics_path = MODELS_DIR / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        return json.load(f)


def predict_order(req: dict, sales_model, profit_model, feature_names: list) -> dict:
    """Run both models on a single order dict and return predictions."""
    df_row = pd.get_dummies(pd.DataFrame([req]), columns=CATEGORICAL_COLS, drop_first=True)
    for col in feature_names:
        if col not in df_row.columns:
            df_row[col] = 0.0
    X = df_row[feature_names].astype(float)

    pred_sales = float(sales_model.predict(X)[0])
    pred_profit = float(profit_model.predict(X)[0])
    margin = (pred_profit / pred_sales * 100) if pred_sales > 0 else 0.0
    return {
        "predicted_sales_amount": round(pred_sales, 2),
        "predicted_profit": round(pred_profit, 2),
        "profit_margin_pct": round(margin, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Streamlit UI
# ══════════════════════════════════════════════════════════════════════════════

# ── UI reference data ─────────────────────────────────────────────────────────
CHANNELS = ["Online", "Retail Store", "B2B Portal", "Phone Order"]
CATEGORIES = ["Electronics", "Furniture", "Office Supplies", "Appliances"]
SUBCATEGORIES = {
    "Electronics": ["Laptops", "Tablets", "Monitors", "Accessories", "Cameras", "Storage"],
    "Furniture": ["Office Furniture", "Lighting"],
    "Office Supplies": ["Paper", "Writing"],
    "Appliances": ["Climate", "Home Office"],
}
SEGMENTS = ["Consumer", "Corporate", "Small Business", "Premium", "New Customer", "Returning Customer"]
GENDERS = ["Male", "Female", "Non-binary"]
PAYMENTS = ["Credit Card", "Debit Card", "Paypal", "Apple Pay", "Cash"]
SHIPPING = ["Standard", "Express", "Next Day", "Pickup"]
COUNTRIES = ["United States", "United Kingdom", "Canada", "Australia", "France", "Germany"]
REGIONS = {
    "United States": ["West", "Northeast", "Midwest", "South"],
    "United Kingdom": ["England"],
    "Canada": ["Ontario", "British Columbia"],
    "Australia": ["New South Wales"],
    "France": ["Île-de-France"],
    "Germany": ["North Rhine-Westphalia"],
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load models ───────────────────────────────────────────────────────────────
sales_model, profit_model, feature_names, df = load_models()
models_ready = sales_model is not None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Sales Prediction")
    st.caption("2022–2025 Transactions")

    page = st.radio(
        "Navigate",
        ["📈 Overview Dashboard", "🔮 Predict Sales", "📦 Product & Region", "📉 Forecasting"],
    )

    st.divider()

    if models_ready:
        st.success("Models loaded ✓", icon="✅")
    else:
        st.warning("Models not trained yet", icon="⚠️")
        if st.button("🚀 Train Models Now", use_container_width=True, type="primary"):
            progress = st.empty()
            msgs = []

            def update(msg):
                msgs.append(msg)
                progress.info("\n\n".join(msgs))

            with st.spinner("Training… this may take a minute."):
                try:
                    train_and_save(status_fn=update)
                    st.cache_resource.clear()
                    st.cache_data.clear()
                    st.success("Training complete! Reloading…")
                    st.rerun()
                except Exception as e:
                    st.error(f"Training failed: {e}")

# ── Guard: stop all pages until models are ready ──────────────────────────────
if not models_ready:
    st.info("👈 Train the models from the sidebar to get started.", icon="ℹ️")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview Dashboard
# ══════════════════════════════════════════════════════════════════════════════
if page == "📈 Overview Dashboard":
    st.header("📈 Sales Overview Dashboard")

    # ── KPIs ──────────────────────────────────────────────────────────────────
    total_revenue = float(df["Sales_Amount"].sum())
    total_profit = float(df["Profit"].sum())
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Revenue", f"${total_revenue:,.0f}")
    c2.metric("Total Profit", f"${total_profit:,.0f}")
    c3.metric("Total Orders", f"{df['Transaction_ID'].nunique():,}")
    c4.metric("Avg Order Value", f"${df['Sales_Amount'].mean():,.2f}")
    c5.metric("Profit Margin", f"{total_profit / total_revenue * 100:.1f}%")

    st.divider()

    # ── Monthly revenue ───────────────────────────────────────────────────────
    df_m = monthly_revenue(df)
    fig_line = px.line(
        df_m, x="YearMonth", y="Revenue",
        title="Monthly Revenue (2022–2025)",
        labels={"YearMonth": "Month", "Revenue": "Revenue ($)"},
        markers=True,
        color_discrete_sequence=["#3b82d4"],
    )
    fig_line.update_layout(xaxis_tickangle=-45, height=350)
    st.plotly_chart(fig_line, use_container_width=True)

    # ── Channel & Category ────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    df_ch = df.groupby("Sales_Channel")["Sales_Amount"].sum().reset_index()
    fig_pie = px.pie(
        df_ch, names="Sales_Channel", values="Sales_Amount",
        title="Revenue by Sales Channel",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    col_a.plotly_chart(fig_pie, use_container_width=True)

    df_cat = (
        df.groupby("Product_Category")["Sales_Amount"]
        .sum().reset_index()
        .sort_values("Sales_Amount", ascending=False)
    )
    fig_bar = px.bar(
        df_cat, x="Product_Category", y="Sales_Amount",
        title="Revenue by Product Category",
        color="Product_Category",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        text_auto=".2s",
    )
    fig_bar.update_layout(showlegend=False, height=350)
    col_b.plotly_chart(fig_bar, use_container_width=True)

    # ── Model metrics ─────────────────────────────────────────────────────────
    metrics = load_metrics()
    if metrics:
        st.subheader("🧠 Model Performance")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Sales MAE", f"${metrics['sales']['MAE']:,.2f}")
        m2.metric("Sales RMSE", f"${metrics['sales']['RMSE']:,.2f}")
        m3.metric("Sales R²", f"{metrics['sales']['R2']:.4f}")
        m4.metric("Profit MAE", f"${metrics['profit']['MAE']:,.2f}")
        m5.metric("Profit RMSE", f"${metrics['profit']['RMSE']:,.2f}")
        m6.metric("Profit R²", f"{metrics['profit']['R2']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Predict Sales
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Sales":
    st.header("🔮 Predict Sales Amount & Profit")
    st.caption("Fill in order details to get an instant ML prediction.")

    with st.form("prediction_form"):
        st.subheader("Order Details")
        col1, col2, col3 = st.columns(3)

        with col1:
            quantity = col1.number_input("Quantity", min_value=1, max_value=100, value=2)
            unit_price = col1.number_input("Unit Price ($)", min_value=1.0, value=120.0, step=0.5)
            discount_pct = col1.slider("Discount (%)", 0, 50, 10)
            customer_age = col1.slider("Customer Age", 18, 80, 35)

        with col2:
            sales_channel = col2.selectbox("Sales Channel", CHANNELS)
            product_category = col2.selectbox("Product Category", CATEGORIES)
            subcats = SUBCATEGORIES.get(product_category, ["Other"])
            product_subcat = col2.selectbox("Product Subcategory", subcats)
            customer_segment = col2.selectbox("Customer Segment", SEGMENTS)

        with col3:
            customer_gender = col3.selectbox("Customer Gender", GENDERS)
            payment_method = col3.selectbox("Payment Method", PAYMENTS)
            shipping_method = col3.selectbox("Shipping Method", SHIPPING)
            delivery_days = col3.slider("Delivery Days", 0, 14, 3)

        st.subheader("Location & Timing")
        t1, t2, t3, t4, t5 = st.columns(5)
        country = t1.selectbox("Country", COUNTRIES)
        region = t2.selectbox("Region", REGIONS.get(country, ["Other"]))
        order_month = t3.slider("Order Month", 1, 12, 6)
        order_dow = t4.slider("Day of Week (0=Mon)", 0, 6, 2)
        order_hour = t5.slider("Order Hour", 0, 23, 14)

        submitted = st.form_submit_button("🚀 Predict", type="primary", use_container_width=True)

    if submitted:
        req = {
            "Quantity": quantity,
            "Unit_Price": unit_price,
            "Discount_Percentage": discount_pct,
            "Customer_Age": customer_age,
            "Delivery_Days": delivery_days,
            "Order_Month": order_month,
            "Order_DayOfWeek": order_dow,
            "Order_Hour": order_hour,
            "Sales_Channel": sales_channel,
            "Product_Category": product_category,
            "Product_Subcategory": product_subcat,
            "Customer_Segment": customer_segment,
            "Customer_Gender": customer_gender,
            "Payment_Method": payment_method,
            "Shipping_Method": shipping_method,
            "Country": country,
            "Region": region,
        }
        result = predict_order(req, sales_model, profit_model, feature_names)

        st.success("Prediction complete!")
        r1, r2, r3 = st.columns(3)
        r1.metric("💰 Predicted Sales Amount", f"${result['predicted_sales_amount']:,.2f}")
        r2.metric("📈 Predicted Profit", f"${result['predicted_profit']:,.2f}")
        r3.metric("💹 Profit Margin", f"{result['profit_margin_pct']}%")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result["profit_margin_pct"],
            title={"text": "Profit Margin %"},
            gauge={
                "axis": {"range": [0, 60]},
                "bar": {"color": "#3b82d4"},
                "steps": [
                    {"range": [0, 15], "color": "#fee2e2"},
                    {"range": [15, 35], "color": "#fef9c3"},
                    {"range": [35, 60], "color": "#dcfce7"},
                ],
            },
        ))
        fig_gauge.update_layout(height=280)
        st.plotly_chart(fig_gauge, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Product & Region
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📦 Product & Region":
    st.header("📦 Product & Regional Analysis")

    col_left, col_right = st.columns(2)

    df_tp = (
        df.groupby("Product_Name")["Sales_Amount"]
        .sum().nlargest(15).reset_index()
        .rename(columns={"Sales_Amount": "Total_Sales"})
        .sort_values("Total_Sales")
    )
    fig_tp = px.bar(
        df_tp, x="Total_Sales", y="Product_Name", orientation="h",
        title="Top 15 Products by Revenue",
        labels={"Total_Sales": "Revenue ($)", "Product_Name": "Product"},
        color="Total_Sales", color_continuous_scale="Blues", text_auto=".2s",
    )
    fig_tp.update_layout(height=480, coloraxis_showscale=False)
    col_left.plotly_chart(fig_tp, use_container_width=True)

    df_reg = (
        df.groupby("Region")["Sales_Amount"]
        .sum().reset_index()
        .rename(columns={"Sales_Amount": "Total_Sales"})
        .sort_values("Total_Sales", ascending=False)
    )
    fig_reg = px.bar(
        df_reg, x="Region", y="Total_Sales",
        title="Revenue by Region",
        color="Total_Sales", color_continuous_scale="Purples", text_auto=".2s",
    )
    fig_reg.update_layout(height=480, xaxis_tickangle=-30, coloraxis_showscale=False)
    col_right.plotly_chart(fig_reg, use_container_width=True)

    df_m = monthly_revenue(df)
    fig_orders = px.bar(
        df_m, x="YearMonth", y="Orders",
        title="Monthly Order Volume",
        color_discrete_sequence=["#7c5cd8"],
    )
    fig_orders.update_layout(xaxis_tickangle=-45, height=320)
    st.plotly_chart(fig_orders, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Forecasting
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📉 Forecasting":
    st.header("📉 Revenue Forecasting")
    st.caption("Linear trend forecast built on historical monthly revenue.")

    df_m = monthly_revenue(df).sort_values("YearMonth").reset_index(drop=True)
    df_m["Index"] = df_m.index

    X_idx = df_m["Index"].values.reshape(-1, 1)
    y_rev = df_m["Revenue"].values
    lr = LinearRegression().fit(X_idx, y_rev)

    forecast_months = st.slider("Months to forecast ahead", 3, 24, 12)
    last_idx = int(df_m["Index"].max())

    future_idx = np.arange(last_idx + 1, last_idx + 1 + forecast_months).reshape(-1, 1)
    future_rev = lr.predict(future_idx)

    last_period = pd.Period(df_m["YearMonth"].iloc[-1], freq="M")
    future_labels = [(last_period + i + 1).strftime("%Y-%m") for i in range(forecast_months)]

    df_forecast = pd.DataFrame({
        "YearMonth": future_labels,
        "Revenue": future_rev,
        "Type": "Forecast",
    })
    df_hist = df_m[["YearMonth", "Revenue"]].copy()
    df_hist["Type"] = "Historical"
    df_combined = pd.concat([df_hist, df_forecast], ignore_index=True)

    fig_forecast = px.line(
        df_combined, x="YearMonth", y="Revenue", color="Type",
        title=f"Monthly Revenue: Historical + {forecast_months}-Month Forecast",
        labels={"YearMonth": "Month", "Revenue": "Revenue ($)"},
        color_discrete_map={"Historical": "#3b82d4", "Forecast": "#f97316"},
        markers=True,
    )
    fig_forecast.add_vline(
        x=df_m["YearMonth"].iloc[-1], line_dash="dot", line_color="gray",
        annotation_text="Forecast start",
    )
    fig_forecast.update_layout(xaxis_tickangle=-45, height=420)
    st.plotly_chart(fig_forecast, use_container_width=True)

    s1, s2, s3 = st.columns(3)
    s1.metric("Avg Forecast Revenue/Month", f"${np.mean(future_rev):,.0f}")
    s2.metric("Total Forecast Revenue", f"${np.sum(future_rev):,.0f}")
    growth = (future_rev[-1] - y_rev[-1]) / y_rev[-1] * 100
    s3.metric("Projected Growth vs Last Month", f"{growth:+.1f}%")

    with st.expander("📋 Forecast Data Table"):
        df_show = df_forecast[["YearMonth", "Revenue"]].copy()
        df_show["Revenue"] = df_show["Revenue"].map("${:,.2f}".format)
        st.dataframe(df_show, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry-point:  python app.py train
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "train":
    train_and_save()

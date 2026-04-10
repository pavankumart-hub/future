# dashboard_app.py
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import timedelta

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

# -----------------------------
# Configuration
# -----------------------------
TICKER = "TCS.NS"
START_DATE = "2015-01-01"
MODEL_PATH = "tft_tcs_model.ckpt"
MAX_ENCODER_LENGTH = 60
MAX_PREDICTION_LENGTH = 20

st.set_page_config(page_title="TFT Forecast Dashboard", layout="wide")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = yf.download(TICKER, start=START_DATE, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df["time_idx"] = (df["Date"] - df["Date"].min()).dt.days
    df["series"] = "TCS"
    df["month"] = df["Date"].dt.month.astype(str)
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["log_return"].fillna(0, inplace=True)
    return df

# -----------------------------
# Load TFT Model
# -----------------------------
@st.cache_resource
def load_tft_model():
    model = TemporalFusionTransformer.load_from_checkpoint(MODEL_PATH)
    model.eval()
    return model

# -----------------------------
# Generate Forecast
# -----------------------------
def generate_forecast(df, horizon):
    model = load_tft_model()

    dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="log_return",
        group_ids=["series"],
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=horizon,
        static_categoricals=["series"],
        time_varying_known_categoricals=["month"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["log_return"],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    loader = dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

    # Predict quantiles
    predictions = model.predict(loader, mode="quantiles", quantiles=[0.1, 0.5, 0.9])
    preds = predictions[0]

    last_price = df["Close"].iloc[-1]
    forecast_prices = last_price * np.exp(np.cumsum(preds[:, 1]))
    upper = last_price * np.exp(np.cumsum(preds[:, 2]))
    lower = last_price * np.exp(np.cumsum(preds[:, 0]))

    future_dates = pd.bdate_range(
        start=df["Date"].iloc[-1] + timedelta(days=1),
        periods=horizon
    )

    return pd.DataFrame({
        "date": future_dates,
        "forecast": forecast_prices,
        "upper": upper,
        "lower": lower
    })

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🔮 Temporal Fusion Transformer Forecast for TCS")

df = load_data()
model = load_tft_model()

horizon = st.slider("Forecast Horizon (Days)", 5, MAX_PREDICTION_LENGTH, 20)

forecast_df = generate_forecast(df, horizon)

# -----------------------------
# Plot Results
# -----------------------------
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["Date"].tail(120),
    y=df["Close"].tail(120),
    name="Historical Price",
    line=dict(color="blue")
))

fig.add_trace(go.Scatter(
    x=forecast_df["date"],
    y=forecast_df["forecast"],
    name="TFT Forecast",
    line=dict(color="orange", dash="dash")
))

fig.add_trace(go.Scatter(
    x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
    y=pd.concat([forecast_df["upper"], forecast_df["lower"][::-1]]),
    fill="toself",
    fillcolor="rgba(255,165,0,0.2)",
    line=dict(color="rgba(255,255,255,0)"),
    name="Prediction Interval"
))

fig.update_layout(
    template="plotly_dark",
    height=500,
    xaxis_title="Date",
    yaxis_title="Price (INR)",
)

st.plotly_chart(fig, use_container_width=True)

st.success("✅ Real TFT forecast generated successfully.")

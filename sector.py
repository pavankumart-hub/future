# -*- coding: utf-8 -*-
"""
dashboard_app.py
================
Streamlit Dashboard — Financial AI Decision-Support System.

Run:  streamlit run dashboard_app.py
"""

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Sector Future | AI Financial Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main {background-color: #0e1117;}
    .stMetric {background-color: #1c2130; border-radius: 8px; padding: 12px;}
    .stMetric label {color: #8b9ab5; font-size: 0.85rem;}
    .stMetric .metric-value {font-size: 1.6rem; font-weight: 700;}
    .block-container {padding-top: 1rem;}
    h1, h2, h3 {color: #e8ecf4;}
    .tag-green {color: #00c853; font-weight: 600;}
    .tag-red   {color: #ff1744; font-weight: 600;}
    .info-box  {background: #1c2130; border-radius: 8px; padding: 16px;
                border-left: 4px solid #4a90d9; margin-bottom: 12px;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
TICKER = "TCS.NS"
START_DATE = "2015-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

PORTFOLIO = {
    "TCS.NS": {"qty": 50, "avg_cost": 3400.0},
    "INFY.NS": {"qty": 100, "avg_cost": 1650.0},
    "HDFCBANK.NS": {"qty": 80, "avg_cost": 1580.0},
    "RELIANCE.NS": {"qty": 30, "avg_cost": 2900.0},
    "WIPRO.NS": {"qty": 200, "avg_cost": 540.0},
}

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def _mock_forecast(close: pd.Series, horizon: int = 30) -> pd.DataFrame:
    last_date = close.index[-1]
    last_price = float(close.iloc[-1])
    log_ret = np.log(close / close.shift(1)).dropna()

    mean_r = log_ret.mean()
    std_r = log_ret.std()

    rng = np.random.default_rng(42)
    simulated_returns = rng.normal(mean_r, std_r * 0.5, horizon)

    prices = [last_price]
    for r in simulated_returns:
        prices.append(prices[-1] * np.exp(r))
    prices = prices[1:]

    dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=horizon)

    return pd.DataFrame({
        "date": dates,
        "forecast": prices,
        "upper": [p * 1.05 for p in prices],
        "lower": [p * 0.95 for p in prices],
    })

# ─────────────────────────────────────────────
# DATA LOADING FUNCTIONS
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_tcs_data():
    df = yf.download(
        TICKER,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        st.error("No data returned from Yahoo Finance.")
        return pd.DataFrame()

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Convert column names to lowercase
    df.columns = [str(col).lower() for col in df.columns]

    # Ensure required columns exist
    required_cols = {"open", "high", "low", "close", "volume"}
    if not required_cols.issubset(df.columns):
        st.error("Missing required columns in downloaded data.")
        return pd.DataFrame()

    # Technical Indicators
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["rsi14"] = _rsi(df["close"])
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    df["macd"] = df["close"].ewm(span=12, adjust=False).mean() - \
                 df["close"].ewm(span=26, adjust=False).mean()
    df["macd_sig"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["bb_mid"] = df["close"].rolling(20).mean()
    df["bb_upper"] = df["bb_mid"] + 2 * df["close"].rolling(20).std()
    df["bb_lower"] = df["bb_mid"] - 2 * df["close"].rolling(20).std()

    return df.dropna()

@st.cache_data(ttl=3600)
def load_portfolio_prices():
    tickers = list(PORTFOLIO.keys())

    raw = yf.download(
        tickers,
        start="2020-01-01",
        end=END_DATE,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        st.error("No portfolio data returned from Yahoo Finance.")
        return pd.DataFrame()

    # Extract Close prices
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw.copy()

    prices = prices.ffill().dropna(how="all")
    return prices

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Sector Future")
    page = st.radio(
        "Navigation",
        ["📈 Market Overview", "🔮 TFT Forecast", "💼 Portfolio Analytics"]
    )

    date_from = st.date_input("From", value=datetime(2020, 1, 1))
    date_to = st.date_input("To", value=datetime.today())

# ─────────────────────────────────────────────
# PAGE 1: MARKET OVERVIEW
# ─────────────────────────────────────────────
if page == "📈 Market Overview":
    st.title("📈 TCS.NS — Market Overview")

    df = load_tcs_data()
    if df.empty:
        st.stop()

    df_range = df.loc[str(date_from):str(date_to)]
    latest = df_range.iloc[-1]
    prev = df_range.iloc[-2]

    pct_chg = (latest["close"] - prev["close"]) / prev["close"] * 100
    ytd_start = df_range[df_range.index.year == latest.name.year]["close"].iloc[0]
    ytd_ret = (latest["close"] - ytd_start) / ytd_start * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Last Price", f"₹{latest['close']:,.2f}", f"{pct_chg:+.2f}%")
    c2.metric("Volume", f"{latest['volume']/1e6:.2f}M")
    c3.metric("52W High", f"₹{df_range['high'].rolling(252).max().iloc[-1]:,.0f}")
    c4.metric("52W Low", f"₹{df_range['low'].rolling(252).min().iloc[-1]:,.0f}")
    c5.metric("YTD Return", f"{ytd_ret:+.2f}%")

    # Candlestick Chart
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(
        x=df_range.index,
        open=df_range["open"],
        high=df_range["high"],
        low=df_range["low"],
        close=df_range["close"],
        name="Price"
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df_range.index,
        y=df_range["volume"],
        name="Volume"
    ), row=2, col=1)

    fig.update_layout(template="plotly_dark", height=600)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 2: TFT FORECAST
# ─────────────────────────────────────────────
elif page == "🔮 TFT Forecast":
    st.title("🔮 Temporal Fusion Transformer — Price Forecast")

    df = load_tcs_data()
    if df.empty:
        st.stop()

    horizon = st.slider("Forecast Horizon", 5, 60, 20)
    fcast_df = _mock_forecast(df["close"], horizon)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["close"].tail(120).index,
        y=df["close"].tail(120),
        name="Historical"
    ))
    fig.add_trace(go.Scatter(
        x=fcast_df["date"],
        y=fcast_df["forecast"],
        name="Forecast",
        line=dict(dash="dash")
    ))

    fig.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 3: PORTFOLIO ANALYTICS
# ─────────────────────────────────────────────
elif page == "💼 Portfolio Analytics":
    st.title("💼 Portfolio Optimisation & Risk Analytics")

    prices_df = load_portfolio_prices()
    if prices_df.empty:
        st.stop()

    tickers = list(PORTFOLIO.keys())
    prices = prices_df[tickers].dropna()
    log_ret = np.log(prices / prices.shift(1)).dropna()

    weights = np.ones(len(tickers)) / len(tickers)
    portfolio_returns = (log_ret @ weights)
    cumulative_returns = (1 + portfolio_returns).cumprod()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cumulative_returns.index,
        y=cumulative_returns,
        name="Portfolio Growth"
    ))
    fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Matrix")
    corr = log_ret.corr()
    st.dataframe(corr)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>"
    "Sector Future Financial AI Platform · Internship Project · 2026"
    "</div>",
    unsafe_allow_html=True
)

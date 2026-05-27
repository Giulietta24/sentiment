import streamlit as st
import yfinance as yf
import pandas as pd
from transformers import pipeline

st.set_page_config(page_title="Narrative Sentiment Engine", layout="wide")

st.title("📈 Narrative Sentiment Engine (US + UK + Europe)")

st.write("### 🔍 Analyzing tickers:")

# -----------------------
# SAFELY LOAD PRICE DATA
# -----------------------
def safe_load_price(ticker):
    try:
        # group_by="column" ensures standard formatting
        df = yf.download(ticker, period="6mo", interval="1d", group_by="column")
        
        if df is None or df.empty:
            st.warning(f"⚠️ No price data returned for **{ticker}**")
            return None

        # FIX: Flatten Multi-Index columns if yfinance returns them (e.g., ('Close', 'IWM') -> 'Close')
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if "Close" not in df.columns:
            st.warning(f"⚠️ Missing 'Close' column for **{ticker}**")
            return None

        if len(df) < 31:
            st.warning(f"⚠️ Not enough price history for **{ticker}** (need 31 days)")
            return None

        return df

    except Exception as e:
        st.error(f"❌ Error loading price for {ticker}: {e}")
        return None


# --------------------------------------------
# SAFE PRICE CHANGE CALCULATION (NO CRASHES)
# --------------------------------------------
def safe_price_change(price_df):
    try:
        close = price_df["Close"]

        # Must have enough rows
        if len(close) < 31:
            return None

        # SAFE ACCESS: iloc prevents KeyError
        last = close.iloc[-1]
        last_30 = close.iloc[-30]

        if pd.isna(last) or pd.isna(last_30):
            return None

        # Handle cases where 'last' might still be a Series
        if isinstance(last, pd.Series):
            last = last.iloc[0]
        if isinstance(last_30, pd.Series):
            last_30 = last_30.iloc[0]

        return float(((last - last_30) / last_30) * 100)

    except Exception:
        return None


# ----------------------------
# NEWS / NARRATIVE SENTIMENT
# ----------------------------
@st.cache_resource # Keeps your app fast so it doesn't reload the model on every click
def load_sentiment_model():
    return pipeline("sentiment-analysis")

sentiment_model = load_sentiment_model()

def analyze_sentiment(texts):
    if not texts:
        return []

    try:
        return sentiment_model(texts)
    except Exception:
        return [{"label": "NEUTRAL", "score": 0.00} for _ in texts]


# -----------------------------------------
# MAIN APPLICATION
# -----------------------------------------
# FIX: Changed 
tickers = ["IWM", "IJR", "FCIT.L", "SMEU.L"]

st.json(tickers)

results = []

for ticker in tickers:
    st.subheader(f"📊 {ticker}")

    price = safe_load_price(ticker)
    if price is None:
        st.write("❌ Skipping due to missing data")
        continue

    change_30d = safe_price_change(price)
    if change_30d is None:
        st.warning(f"⚠️ Unable to compute 30-day price change for {ticker}")
        continue

    st.write(f"**30-day Price Change:** {change_30d:.2f}%")

    # Dummy news for now (safe placeholder)
    news = [f"{ticker} market sentiment update."]

    sentiments = analyze_sentiment(news)

    results.append({
        "Ticker": ticker,
        "30d Change": change_30d,
        "Sentiment": sentiments[0]["label"],
        "Sentiment Score": sentiments[0]["score"]
    })

if results:
    df = pd.DataFrame(results)
    st.write("### 📌 Final Output")
    st.dataframe(df)
else:
    st.info("No data successfully processed.")

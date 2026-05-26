
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from transformers import pipeline
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Narrative Sentiment Engine",
    layout="wide"
)

st.title("📈 Narrative Sentiment Engine (US + UK + Europe)")

st.sidebar.header("Market Universe")

region_us = st.sidebar.checkbox("🇺🇸 US Small Caps (Russell 2000, S&P 600)", True)
region_uk = st.sidebar.checkbox("🇬🇧 UK Small Caps (FTSE SmallCap)", True)
region_eu = st.sidebar.checkbox("🇪🇺 Europe Small Caps (STOXX / MSCI)", True)

include_large_caps = st.sidebar.checkbox("Include Large Caps (S&P500, FTSE100, DAX)", False)

sent_model = pipeline("sentiment-analysis", model="ProsusAI/finbert")

US_SMALL = ["IWM", "IJR"]
UK_SMALL = ["FSC.L"]
EU_SMALL = ["SMEU"]

US_LARGE = ["SPY"]
UK_LARGE = ["UKX.L"]
EU_LARGE = ["DAX"]

universe = []

if region_us:
    universe += US_SMALL
if region_uk:
    universe += UK_SMALL
if region_eu:
    universe += EU_SMALL

if include_large_caps:
    universe += US_LARGE + UK_LARGE + EU_LARGE

st.subheader("🔍 Analyzing tickers:")
st.write(universe)

def get_price(ticker):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        return df
    except:
        return None

def get_sentiment_from_text(text):
    if not text:
        return 0
    try:
        result = sent_model(text[:512])[0]
        if result['label'] == "positive":
            return result['score'] * 100
        elif result['label'] == "negative":
            return -result['score'] * 100
        return 0
    except:
        return 0

EXAMPLE_NEWS = {
    "IWM": ["Small caps rally on rate cut expectations", "Investors rotate back into growth"],
    "IJR": ["Manufacturing outlook improves", "Liquidity conditions still tight"],
    "FSC.L": ["UK small caps rebound as inflation slows", "IPO pipeline strengthens"],
    "SMEU": ["European small caps benefit from ECB guidance", "Energy prices easing supports SMEs"],
    "SPY": ["AI megacap rally continues", "Soft landing narrative remains intact"],
}

results = []

for ticker in universe:
    price = get_price(ticker)

    if price is None or len(price) == 0:
        continue

    price_change_30d = (price['Close'][-1] - price['Close'][-30]) / price['Close'][-30] * 100

    headlines = EXAMPLE_NEWS.get(ticker, ["No major headlines"])
    narrative_score = np.mean([get_sentiment_from_text(h) for h in headlines])

    final_score = (0.6 * narrative_score) + (0.4 * price_change_30d)

    trend = "⬆️ Rising" if final_score > 20 else "⬇️ Falling" if final_score < -20 else "➡️ Neutral"

    results.append({
        "Ticker": ticker,
        "Narrative Score": round(narrative_score, 2),
        "30D Price Change %": round(price_change_30d, 2),
        "Final Sentiment Score": round(final_score, 2),
        "Trend": trend,
        "Narrative": ", ".join(headlines)
    })

df = pd.DataFrame(results).sort_values("Final Sentiment Score", ascending=False)

st.subheader("📊 Narrative Sentiment Ranking")
st.dataframe(df, use_container_width=True)

st.subheader("📈 Top BUY Signals")
st.write(df.head(5))

st.subheader("📉 Top SELL Signals")
st.write(df.tail(5))

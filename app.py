# Narrative Sentiment Dashboard — C-Plus Version
# (Moderately expanded functional edition)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
import plotly.express as px
from io import BytesIO

st.set_page_config(page_title="Narrative Sentiment Dashboard", layout="wide")

########################################
# Load Model
########################################
@st.cache_resource
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

tokenizer, model = load_finbert()

########################################
# Sentiment Engine
########################################
def finbert_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    probs = torch.softmax(logits, dim=0)
    labels = ["negative", "neutral", "positive"]
    return dict(zip(labels, probs.tolist()))

########################################
# News Fetch (Yahoo Finance)
########################################
def fetch_news(ticker):
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={ticker}"
    try:
        r = requests.get(url, timeout=6).json()
        return [n.get("title","") for n in r.get("news", [])][:10]
    except:
        return []

########################################
# UI
########################################
st.title("📈 Narrative Sentiment Dashboard — C‑Plus Version")

tickers = st.text_input("Enter tickers (comma separated):", "AAPL,MSFT,TSLA").upper().split(",")
tickers = [t.strip() for t in tickers if t.strip()]

########################################
# Processing
########################################
results = {}

for t in tickers:
    news = fetch_news(t)
    scores = []
    for n in news:
        if n.strip():
            scores.append(finbert_sentiment(n)["positive"])
    avg_sent = np.mean(scores) if scores else 0.33

    results[t] = {
        "sentiment": avg_sent,
        "news": news
    }

########################################
# Display Global Metrics
########################################
global_sent = np.mean([v["sentiment"] for v in results.values()])
st.metric("🌍 Global Narrative Sentiment", f"{global_sent:.2f}")

########################################
# Bar Chart
########################################
fig = px.bar(
    x=list(results.keys()),
    y=[v["sentiment"] for v in results.values()],
    title="Ticker Narrative Sentiment Scores",
    labels={"x": "Ticker", "y": "Sentiment Score"},
)
st.plotly_chart(fig)

########################################
# Detailed Breakdown
########################################
st.subheader("Ticker Details")

for t, d in results.items():
    st.write(f"### {t}")
    st.write(f"**Narrative Sentiment:** {d['sentiment']:.2f}")
    for n in d["news"]:
        st.write("-", n)

########################################
# Export to Excel
########################################
if st.button("Export Results to Excel"):
    df = pd.DataFrame([
        {"ticker": t, "sentiment": d["sentiment"]}
        for t, d in results.items()
    ])
    bio = BytesIO()
    df.to_excel(bio, index=False)
    st.download_button("Download Excel", bio.getvalue(), "sentiment_output.xlsx")

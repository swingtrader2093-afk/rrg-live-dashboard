import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("🇮🇳 Nifty Sector Relative Rotation Graph (RRG)")

# =============================
# CONFIG
# =============================
benchmark = "^NSEI"

sectors = {
    "Auto": "^CNXAUTO",
    "Bank": "^NSEBANK",
    "FMCG": "^CNXFMCG",
    "IT": "^CNXIT",
    "Media": "^CNXMEDIA",
    "Metal": "^CNXMETAL",
    "Pharma": "^CNXPHARMA",
    "PSU Bank": "^CNXPSUBANK",
    "Realty": "^CNXREALTY",
}

period = st.sidebar.selectbox("Period", ["6mo", "1y", "2y"], index=1)
interval = "1wk"

# =============================
# DATA FETCH
# =============================
@st.cache_data(ttl=3600)
def fetch(symbol):
    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df.empty:
        return None
    return df["Close"]

bench = fetch(benchmark)
if bench is None:
    st.error("Benchmark data not available")
    st.stop()

fig = go.Figure()

valid_count = 0

for name, symbol in sectors.items():
    try:
        sec = fetch(symbol)

        if sec is None:
            continue

        df = pd.concat([sec, bench], axis=1)
        df.columns = ["sec", "bench"]
        df = df.dropna()


        if len(df) < 30:
            continue

        # Relative strength
        df["RS"] = df["sec"] / df["bench"]

        # JdK approximations
        df["RS_ratio"] = (df["RS"] / df["RS"].rolling(10).mean()) * 100
        df["RS_mom"] = (df["RS_ratio"] / df["RS_ratio"].rolling(10).mean()) * 100

        tail = df.dropna().tail(10)

        if len(tail) == 0:
            continue

        valid_count += 1

        fig.add_trace(go.Scatter(
            x=tail["RS_ratio"],
            y=tail["RS_mom"],
            mode="lines+markers+text",
            text=[name] + [""]*(len(tail)-1),
            textposition="top center",
            marker=dict(size=7),
            name=name
        ))

    except Exception:
        continue

# =============================
# QUADRANT LINES
# =============================
fig.add_vline(x=100, line_dash="dash")
fig.add_hline(y=100, line_dash="dash")

fig.update_layout(
    height=760,
    xaxis=dict(range=[85,115], title="RS Ratio"),
    yaxis=dict(range=[85,115], title="RS Momentum"),
    legend=dict(orientation="h"),
)

st.plotly_chart(fig, width="stretch")

st.caption(f"✅ Active sectors plotted: {valid_count}")

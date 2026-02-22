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

# =============================
# QUADRANT COLOR MAP
# =============================
quad_colors = {
    "Leading": "#00C853",     # green
    "Weakening": "#FF9800",   # orange
    "Lagging": "#FF5252",     # red
    "Improving": "#2979FF",   # blue
}

# =============================
# SECTOR COLOR PALETTE
# =============================
import plotly.express as px
sector_palette = px.colors.qualitative.Set2


period = st.sidebar.selectbox("Period", ["6mo", "1y", "2y"], index=1)

playback = st.sidebar.slider(
    "RRG History (bars)",
    min_value=5,
    max_value=30,
    value=10
)

timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["Daily", "Weekly", "Monthly"],
    index=1
)

interval_map = {
    "Daily": "1d",
    "Weekly": "1wk",
    "Monthly": "1mo"
}

interval = interval_map[timeframe]


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
    
# =============================
# TIME CONTROL (DATE-BASED — PRO)
# =============================
bench_clean = bench.dropna()

available_dates = bench_clean.index

selected_date = st.select_slider(
    "⏳ RRG Time Position",
    options=available_dates,
    value=available_dates[-1],
)

# convert back to positional index for existing logic
time_index = bench_clean.index.get_loc(selected_date)

# =============================
# SHOW SELECTED DATE
# =============================
selected_date = bench_clean.index[time_index]
st.caption(f"📅 Selected Date: {selected_date.date()}")

# =============================
# NIFTY PRICE CHART (TOP) — WITH VERTICAL CROSSHAIR
# =============================
nifty_df = yf.download(
    benchmark,
    period=period,
    interval=interval,
    auto_adjust=True,
    progress=False,
    threads=False,
)

if not nifty_df.empty:
    st.subheader("📈 Nifty 50")

    nifty_clean = nifty_df["Close"].squeeze().dropna()

    # --- use globally selected date for perfect sync ---
    selected_ts = selected_date
    
    # if date not present in nifty (daily gaps), snap to nearest
    if selected_ts not in nifty_clean.index:
        nearest_loc = nifty_clean.index.get_indexer([selected_ts], method="nearest")[0]
        selected_ts = nifty_clean.index[nearest_loc]


    nifty_fig = go.Figure()

    # price line
    nifty_fig.add_trace(
        go.Scatter(
            x=nifty_clean.index,
            y=nifty_clean.values,
            mode="lines",
            name="Nifty 50",
            line=dict(width=2),
        )
    )

    # 🔥 vertical crosshair synced to slider
    nifty_fig.add_vline(
        x=selected_ts,
        line_width=2,
        line_dash="dash",
        line_color="red",
        opacity=0.7,
    )

    nifty_fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
        hovermode="x unified",
    )

    st.plotly_chart(
        nifty_fig,
        width="stretch",
        config={"displaylogo": False},
    )

fig = go.Figure()

# =============================
# QUADRANT BACKGROUND
# =============================
fig.add_shape(type="rect", x0=100, y0=100, x1=115, y1=115,
              fillcolor="rgba(0,200,0,0.08)", line_width=0)  # Leading

fig.add_shape(type="rect", x0=85, y0=100, x1=100, y1=115,
              fillcolor="rgba(0,0,255,0.08)", line_width=0)  # Improving

fig.add_shape(type="rect", x0=85, y0=85, x1=100, y1=100,
              fillcolor="rgba(255,0,0,0.08)", line_width=0)  # Lagging

fig.add_shape(type="rect", x0=100, y0=85, x1=115, y1=100,
              fillcolor="rgba(255,165,0,0.08)", line_width=0)  # Weakening

ranking_rows = []

valid_count = 0

for name, symbol in sectors.items():
    try:
        sec = fetch(symbol)

        if sec is None:
            continue

        df = pd.concat([sec, bench], axis=1)
        df.columns = ["sec", "bench"]
        df = df.dropna()


        if len(df) < 15:
            continue

        # Relative strength
        df["RS"] = df["sec"] / df["bench"]

        # JdK approximations
        df["RS_ratio"] = (df["RS"] / df["RS"].rolling(6).mean()) * 100
        df["RS_mom"] = (df["RS_ratio"] / df["RS_ratio"].rolling(6).mean()) * 100

        hist = df.dropna()

        # --- align time safely ---
        safe_index = min(time_index, len(hist) - 1)
        
        start_idx = max(0, safe_index - playback)
        tail = hist.iloc[start_idx:safe_index + 1]



        if len(tail) == 0:
            continue

        # ===== ADD THIS BLOCK HERE =====
        last_rs = tail["RS_ratio"].iloc[-1]
        last_mom = tail["RS_mom"].iloc[-1]
        
        if last_rs > 100 and last_mom > 100:
            quad = "Leading"
        elif last_rs > 100 and last_mom < 100:
            quad = "Weakening"
        elif last_rs < 100 and last_mom < 100:
            quad = "Lagging"
        else:
            quad = "Improving"
        # ===== END BLOCK =====

        ranking_rows.append({
            "Sector": name,
            "RS_Ratio": round(last_rs, 2),
            "RS_Momentum": round(last_mom, 2),
            "Quadrant": quad,
        })
        
        valid_count += 1

        base_color = sector_palette[list(sectors.keys()).index(name) % len(sector_palette)]
        color = base_color

        # tail line (soft glow)
        fig.add_trace(go.Scatter(
            x=tail["RS_ratio"],
            y=tail["RS_mom"],
            mode="lines",
            text=[name] * len(tail),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "RS Ratio: %{x:.2f}<br>"
                "RS Momentum: %{y:.2f}<extra></extra>"
            ),
            legendgroup=name,
            line=dict(width=3, color=color),
            opacity=0.35,
            showlegend=False,
        ))
        
        # current point index
        sizes = [6] * len(tail)
        sizes[-1] = 16  # 🔥 bigger current point
        
        fig.add_trace(go.Scatter(
            x=tail["RS_ratio"],
            y=tail["RS_mom"],
            mode="markers+text",
            hovertemplate=(
                "<b>%{text}</b><br>"
                "RS Ratio: %{x:.2f}<br>"
                "RS Momentum: %{y:.2f}<extra></extra>"
            ),
            legendgroup=name,
            text=[f"{name}"] + [""]*(len(tail)-1),
            textposition="top center",
            marker=dict(
                size=sizes,
                color=color,
                line=dict(width=1, color="white")
            ),
            name=name,
            legendgrouptitle_text=name,
        ))



    except Exception:
        continue

# =============================
# QUADRANT LINES
# =============================
fig.add_vline(x=100, line_dash="dash", line_color="#888")
fig.add_hline(y=100, line_dash="dash", line_color="#888")

# =============================
# QUADRANT TEXT LABELS
# =============================
fig.add_annotation(x=112, y=112, text="Leading", showarrow=False)
fig.add_annotation(x=88, y=112, text="Improving", showarrow=False)
fig.add_annotation(x=88, y=88, text="Lagging", showarrow=False)
fig.add_annotation(x=112, y=88, text="Weakening", showarrow=False)

fig.update_layout(
    height=760,
    xaxis=dict(range=[85,115], title="RS Ratio"),
    yaxis=dict(range=[85,115], title="RS Momentum"),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
)

st.plotly_chart(
    fig,
    width="stretch",
    config={
        "displaylogo": False,
        "doubleClick": "reset",
    },
)

st.caption(f"✅ Active sectors plotted: {valid_count}")

# =============================
# SECTOR RANKING TABLE
# =============================
if ranking_rows:
    rank_df = pd.DataFrame(ranking_rows)
    rank_df = rank_df.sort_values("RS_Ratio", ascending=False)
    
    st.subheader("📊 Sector Strength Ranking")
    st.dataframe(rank_df, use_container_width=True)


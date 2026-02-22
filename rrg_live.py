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
@st.cache_data(ttl=300)  # refresh every 5 minutes
def fetch(symbol, period, interval):
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

bench = fetch(benchmark, period, interval)
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
              fillcolor="rgba(0,200,0,0.05)", line_width=0)  # Leading

fig.add_shape(type="rect", x0=85, y0=100, x1=100, y1=115,
              fillcolor="rgba(0,0,255,0.05)", line_width=0)  # Improving

fig.add_shape(type="rect", x0=85, y0=85, x1=100, y1=100,
              fillcolor="rgba(255,0,0,0.05)", line_width=0)  # Lagging

fig.add_shape(type="rect", x0=100, y0=85, x1=115, y1=100,
              fillcolor="rgba(255,165,0,0.05)", line_width=0)  # Weakening

ranking_rows = []
valid_count = 0
emerging_leaders = []

for name, symbol in sectors.items():
    try:
        sec = fetch(symbol, period, interval)

        if sec is None:
            continue

        df = pd.concat([sec, bench], axis=1)
        df.columns = ["sec", "bench"]
        df = df.dropna()


        # =============================
        # MIN HISTORY REQUIREMENT (adaptive)
        # =============================
        min_required = 15
        
        if timeframe == "Monthly":
            min_required = 8
        elif timeframe == "Weekly":
            min_required = 12
        else:  # Daily
            min_required = 30
        
        if len(df) < min_required:
            continue


        # Relative strength
        df["RS"] = df["sec"] / df["bench"]
        
        # =============================
        # ADAPTIVE JdK SMOOTHING (PRO FIX)
        # =============================
        if timeframe == "Monthly":
            rs_window = 3
            mom_window = 3
        elif timeframe == "Weekly":
            rs_window = 6
            mom_window = 6
        else:  # Daily
            rs_window = 10
            mom_window = 10
        
        # --- JdK-style RS-Ratio (EMA based) ---
        rs_ema = df["RS"].ewm(span=rs_window, adjust=False).mean()
        df["RS_ratio"] = (df["RS"] / rs_ema) * 100
        
        # --- JdK-style RS-Momentum ---
        mom_ema = df["RS_ratio"].ewm(span=mom_window, adjust=False).mean()
        df["RS_mom"] = (df["RS_ratio"] / mom_ema) * 100

        # =============================
        # ROTATION VELOCITY (PRO EDGE)
        # =============================
        df["velocity"] = df["RS_ratio"].diff()

        hist = df.dropna()

        # =============================
        # DATE-ALIGNED SAFE POSITION
        # =============================
        if selected_date in hist.index:
            safe_loc = hist.index.get_loc(selected_date)
        else:
            safe_loc = hist.index.get_indexer([selected_date], method="nearest")[0]
        
        start_idx = max(0, safe_loc - playback)
        tail = hist.iloc[start_idx:safe_loc + 1]

        # =============================
        # TAIL DOWNSAMPLING (PRO EVEN SPACING)
        # =============================
        max_tail_points = min(playback, 12)  # 🔥 was 4
        
        if len(tail) > max_tail_points:
            import numpy as np
        
            idx = np.linspace(0, len(tail) - 1, max_tail_points, dtype=int)
            tail = tail.iloc[idx]

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

        # =============================
        # EMERGING LEADER DETECTION
        # =============================
        prev_rs = tail["RS_ratio"].iloc[-2] if len(tail) >= 2 else last_rs
        prev_mom = tail["RS_mom"].iloc[-2] if len(tail) >= 2 else last_mom
        
        # =============================
        # EMERGING LEADER DETECTION (PRO)
        # =============================
        was_improving = (prev_rs < 100 and prev_mom > 100)
        now_leading = (last_rs > 100 and last_mom > 100)
        
        is_emerging = was_improving and now_leading

        
        if is_emerging:
            emerging_leaders.append(name)


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
            opacity=0.20,
            showlegend=False,
        ))
        
        # current point index
        sizes = [5] * len(tail)
        sizes[-1] = 20  # 🔥 bigger current point
        
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
                line=dict(
                    width=3 if name in emerging_leaders else 2,
                    color="gold" if name in emerging_leaders else "white"
                ),
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

# =============================
# DYNAMIC AXIS RANGE (PRO)
# =============================
all_x = []
all_y = []

for trace in fig.data:
    if hasattr(trace, "x") and hasattr(trace, "y"):
        all_x.extend(trace.x)
        all_y.extend(trace.y)

if all_x and all_y:
    import numpy as np
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)

    padding = 2
    x_range = [min(85, xmin - padding), max(115, xmax + padding)]
    y_range = [min(85, ymin - padding), max(115, ymax + padding)]
else:
    x_range = [85, 115]
    y_range = [85, 115]

fig.update_layout(
    height=760,
    xaxis=dict(range=x_range, title="RS Ratio"),
    yaxis=dict(range=y_range, title="RS Momentum"),
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

# =============================
# EMERGING LEADERS PANEL
# =============================
if emerging_leaders:
    st.subheader("🚀 Emerging Leaders")
    st.success(", ".join(emerging_leaders))

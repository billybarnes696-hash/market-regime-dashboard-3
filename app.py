import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="SPY Regime + Internals (Cash/ETF)", layout="wide", page_icon="🔄")

st.markdown(
    """
<style>
    .signal-spy {background: linear-gradient(135deg, #10b981, #059669); color: white; padding: 0.85rem; border-radius: 10px; font-weight: 800; text-align: center; font-size: 1.25rem;}
    .signal-def {background: linear-gradient(135deg, #dc2626, #b91c1c); color: white; padding: 0.85rem; border-radius: 10px; font-weight: 800; text-align: center; font-size: 1.25rem;}
    .signal-hold {background: linear-gradient(135deg, #f59e0b, #d97706); color: white; padding: 0.85rem; border-radius: 10px; font-weight: 800; text-align: center; font-size: 1.25rem;}
    .subtle {opacity: 0.9; font-size: 0.95rem; font-weight: 500;}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# HELPERS
# ============================================================
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())

def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a / b.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)

def rolling_z(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window, min_periods=max(30, window // 4)).mean()
    sd = s.rolling(window, min_periods=max(30, window // 4)).std()
    z = (s - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan)

@st.cache_data(ttl=3600)
def fetch_adj_close(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    px = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if len(tickers) == 1:
        close = px["Close"].to_frame(tickers[0])
    else:
        close = px["Close"].copy()

    close.index = pd.to_datetime(close.index)
    close = close.sort_index()
    close = close.dropna(how="all")
    return close

# ============================================================
# CORE: SPY TRIGGER
# ============================================================
def spy_trigger(spy: pd.Series, fast: int, slow: int, smooth: int) -> pd.Series:
    fe = ema(spy, fast)
    se = ema(spy, slow)
    line = (fe - se) / se * 100.0
    line = line.replace([np.inf, -np.inf], np.nan)
    line = ema(line, smooth)
    return line

def make_osc_look(line: pd.Series, look_window: int, visual_range: float, sensitivity: float) -> pd.Series:
    std = line.rolling(look_window, min_periods=max(30, look_window // 3)).std()
    z = line / std.replace(0, np.nan)
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pd.Series(visual_range * np.tanh(z / sensitivity), index=line.index)

# ============================================================
# INTERNALS: HEALTH SCORE (SIZING ONLY)
# ============================================================
def build_health_score(closes: pd.DataFrame, zwin: int) -> pd.Series:
    df = pd.DataFrame(index=closes.index)

    df["spy_vxx"] = safe_div(closes["SPY"], closes["VXX"]) if "VXX" in closes.columns else np.nan
    df["hyg_lqd"] = safe_div(closes["HYG"], closes["LQD"]) if ("HYG" in closes.columns and "LQD" in closes.columns) else np.nan
    df["rsp_spy"] = safe_div(closes["RSP"], closes["SPY"]) if "RSP" in closes.columns else np.nan
    df["xlf_spy"] = safe_div(closes["XLF"], closes["SPY"]) if "XLF" in closes.columns else np.nan

    for c in ["spy_vxx", "hyg_lqd", "rsp_spy", "xlf_spy"]:
        df[c + "_roc20"] = df[c].pct_change(20) * 100.0

    zs = []
    for c in ["spy_vxx_roc20", "hyg_lqd_roc20", "rsp_spy_roc20", "xlf_spy_roc20"]:
        zs.append(rolling_z(df[c], zwin))

    hs = pd.concat(zs, axis=1).mean(axis=1)
    hs = ema(hs, 5)
    return hs

# ============================================================
# BACKTEST
# ============================================================
def backtest(
    closes: pd.DataFrame,
    defensive_ticker: str,
    line: pd.Series,
    deadband: float,
    trend_filter_ma: int,
    use_health_sizing: bool,
    health_score: pd.Series,
    health_threshold: float,
    weak_weight: float,
    trade_cost_bps: float,
    look_window: int,
    visual_range: float,
    sensitivity: float,
) -> pd.DataFrame:
    """
    Signal logic:
      - if line > +deadband and trend_ok => SPY
      - if line < -deadband             => DEF
      - else                            => HOLD prior
    Execution:
      - Held = Signal.shift(1) (no lookahead)
    """
    df = pd.DataFrame(index=closes.index).copy()
    df["SPY"] = closes["SPY"]
    df["DEF"] = closes[defensive_ticker]
    df = df.dropna(subset=["SPY", "DEF"])

    df["spy_ma"] = df["SPY"].rolling(trend_filter_ma, min_periods=trend_filter_ma).mean()
    df["trend_ok"] = df["SPY"] > df["spy_ma"]

    df["line"] = line.reindex(df.index)
    df = df.dropna(subset=["line", "spy_ma"])

    sig = pd.Series(index=df.index, dtype="object")
    sig.iloc[0] = "DEF"

    for i in range(1, len(df)):
        prev = sig.iloc[i - 1]
        v = float(df["line"].iloc[i])
        trend_ok = bool(df["trend_ok"].iloc[i])

        if v > deadband and trend_ok:
            sig.iloc[i] = "SPY"
        elif v < -deadband:
            sig.iloc[i] = "DEF"
        else:
            sig.iloc[i] = prev

    df["Signal"] = sig
    df["Held"] = df["Signal"].shift(1)
    df = df.dropna(subset=["Held"])

    df["spy_ret"] = df["SPY"].pct_change().fillna(0.0)
    df["def_ret"] = df["DEF"].pct_change().fillna(0.0)

    if use_health_sizing:
        hs = health_score.reindex(df.index)
        df["health"] = hs
        df["spy_w"] = np.where(
            df["Held"] == "SPY",
            np.where(df["health"] < health_threshold, weak_weight, 1.0),
            0.0,
        )
    else:
        df["health"] = np.nan
        df["spy_w"] = np.where(df["Held"] == "SPY", 1.0, 0.0)

    df["def_w"] = 1.0 - df["spy_w"]
    df["gross_ret"] = df["spy_w"] * df["spy_ret"] + df["def_w"] * df["def_ret"]

    df["turnover"] = (df["Held"] != df["Held"].shift(1)).fillna(False).astype(int)
    cost = (trade_cost_bps / 10000.0) * df["turnover"]
    df["net_ret"] = df["gross_ret"] - cost

    df["strat_cum"] = (1.0 + df["net_ret"]).cumprod() * 100.0
    df["buyhold_cum"] = (1.0 + df["spy_ret"]).cumprod() * 100.0
    df["def_cum"] = (1.0 + df["def_ret"]).cumprod() * 100.0

    df["Osc"] = make_osc_look(df["line"], look_window=look_window, visual_range=visual_range, sensitivity=sensitivity)
    df["CrossUp"] = (df["Osc"] > 0) & (df["Osc"].shift(1) <= 0)
    df["CrossDn"] = (df["Osc"] < 0) & (df["Osc"].shift(1) >= 0)

    return df

# ============================================================
# PLOT
# ============================================================
def plot_dashboard(df: pd.DataFrame, defensive_label: str) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20], vertical_spacing=0.05
    )

    fig.add_trace(go.Scatter(x=df.index, y=df["SPY"], name="SPY", line=dict(width=2)), row=1, col=1)

    hold = df["Held"]
    changes = (hold != hold.shift(1)).fillna(True)
    starts = list(df.index[changes])
    starts.append(df.index[-1])

    for i in range(len(starts) - 1):
        s, e = starts[i], starts[i + 1]
        col = "rgba(16,185,129,0.12)" if hold.loc[s] == "SPY" else "rgba(220,38,38,0.12)"
        fig.add_vrect(x0=s, x1=e, fillcolor=col, opacity=0.5, layer="below", line_width=0)

    fig.add_trace(go.Bar(x=df.index, y=df["Osc"], name="Signal (osc)", opacity=0.85), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Osc"], name="Osc line", line=dict(width=2)), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    ups = df[df["CrossUp"]]
    dns = df[df["CrossDn"]]
    if not ups.empty:
        fig.add_trace(
            go.Scatter(x=ups.index, y=ups["Osc"], mode="markers", name="Cross↑",
                       marker=dict(size=9, symbol="triangle-up")),
            row=2, col=1
        )
    if not dns.empty:
        fig.add_trace(
            go.Scatter(x=dns.index, y=dns["Osc"], mode="markers", name=f"Cross↓→{defensive_label}",
                       marker=dict(size=9, symbol="triangle-down")),
            row=2, col=1
        )

    fig.add_trace(go.Scatter(x=df.index, y=df["strat_cum"], name="Strategy", line=dict(width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["buyhold_cum"], name="SPY Buy/Hold", line=dict(width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["def_cum"], name=f"{defensive_label} Buy/Hold", line=dict(width=1.5)), row=3, col=1)

    fig.update_layout(
        height=860, hovermode="x unified", bargap=0,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title="SPY Regime (Trigger) + Defensive (Cash/ETF) + Internals (Sizing)"
    )
    fig.update_yaxes(title_text="SPY Price", row=1, col=1)
    fig.update_yaxes(title_text="Osc (0-line)", row=2, col=1)
    fig.update_yaxes(title_text="Equity (Start=100)", row=3, col=1)
    return fig

# ============================================================
# APP
# ============================================================
def main():
    st.title("🔄 Efficient SPY Regime + Cash/ETF Defensive")

    with st.sidebar:
        st.header("⚙️ Backtest")

        years = st.slider("History (years)", 1, 20, 15)

        st.subheader("SPY Trigger (simple)")
        fast = st.slider("Fast EMA", 2, 30, 5)
        slow = st.slider("Slow EMA", 5, 80, 13)
        smooth = st.slider("Signal smoothing (EMA)", 1, 30, 5)

        st.subheader("Whipsaw Control")
        deadband = st.slider("Deadband around 0", 0.0, 1.5, 0.25, 0.01)
        trend_ma = st.slider("Trend filter MA (days)", 50, 300, 200, 10)

        st.subheader("Defensive")
        use_etf = st.toggle("Use defensive ETF (instead of cash proxy)", value=False)
        def_ticker = st.text_input("Defensive ticker (AGG / GLD / SHY / etc)", value="AGG").strip().upper()
        cash_proxy = st.selectbox("Cash proxy ticker", ["BIL", "SHY", "SGOV"], index=0)
        defensive_ticker = def_ticker if use_etf else cash_proxy
        defensive_label = defensive_ticker

        st.subheader("Internals (sizing only)")
        use_health = st.toggle("Reduce SPY size when internals weak", value=True)
        zwin = st.slider("Health z-score window", 60, 504, 252, 21)
        health_th = st.slider("Health threshold (below = weak)", -2.0, 2.0, -0.25, 0.05)
        weak_w = st.slider("SPY weight when weak", 0.0, 1.0, 0.5, 0.05)

        st.subheader("Realism")
        trade_cost_bps = st.slider("Trading cost (bps per switch)", 0.0, 20.0, 2.0, 0.5)

        st.subheader("Oscillator LOOK (visual only)")
        look_window = st.slider("Look window (rolling std days)", 30, 252, 126, 7)
        visual_range = st.slider("Visual range (+/-)", 1.0, 8.0, 3.0, 0.5)
        sensitivity = st.slider("Sensitivity (lower=more swing)", 0.5, 3.0, 1.5, 0.1)

        if st.button("🔄 Refresh / clear cache", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.caption("⚠️ Not investment advice.")

    end = datetime.now().date() + timedelta(days=1)
    start = (datetime.now() - timedelta(days=int(365.25 * years))).date()

    tickers = sorted(list(set([
        "SPY",
        defensive_ticker,
        "VXX", "HYG", "LQD", "RSP", "XLF"
    ])))

    try:
        closes = fetch_adj_close(tickers, start=str(start), end=str(end))

        if "SPY" not in closes.columns or defensive_ticker not in closes.columns:
            st.error("Missing required data (SPY or defensive ticker). Try another defensive ticker.")
            return

        line = spy_trigger(closes["SPY"], fast=fast, slow=slow, smooth=smooth)
        hs = build_health_score(closes, zwin=zwin) if use_health else pd.Series(index=closes.index, dtype=float)

        df = backtest(
            closes=closes,
            defensive_ticker=defensive_ticker,
            line=line,
            deadband=deadband,
            trend_filter_ma=trend_ma,
            use_health_sizing=use_health,
            health_score=hs,
            health_threshold=health_th,
            weak_weight=weak_w,
            trade_cost_bps=trade_cost_bps,
            look_window=look_window,
            visual_range=visual_range,
            sensitivity=sensitivity,
        )
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

    tab1, tab2 = st.tabs(["🎯 Current", "📊 Backtest"])

    with tab1:
        last = df.iloc[-1]

        signal_today = last["Signal"]
        held_today = last["Held"]
        inside_deadband = (abs(float(last["line"])) <= deadband)

        if inside_deadband:
            st.markdown(
                f"""<div class="signal-hold">🟡 HOLD ({held_today})
                <div class="subtle">line={float(last["line"]):+.3f} • deadband=±{deadband:.2f} • signal_today={signal_today} • held_today={held_today}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        elif signal_today == "SPY":
            st.markdown(
                f"""<div class="signal-spy">🟢 SIGNAL → SPY
                <div class="subtle">line={float(last["line"]):+.3f} • deadband=±{deadband:.2f} • held_today={held_today}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""<div class="signal-def">🔴 SIGNAL → {defensive_label}
                <div class="subtle">line={float(last["line"]):+.3f} • deadband=±{deadband:.2f} • held_today={held_today}</div>
                </div>""",
                unsafe_allow_html=True,
            )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("SPY", f"{last['SPY']:.2f}")
        with c2:
            st.metric(defensive_label, f"{last['DEF']:.2f}")
        with c3:
            st.metric("Held today (no lookahead)", held_today)
        with c4:
            if pd.notna(last.get("health", np.nan)):
                st.metric("Health", f"{float(last['health']):+.2f}")
            else:
                st.metric("Health", "—")

    with tab2:
        fig = plot_dashboard(df, defensive_label=defensive_label)
        st.plotly_chart(fig, use_container_width=True)

        strat_total = float(df["strat_cum"].iloc[-1] - 100.0)
        bh_total = float(df["buyhold_cum"].iloc[-1] - 100.0)
        mdd_strat = max_drawdown(df["strat_cum"])
        mdd_bh = max_drawdown(df["buyhold_cum"])
        rotations = int((df["Held"] != df["Held"].shift(1)).fillna(False).sum())

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Strategy Return", f"{strat_total:+.1f}%", delta=f"{(strat_total - bh_total):+.1f}% vs SPY")
        with col2:
            st.metric("SPY Buy/Hold", f"{bh_total:+.1f}%")
        with col3:
            st.metric("Max Drawdown", f"{mdd_strat:.1%}", delta=f"SPY {mdd_bh:.1%}")
        with col4:
            st.metric("Rotations", f"{rotations}")

        with st.expander("Recent rows (last 6 months)"):
            cols = ["SPY", "DEF", "line", "Osc", "Signal", "Held", "spy_w", "gross_ret", "net_ret", "strat_cum", "buyhold_cum", "turnover"]
            cols = [c for c in cols if c in df.columns]

            recent_start = df.index.max() - pd.DateOffset(months=6)
            recent_df = df.loc[df.index >= recent_start, cols]

            st.dataframe(recent_df, use_container_width=True)

if __name__ == "__main__":
    main()

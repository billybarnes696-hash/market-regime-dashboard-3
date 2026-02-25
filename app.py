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
# CORE: SPY TRIGGER (simple, effective)
# ============================================================
def spy_trigger(spy: pd.Series, fast: int, slow: int, smooth: int) -> pd.Series:
    """
    PPO-ish line: (EMAfast - EMAslow) / EMAslow * 100
    """
    fe = ema(spy, fast)
    se = ema(spy, slow)
    line = (fe - se) / se * 100.0
    line = line.replace([np.inf, -np.inf], np.nan)
    line = ema(line, smooth)
    return line

def make_osc_look(line: pd.Series, look_window: int, visual_range: float, sensitivity: float) -> pd.Series:
    """
    Purely visual. Keeps the oscillator in a stable +/- band.
    """
    std = line.rolling(look_window, min_periods=max(30, look_window // 3)).std()
    z = line / std.replace(0, np.nan)
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pd.Series(visual_range * np.tanh(z / sensitivity), index=line.index)

# ============================================================
# INTERNALS: CONTEXT (NOT TRIGGERS)
# ============================================================
def build_health_score(closes: pd.DataFrame, zwin: int) -> pd.Series:
    """
    Health score (higher = healthier):
      - SPY:VXX roc20 (vol suppression / fear)
      - HYG:LQD roc20 (credit appetite)
      - RSP:SPY roc20 (participation breadth proxy)
      - XLF:SPY roc20 (leadership/stress proxy)
    """
    df = pd.DataFrame(index=closes.index)

    if "VXX" in closes.columns:
        df["spy_vxx"] = safe_div(closes["SPY"], closes["VXX"])
    else:
        df["spy_vxx"] = np.nan

    if "HYG" in closes.columns and "LQD" in closes.columns:
        df["hyg_lqd"] = safe_div(closes["HYG"], closes["LQD"])
    else:
        df["hyg_lqd"] = np.nan

    if "RSP" in closes.columns:
        df["rsp_spy"] = safe_div(closes["RSP"], closes["SPY"])
    else:
        df["rsp_spy"] = np.nan

    if "XLF" in closes.columns:
        df["xlf_spy"] = safe_div(closes["XLF"], closes["SPY"])
    else:
        df["xlf_spy"] = np.nan

    for c in ["spy_vxx", "hyg_lqd", "rsp_spy", "xlf_spy"]:
        df[c + "_roc20"] = df[c].pct_change(20) * 100.0

    zs = []
    for c in ["spy_vxx_roc20", "hyg_lqd_roc20", "rsp_spy_roc20", "xlf_spy_roc20"]:
        zs.append(rolling_z(df[c], zwin))

    hs = pd.concat(zs, axis=1).mean(axis=1)
    hs = ema(hs, 5)  # smooth
    return hs

# ============================================================
# BACKTEST ENGINE
# ============================================================
def backtest(
    closes: pd.DataFrame,
    defensive_ticker: str,
    use_defensive_etf: bool,
    line: pd.Series,
    deadband: float,
    trend_filter_ma: int,
    use_health_sizing: bool,
    health_score: pd.Series,
    health_threshold: float,
    weak_weight: float,
    trade_cost_bps: float,
) -> pd.DataFrame:
    """
    Strategy:
      - Base signal: line > +deadband => SPY
                    line < -deadband => DEF
                    else => HOLD PRIOR (reduces whipsaw)
      - Optional trend filter: only allow SPY if SPY > MA(trend_filter_ma)
      - Defensive:
          - if use_defensive_etf: DEF = defensive_ticker (e.g., AGG/GLD/SHY/BIL)
          - else: use BIL as "cash proxy" (must exist in closes)
      - Execution: signal at t applies to returns at t+1 (lag)
      - Optional internals sizing: when in SPY, if health_score < threshold => reduce SPY weight to weak_weight
      - Trading cost: cost applied on days where position changes (bps)
    """
    df = pd.DataFrame(index=closes.index).copy()
    df["SPY"] = closes["SPY"]

    # defensive series
    df["DEF"] = closes[defensive_ticker]  # defensive_ticker is already set to BIL if "cash proxy"
    df = df.dropna(subset=["SPY", "DEF"])

    # trend filter
    df["spy_ma"] = df["SPY"].rolling(trend_filter_ma, min_periods=trend_filter_ma).mean()
    df["trend_ok"] = df["SPY"] > df["spy_ma"]

    # raw line
    df["line"] = line.reindex(df.index)
    df = df.dropna(subset=["line", "spy_ma"])

    # build stateful signal with deadband (hold prior inside band)
    sig = pd.Series(index=df.index, dtype="object")
    sig.iloc[0] = "DEF"  # start defensive until proven otherwise

    for i in range(1, len(df)):
        prev = sig.iloc[i - 1]
        v = df["line"].iloc[i]
        trend_ok = bool(df["trend_ok"].iloc[i])

        if v > deadband and trend_ok:
            sig.iloc[i] = "SPY"
        elif v < -deadband:
            sig.iloc[i] = "DEF"
        else:
            sig.iloc[i] = prev

    df["Signal"] = sig

    # lag execution (no lookahead)
    df["Held"] = df["Signal"].shift(1)
    df = df.dropna(subset=["Held"])

    # returns
    df["spy_ret"] = df["SPY"].pct_change().fillna(0.0)
    df["def_ret"] = df["DEF"].pct_change().fillna(0.0)

    # weights
    if use_health_sizing:
        hs = health_score.reindex(df.index)
        df["health"] = hs
        df["spy_w"] = np.where(df["Held"] == "SPY",
                               np.where(df["health"] < health_threshold, weak_weight, 1.0),
                               0.0)
    else:
        df["health"] = np.nan
        df["spy_w"] = np.where(df["Held"] == "SPY", 1.0, 0.0)

    df["def_w"] = 1.0 - df["spy_w"]

    # gross strategy return
    df["gross_ret"] = df["spy_w"] * df["spy_ret"] + df["def_w"] * df["def_ret"]

    # trading cost on position changes (bps => fraction)
    # cost triggers when Held changes vs prior Held
    df["turnover"] = (df["Held"] != df["Held"].shift(1)).fillna(False).astype(int)
    cost = (trade_cost_bps / 10000.0) * df["turnover"]
    df["net_ret"] = df["gross_ret"] - cost

    # equity curves
    df["strat_cum"] = (1.0 + df["net_ret"]).cumprod() * 100.0
    df["buyhold_cum"] = (1.0 + df["spy_ret"]).cumprod() * 100.0
    df["def_cum"] = (1.0 + df["def_ret"]).cumprod() * 100.0

    # oscillator display and cross markers (for visualization only)
    df["Osc"] = make_osc_look(df["line"], look_window=126, visual_range=3.0, sensitivity=1.5)
    df["CrossUp"] = (df["Osc"] > 0) & (df["Osc"].shift(1) <= 0)
    df["CrossDn"] = (df["Osc"] < 0) & (df["Osc"].shift(1) >= 0)

    return df

# ============================================================
# PLOTS
# ============================================================
def plot_dashboard(df: pd.DataFrame, defensive_label: str) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20], vertical_spacing=0.05
    )

    # price
    fig.add_trace(go.Scatter(x=df.index, y=df["SPY"], name="SPY", line=dict(width=2)), row=1, col=1)

    # shade regimes by Held
    hold = df["Held"]
    changes = (hold != hold.shift(1)).fillna(True)
    starts = list(df.index[changes])
    starts.append(df.index[-1])

    for i in range(len(starts) - 1):
        s, e = starts[i], starts[i + 1]
        col = "rgba(16,185,129,0.12)" if hold.loc[s] == "SPY" else "rgba(220,38,38,0.12)"
        fig.add_vrect(x0=s, x1=e, fillcolor=col, opacity=0.5, layer="below", line_width=0)

    # oscillator pane
    fig.add_trace(go.Bar(x=df.index, y=df["Osc"], name="Signal (osc)", opacity=0.85), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Osc"], name="Osc line", line=dict(width=2)), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    ups = df[df["CrossUp"]]
    dns = df[df["CrossDn"]]
    if not ups.empty:
        fig.add_trace(go.Scatter(x=ups.index, y=ups["Osc"], mode="markers", name="Cross↑",
                                 marker=dict(size=9, symbol="triangle-up")), row=2, col=1)
    if not dns.empty:
        fig.add_trace(go.Scatter(x=dns.index, y=dns["Osc"], mode="markers", name=f"Cross↓→{defensive_label}",
                                 marker=dict(size=9, symbol="triangle-down")), row=2, col=1)

    # equity curves
    fig.add_trace(go.Scatter(x=df.index, y=df["strat_cum"], name="Strategy", line=dict(width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["buyhold_cum"], name="SPY Buy/Hold", line=dict(width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["def_cum"], name=f"{defensive_label} Buy/Hold", line=dict(width=1.5)),
                  row=3, col=1)

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
    st.title("🔄 Efficient SPY Regime + Cash/ETF Defensive (15Y)")

    with st.sidebar:
        st.header("⚙️ Backtest")
        years = st.slider("History (years)", 10, 20, 15)

        st.subheader("SPY Trigger (simple)")
        fast = st.slider("Fast EMA", 2, 30, 5)
        slow = st.slider("Slow EMA", 5, 80, 13)
        smooth = st.slider("Signal smoothing (EMA)", 1, 30, 5)

        st.subheader("Whipsaw Control (big deal)")
        deadband = st.slider("Deadband around 0", 0.0, 1.5, 0.25, 0.05,
                             help="Inside +/- deadband the system holds prior position. Reduces churn.")
        trend_ma = st.slider("Trend filter MA (days)", 50, 300, 200, 10,
                             help="Only allow SPY when SPY > MA. Helps avoid bear-market chop.")

        st.subheader("Defensive")
        use_etf = st.toggle("Use defensive ETF (instead of cash proxy)", value=False)
        def_ticker = st.text_input("Defensive ticker (AGG / GLD / SHY / etc)", value="AGG").strip().upper()

        # cash proxy = BIL by default (real series; better than fake interest rate)
        cash_proxy = st.selectbox("Cash proxy ticker", ["BIL", "SHY", "SGOV"], index=0)

        st.subheader("Internals (sizing only)")
        use_health = st.toggle("Reduce SPY size when internals weak", value=True)
        zwin = st.slider("Health z-score window", 60, 504, 252, 21)
        health_th = st.slider("Health threshold (below = weak)", -2.0, 2.0, -0.25, 0.05)
        weak_w = st.slider("SPY weight when weak", 0.0, 1.0, 0.5, 0.05)

        st.subheader("Realism")
        trade_cost_bps = st.slider("Trading cost (bps per switch)", 0.0, 20.0, 2.0, 0.5,
                                   help="Applied each time Held position changes (lagged).")

        if st.button("🔄 Refresh / clear cache", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.caption("⚠️ Not investment advice.")

    end = datetime.now().date() + timedelta(days=1)
    start = (datetime.now() - timedelta(days=int(365.25 * years))).date()

    # Determine defensive ticker used in backtest
    defensive_ticker = def_ticker if use_etf else cash_proxy
    defensive_label = defensive_ticker

    tickers = sorted(list(set([
        "SPY",
        defensive_ticker,
        "VXX", "HYG", "LQD", "RSP", "XLF"
    ])))

    try:
        closes = fetch_adj_close(tickers, start=str(start), end=str(end))
        if "SPY" not in closes.columns or defensive_ticker not in closes.columns:
            st.error("Missing required data from yfinance (SPY or defensive ticker). Try another defensive ticker.")
            return

        line = spy_trigger(closes["SPY"], fast=fast, slow=slow, smooth=smooth)
        hs = build_health_score(closes, zwin=zwin) if use_health else None

        df = backtest(
            closes=closes,
            defensive_ticker=defensive_ticker,
            use_defensive_etf=use_etf,
            line=line,
            deadband=deadband,
            trend_filter_ma=trend_ma,
            use_health_sizing=use_health,
            health_score=hs if hs is not None else pd.Series(index=closes.index, dtype=float),
            health_threshold=health_th,
            weak_weight=weak_w,
            trade_cost_bps=trade_cost_bps,
        )

    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

    tab1, tab2 = st.tabs(["🎯 Current", "📊 Backtest"])

    with tab1:
        last = df.iloc[-1]
        today_signal = "SPY" if last["line"] > deadband and bool((last["SPY"] > last["spy_ma"])) else defensive_label
        held = last["Held"]

        if today_signal == "SPY":
            st.markdown(
                f"""<div class="signal-spy">🟢 SPY ON
                <div class="subtle">line={float(last["line"]):+.3f} • deadband=±{deadband:.2f} • held={held}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""<div class="signal-def">🔴 DEFENSIVE → {defensive_label}
                <div class="subtle">line={float(last["line"]):+.3f} • deadband=±{deadband:.2f} • held={held}</div>
                </div>""",
                unsafe_allow_html=True,
            )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("SPY", f"{last['SPY']:.2f}")
        with c2:
            st.metric(defensive_label, f"{last['DEF']:.2f}")
        with c3:
            st.metric("Held (no lookahead)", held)
        with c4:
            if "health" in df.columns and pd.notna(last.get("health", np.nan)):
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

        with st.expander("Recent rows"):
            cols = ["SPY", "DEF", "line", "Osc", "Held", "spy_w", "gross_ret", "net_ret", "strat_cum", "buyhold_cum"]
            cols = [c for c in cols if c in df.columns]
            st.dataframe(df[cols].tail(40), use_container_width=True)

if __name__ == "__main__":
    main()

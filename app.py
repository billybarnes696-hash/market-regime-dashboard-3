import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =============================
# Helpers: parse uploaded files
# =============================
OHLC_PATTERN = re.compile(
    r"(?P<day>Mon|Tue|Wed|Thu|Fri)\s+"
    r"(?P<date>\d{2}-\d{2}-\d{4})\s+"
    r"(?P<open>-?\d+(?:\.\d+)?)\s+"
    r"(?P<high>-?\d+(?:\.\d+)?)\s+"
    r"(?P<low>-?\d+(?:\.\d+)?)\s+"
    r"(?P<close>-?\d+(?:\.\d+)?)\s+"
    r"(?P<volume>-?\d+(?:\.\d+)?)"
)


def normalize_symbol(sym: str) -> str:
    return sym.strip()


def parse_historical_raw(raw_csv_path: str) -> pd.DataFrame:
    """Parse the 1-column StockCharts export where each row is a full symbol history."""
    raw = pd.read_csv(raw_csv_path, header=None)
    records: List[Dict] = []

    for cell in raw.iloc[:, 0].dropna().astype(str):
        symbol_match = re.match(r"\s*([^,]+),", cell)
        if not symbol_match:
            continue
        symbol = normalize_symbol(symbol_match.group(1))

        for m in OHLC_PATTERN.finditer(cell):
            dt = pd.to_datetime(m.group("date"), format="%m-%d-%Y", errors="coerce")
            if pd.isna(dt):
                continue
            records.append(
                {
                    "date": dt,
                    "symbol": symbol,
                    "open": float(m.group("open")),
                    "high": float(m.group("high")),
                    "low": float(m.group("low")),
                    "close": float(m.group("close")),
                    "volume": float(m.group("volume")),
                }
            )

    df = pd.DataFrame(records).sort_values(["symbol", "date"]).reset_index(drop=True)
    return df


def load_realtime_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Symbol"] = df["Symbol"].astype(str).map(normalize_symbol)
    return df


# =============================
# Helpers: indicators
# =============================

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1 / period, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    c = (tp - sma) / (0.015 * mad.replace(0, np.nan))
    return c


def true_strength_index(series: pd.Series, long: int = 25, short: int = 13, signal: int = 7) -> Tuple[pd.Series, pd.Series]:
    m = series.diff()
    a = m.abs()
    m1 = ema(ema(m, long), short)
    a1 = ema(ema(a, long), short)
    tsi = 100 * (m1 / a1.replace(0, np.nan))
    sig = ema(tsi, signal)
    return tsi, sig


def percent_b(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    pb = (series - lower) / (upper - lower).replace(0, np.nan)
    return pb


def roc(series: pd.Series, period: int = 3) -> pd.Series:
    return (series / series.shift(period) - 1) * 100


# =============================
# Breadth engine logic
# =============================
CORE_SYMBOLS = [
    "$BPNYA",
    "$BPSPX",
    "$CPCE",
    "$NYAD",
    "$NYHL",
    "$NYMO",
    "$NYSI",
    "$OEXA150R",
    "$OEXA200R",
    "$OEXA50R",
    "$SPX",
    "$SPXA50R",
    "$SPXADP",
    "HYG:IEF",
    "HYG:TLT",
    "IWM:SPY",
    "RSP",
    "RSP:SPY",
    "SMH:SPY",
    "SPXS:SVOL",
    "URSP",
    "VXX",
    "XLF:SPY",
]


def wide_from_long(hist: pd.DataFrame) -> pd.DataFrame:
    wide = hist.pivot(index="date", columns="symbol", values="close").sort_index()
    return wide


def add_indicator_features(hist: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for sym, g in hist.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()
        g["rsi14"] = rsi(g["close"], 14)
        g["cci20"] = cci(g["high"], g["low"], g["close"], 20)
        g["pct_b20"] = percent_b(g["close"], 20, 2.0)
        g["roc3"] = roc(g["close"], 3)
        g["roc5"] = roc(g["close"], 5)
        g["tsi_fast"], g["tsi_fast_sig"] = true_strength_index(g["close"], 4, 2, 4)
        g["tsi_slow"], g["tsi_slow_sig"] = true_strength_index(g["close"], 25, 13, 7)
        g["ma20"] = g["close"].rolling(20).mean()
        g["ma50"] = g["close"].rolling(50).mean()
        g["ma200"] = g["close"].rolling(200).mean()
        g["daily_change"] = g["close"].diff()
        g["daily_pct"] = g["close"].pct_change() * 100
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def detect_candle_pattern(row: pd.Series) -> str:
    rng = row["high"] - row["low"]
    if pd.isna(rng) or rng == 0:
        return "n/a"
    body = abs(row["close"] - row["open"])
    upper = row["high"] - max(row["open"], row["close"])
    lower = min(row["open"], row["close"]) - row["low"]
    close_pos = (row["close"] - row["low"]) / rng

    if body / rng < 0.2 and upper / rng > 0.4 and lower / rng < 0.2:
        return "upper-wick rejection"
    if body / rng < 0.2 and lower / rng > 0.4 and upper / rng < 0.2:
        return "lower-wick support"
    if close_pos > 0.8 and body / rng > 0.5:
        return "strong close near high"
    if close_pos < 0.2 and body / rng > 0.5:
        return "weak close near low"
    return "neutral"


def detect_bear_kiss(g: pd.DataFrame, ma_col: str = "ma50") -> pd.Series:
    close = g["close"]
    ma = g[ma_col]
    touched = ((close < ma) & (g["high"] >= ma * 0.995) & (g["close"] < ma))
    return touched.fillna(False)


def detect_bull_kiss(g: pd.DataFrame, ma_col: str = "ma50") -> pd.Series:
    close = g["close"]
    ma = g[ma_col]
    touched = ((close > ma) & (g["low"] <= ma * 1.005) & (g["close"] > ma))
    return touched.fillna(False)


def score_nyad(val: float) -> int:
    if val > 1500:
        return 2
    if val > 300:
        return 1
    if val < -1500:
        return -2
    if val < -300:
        return -1
    return 0


def score_spxadp(val: float) -> int:
    if val > 60:
        return 2
    if val > 20:
        return 1
    if val < -60:
        return -2
    if val < -20:
        return -1
    return 0


def compute_nymo_proxy(nyad: float, spxadp: float, bpspx_dir: int = 0) -> float:
    return 0.5 * score_nyad(nyad) + 0.3 * score_spxadp(spxadp) + 0.2 * bpspx_dir


def derive_bpspx_direction(latest_close: float, prior_close: Optional[float]) -> int:
    if prior_close is None or pd.isna(prior_close):
        return 0
    diff = latest_close - prior_close
    if diff >= 3:
        return 2
    if diff > 0:
        return 1
    if diff <= -3:
        return -2
    if diff < 0:
        return -1
    return 0


def classify_market_state(snapshot: Dict[str, float]) -> str:
    bpspx = snapshot.get("$BPSPX", np.nan)
    spxa50r = snapshot.get("$SPXA50R", np.nan)
    nymo = snapshot.get("$NYMO", np.nan)
    proxy = snapshot.get("NYMO_PROXY", np.nan)

    if pd.notna(nymo) and nymo < -80:
        return "capitulation washout"
    if pd.notna(proxy) and proxy >= 1.5 and bpspx < 45:
        return "breadth thrust / early repair"
    if bpspx >= 45 and spxa50r >= 30 and (pd.isna(nymo) or nymo > -20):
        return "confirmed trend repair"
    if bpspx < 45 and spxa50r < 30:
        return "bearish bounce / incomplete repair"
    return "repair phase"


def bounce_quality(snapshot: Dict[str, float], prev_snapshot: Optional[Dict[str, float]] = None) -> Tuple[int, List[str]]:
    notes = []
    score = 0
    bpspx = snapshot.get("$BPSPX", np.nan)
    spxa50r = snapshot.get("$SPXA50R", np.nan)
    nyad = snapshot.get("$NYAD", np.nan)
    spxadp = snapshot.get("$SPXADP", np.nan)
    vxx = snapshot.get("VXX", np.nan)
    rsp_spy = snapshot.get("RSP:SPY", np.nan)

    if bpspx >= 45:
        score += 2; notes.append("Participation >45")
    elif bpspx >= 40:
        score += 1; notes.append("Participation improving")

    if spxa50r >= 30:
        score += 2; notes.append("Breadth depth >30")
    elif spxa50r >= 25:
        score += 1; notes.append("Depth improving")

    if nyad > 1500 and spxadp > 60:
        score += 2; notes.append("Strong breadth thrust")
    elif nyad > 500 or spxadp > 20:
        score += 1; notes.append("Moderate thrust")

    if prev_snapshot and pd.notna(vxx):
        prev_vxx = prev_snapshot.get("VXX", np.nan)
        if pd.notna(prev_vxx) and vxx < prev_vxx:
            score += 1; notes.append("Volatility easing")

    if prev_snapshot and pd.notna(rsp_spy):
        prev_ratio = prev_snapshot.get("RSP:SPY", np.nan)
        if pd.notna(prev_ratio) and rsp_spy >= prev_ratio:
            score += 1; notes.append("Equal-weight not lagging")

    return min(score, 7), notes


def build_narrative(snapshot: Dict[str, float], prev_snapshot: Optional[Dict[str, float]] = None) -> str:
    state = classify_market_state(snapshot)
    bq, notes = bounce_quality(snapshot, prev_snapshot)
    nymo = snapshot.get("$NYMO", np.nan)
    nysi = snapshot.get("$NYSI", np.nan)
    bpspx = snapshot.get("$BPSPX", np.nan)
    spxa50r = snapshot.get("$SPXA50R", np.nan)
    nyad = snapshot.get("$NYAD", np.nan)
    spxadp = snapshot.get("$SPXADP", np.nan)

    parts = [f"State: {state}."]
    if pd.notna(nymo):
        parts.append(f"Official NYMO is {nymo:.2f}.")
    if pd.notna(nysi):
        parts.append(f"NYSI is {nysi:.2f}.")
    parts.append(f"BPSPX {bpspx:.2f} and SPXA50R {spxa50r:.2f} define participation and depth.")
    if nyad > 1500 and spxadp > 60:
        parts.append("Today qualifies as a major breadth thrust day.")
    parts.append(f"Bounce Quality Score: {bq}/7 ({'; '.join(notes) if notes else 'limited confirmation'}).")
    if bpspx >= 45 and spxa50r >= 30:
        parts.append("Broad repair is close to or inside confirmation territory.")
    elif bpspx < 45 or spxa50r < 30:
        parts.append("Repair is underway, but not fully confirmed for leverage.")
    return " ".join(parts)


def rank_signals(snapshot: Dict[str, float], prev_snapshot: Optional[Dict[str, float]]) -> pd.DataFrame:
    rows = []
    def add(signal, status, detail, score):
        rows.append({"Signal": signal, "Status": status, "Detail": detail, "Score": score})

    bpspx = snapshot.get("$BPSPX", np.nan)
    spxa50r = snapshot.get("$SPXA50R", np.nan)
    nymo = snapshot.get("$NYMO", np.nan)
    nysi = snapshot.get("$NYSI", np.nan)
    nyad = snapshot.get("$NYAD", np.nan)
    spxadp = snapshot.get("$SPXADP", np.nan)
    nyhl = snapshot.get("$NYHL", np.nan)
    ratio = snapshot.get("RSP:SPY", np.nan)

    add("Participation", "PASS" if bpspx >= 45 else "WAIT", f"BPSPX {bpspx:.2f}", 2 if bpspx >= 45 else 0)
    add("Depth repair", "PASS" if spxa50r >= 30 else "WAIT", f"SPXA50R {spxa50r:.2f}", 2 if spxa50r >= 30 else 0)
    add("Momentum thrust", "PASS" if nyad > 1500 and spxadp > 60 else "PARTIAL", f"NYAD {nyad:.0f}, SPXADP {spxadp:.1f}", 2 if nyad > 1500 and spxadp > 60 else 1)
    add("Official NYMO", "PASS" if pd.notna(nymo) and nymo > -20 else "PARTIAL", f"NYMO {nymo:.2f}" if pd.notna(nymo) else "stale/missing", 2 if pd.notna(nymo) and nymo > -20 else 1 if pd.notna(nymo) and nymo > -50 else 0)
    add("Durability", "PASS" if pd.notna(nysi) and nysi > 0 else "WAIT", f"NYSI {nysi:.2f}" if pd.notna(nysi) else "missing", 2 if pd.notna(nysi) and nysi > 0 else 0)
    add("Leadership", "PASS" if pd.notna(nyhl) and nyhl > 0 else "WAIT", f"NYHL {nyhl:.2f}, RSP:SPY {ratio:.3f}" if pd.notna(ratio) else f"NYHL {nyhl:.2f}", 1 if pd.notna(nyhl) and nyhl > 0 else 0)

    out = pd.DataFrame(rows).sort_values(["Score", "Signal"], ascending=[False, True]).reset_index(drop=True)
    return out


# =============================
# App
# =============================
st.set_page_config(page_title="Breadth Engine Dashboard", layout="wide")
st.title("Breadth Engine Dashboard")
st.caption("Historical breadth + real-time overlay + narrative engine")

with st.sidebar:
    st.header("Inputs")
    hist_file = st.file_uploader("Historical CSV", type=["csv"])
    rt_file = st.file_uploader("Real-time CSV", type=["csv"])
    show_symbols = st.multiselect("Charts", CORE_SYMBOLS, default=["$BPSPX", "$SPXA50R", "$NYMO", "$NYSI", "$NYAD", "$NYHL", "RSP", "VXX"])
    use_defaults = st.checkbox("Use default uploaded files from /mnt/data", value=(hist_file is None and rt_file is None))

hist_path = "/mnt/data/1yrbreadth - Sheet1.csv"
rt_path = "/mnt/data/real time.csv"

if hist_file is not None:
    hist_path = "/tmp/historical_upload.csv"
    with open(hist_path, "wb") as f:
        f.write(hist_file.getbuffer())
if rt_file is not None:
    rt_path = "/tmp/realtime_upload.csv"
    with open(rt_path, "wb") as f:
        f.write(rt_file.getbuffer())

if not use_defaults and (hist_file is None or rt_file is None):
    st.info("Upload both files or check 'Use default uploaded files'.")
    st.stop()

with st.spinner("Parsing files and building indicators..."):
    hist_long = parse_historical_raw(hist_path)
    hist_feat = add_indicator_features(hist_long)
    rt = load_realtime_table(rt_path)

wide = wide_from_long(hist_long)
latest_hist_date = wide.index.max()
prev_hist_date = wide.index[-2] if len(wide.index) > 1 else latest_hist_date
latest_hist = wide.loc[latest_hist_date].to_dict()
prev_hist = wide.loc[prev_hist_date].to_dict() if prev_hist_date is not None else None

# Build current snapshot from real-time if symbol present else fallback to historical
snapshot: Dict[str, float] = latest_hist.copy()
for _, row in rt.iterrows():
    snapshot[row["Symbol"]] = float(row["Close"])

bpspx_dir = derive_bpspx_direction(snapshot.get("$BPSPX", np.nan), prev_hist.get("$BPSPX") if prev_hist else None)
snapshot["NYMO_PROXY"] = compute_nymo_proxy(snapshot.get("$NYAD", np.nan), snapshot.get("$SPXADP", np.nan), bpspx_dir)

# Summary cards
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("BPSPX", f"{snapshot.get('$BPSPX', np.nan):.2f}")
c2.metric("SPXA50R", f"{snapshot.get('$SPXA50R', np.nan):.2f}")
c3.metric("Official NYMO", f"{snapshot.get('$NYMO', np.nan):.2f}")
c4.metric("NYSI", f"{snapshot.get('$NYSI', np.nan):.2f}")
c5.metric("NYMO Proxy", f"{snapshot['NYMO_PROXY']:.2f}")

left, right = st.columns([1.1, 0.9])

with left:
    st.subheader("Market Narrative")
    st.write(build_narrative(snapshot, prev_hist))

    st.subheader("Signal Ranking")
    st.dataframe(rank_signals(snapshot, prev_hist), use_container_width=True, hide_index=True)

    st.subheader("Real-time Snapshot")
    st.dataframe(rt[[c for c in ["Symbol", "Close", "Daily PctChange(1,Daily Close)"] if c in rt.columns]], use_container_width=True, hide_index=True)

with right:
    st.subheader("Breadth Readiness")
    rsp_ready = snapshot.get("$BPSPX", 0) > 45 and snapshot.get("NYMO_PROXY", 0) > 0.5
    ursp_ready = snapshot.get("$BPSPX", 0) > 45 and snapshot.get("$SPXA50R", 0) > 30 and snapshot.get("$NYSI", -999) > -50
    st.success("RSP Swing Ready", icon="✅") if rsp_ready else st.warning("RSP Swing Not Fully Confirmed", icon="⚠️")
    st.success("URSP / Leverage Ready", icon="✅") if ursp_ready else st.warning("URSP Not Ready", icon="⚠️")

    bq, bq_notes = bounce_quality(snapshot, prev_hist)
    st.metric("Bounce Quality", f"{bq}/7")
    st.write("; ".join(bq_notes) if bq_notes else "Limited confirmation")

# Charts
st.subheader("Historical Charts + Oscillators")
for sym in show_symbols:
    g = hist_feat[hist_feat["symbol"] == sym].sort_values("date").copy()
    if g.empty:
        continue
    g["candle_pattern"] = g.apply(detect_candle_pattern, axis=1)
    g["bear_kiss_50"] = detect_bear_kiss(g, "ma50")
    g["bull_kiss_50"] = detect_bull_kiss(g, "ma50")
    last = g.iloc[-1]
    with st.expander(f"{sym}", expanded=sym in ["$BPSPX", "$SPXA50R", "RSP"]):
        col_a, col_b = st.columns([1.7, 1])
        with col_a:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=g["date"], open=g["open"], high=g["high"], low=g["low"], close=g["close"], name=sym))
            if g["ma20"].notna().any():
                fig.add_trace(go.Scatter(x=g["date"], y=g["ma20"], name="MA20", line=dict(width=1)))
            if g["ma50"].notna().any():
                fig.add_trace(go.Scatter(x=g["date"], y=g["ma50"], name="MA50", line=dict(width=1)))
            fig.update_layout(height=350, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)

            osc = go.Figure()
            osc.add_trace(go.Scatter(x=g["date"], y=g["tsi_fast"], name="TSI(4,2,4)"))
            osc.add_trace(go.Scatter(x=g["date"], y=g["tsi_fast_sig"], name="TSI Signal"))
            osc.add_trace(go.Scatter(x=g["date"], y=g["cci20"], name="CCI20", visible="legendonly"))
            osc.add_trace(go.Scatter(x=g["date"], y=g["rsi14"], name="RSI14", visible="legendonly"))
            osc.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(osc, use_container_width=True)

        with col_b:
            tsi_cross_up = last["tsi_fast"] > last["tsi_fast_sig"] and g.iloc[-2]["tsi_fast"] <= g.iloc[-2]["tsi_fast_sig"] if len(g) > 1 and pd.notna(last["tsi_fast"]) and pd.notna(last["tsi_fast_sig"]) else False
            tsi_cross_down = last["tsi_fast"] < last["tsi_fast_sig"] and g.iloc[-2]["tsi_fast"] >= g.iloc[-2]["tsi_fast_sig"] if len(g) > 1 and pd.notna(last["tsi_fast"]) and pd.notna(last["tsi_fast_sig"]) else False
            cci_creep = g["cci20"].tail(3).is_monotonic_increasing if len(g) >= 3 else False
            rsi_creep = g["rsi14"].tail(3).is_monotonic_increasing if len(g) >= 3 else False
            st.markdown("**Latest diagnostics**")
            diag = {
                "Close": round(float(last["close"]), 2),
                "%B20": None if pd.isna(last["pct_b20"]) else round(float(last["pct_b20"]), 2),
                "ROC3": None if pd.isna(last["roc3"]) else round(float(last["roc3"]), 2),
                "RSI14": None if pd.isna(last["rsi14"]) else round(float(last["rsi14"]), 2),
                "CCI20": None if pd.isna(last["cci20"]) else round(float(last["cci20"]), 2),
                "TSI": None if pd.isna(last["tsi_fast"]) else round(float(last["tsi_fast"]), 2),
                "TSI signal": None if pd.isna(last["tsi_fast_sig"]) else round(float(last["tsi_fast_sig"]), 2),
                "Candle": last["candle_pattern"],
                "Bear kiss 50MA": bool(last["bear_kiss_50"]),
                "Bull kiss 50MA": bool(last["bull_kiss_50"]),
                "TSI cross up": tsi_cross_up,
                "TSI cross down": tsi_cross_down,
                "CCI creeping up": bool(cci_creep),
                "RSI creeping up": bool(rsi_creep),
            }
            st.json(diag)

# LLM-style chat scaffold (rule-based for now)
st.subheader("Narrative / Chat")
q = st.text_input("Ask about the breadth state", placeholder="Is this RSP-ready? Did SPXA50R reject 30? Is this a bear kiss?")
if q:
    ql = q.lower()
    answer = []
    if "rsp" in ql:
        answer.append("RSP is closer than leverage. It becomes cleaner when BPSPX > 45 and participation keeps expanding.")
    if "ursp" in ql or "spxl" in ql:
        answer.append("Leverage is best reserved for confirmed repair: BPSPX > 45, SPXA50R > 30, and NYSI not collapsing.")
    if "bear kiss" in ql:
        recent = hist_feat[hist_feat["symbol"] == "RSP"].sort_values("date").copy()
        if not recent.empty:
            recent["bear_kiss_50"] = detect_bear_kiss(recent, "ma50")
            answer.append(f"RSP latest bear-kiss-on-50MA flag: {bool(recent.iloc[-1]['bear_kiss_50'])}.")
    if "thrust" in ql:
        answer.append(f"Current NYMO proxy is {snapshot['NYMO_PROXY']:.2f}. NYAD {snapshot.get('$NYAD', np.nan):.0f} and SPXADP {snapshot.get('$SPXADP', np.nan):.1f} define whether the thrust is real.")
    if not answer:
        answer.append(build_narrative(snapshot, prev_hist))
    st.write(" ".join(answer))

# Downloads
st.subheader("Downloads")
parsed_hist_path = "/mnt/data/parsed_breadth_history.csv"
hist_feat.to_csv(parsed_hist_path, index=False)
st.download_button("Download parsed historical CSV", data=open(parsed_hist_path, "rb").read(), file_name="parsed_breadth_history.csv", mime="text/csv")

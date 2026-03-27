
import io
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# -----------------------------
# Page + Styling
# -----------------------------
st.set_page_config(page_title="RSP / URSP Breadth Model v4", layout="wide", page_icon="📈")

CUSTOM_CSS = """
<style>
:root{
  --bg:#0b1020;
  --panel:#111936;
  --panel-2:#162246;
  --text:#eaf0ff;
  --muted:#9cb0df;
  --green:#22c55e;
  --yellow:#f59e0b;
  --red:#ef4444;
  --blue:#38bdf8;
  --purple:#a78bfa;
}
.block-container{
  padding-top: 1.2rem;
  padding-bottom: 2rem;
}
.main-title{
  padding: 1rem 1.25rem;
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(56,189,248,.18), rgba(167,139,250,.18));
  border: 1px solid rgba(148,163,184,.20);
  margin-bottom: 1rem;
}
.soft-card{
  background: linear-gradient(180deg, rgba(17,25,54,.96), rgba(10,17,38,.98));
  border: 1px solid rgba(148,163,184,.24);
  border-radius: 18px;
  padding: 1rem 1rem .95rem 1rem;
  box-shadow: 0 10px 35px rgba(0,0,0,.22);
}
.score-card{
  min-height: 170px;
  display:flex;
  flex-direction:column;
  justify-content:space-between;
}
.score-title{
  color:#bcd0ff;
  font-size:1.05rem;
  font-weight:800;
  letter-spacing:.02em;
}
.score-value{
  font-size:3.15rem;
  line-height:1.0;
  font-weight:950;
  color:#ffffff;
  margin:.35rem 0 .35rem 0;
  letter-spacing:-.03em;
}
.score-subtitle{
  font-size:1.02rem;
  font-weight:800;
}
.score-bar{
  width:100%;
  height:12px;
  border-radius:999px;
  background:rgba(255,255,255,.10);
  overflow:hidden;
  border:1px solid rgba(255,255,255,.08);
  margin-top:.65rem;
}
.score-fill{
  height:100%;
  border-radius:999px;
}
.score-fill-green{background:linear-gradient(90deg,#22c55e,#4ade80);}
.score-fill-yellow{background:linear-gradient(90deg,#f59e0b,#fbbf24);}
.score-fill-red{background:linear-gradient(90deg,#ef4444,#f87171);}
.score-fill-blue{background:linear-gradient(90deg,#38bdf8,#60a5fa);}
.kpi-row{
  display:grid;
  grid-template-columns:repeat(3,minmax(0,1fr));
  gap:.75rem;
  margin-top:.85rem;
}
.kpi-box{
  background:rgba(255,255,255,.04);
  border:1px solid rgba(255,255,255,.10);
  border-radius:14px;
  padding:.8rem .85rem;
}
.kpi-label{
  color:#9fb2de;
  font-size:.9rem;
  font-weight:700;
}
.kpi-value{
  color:white;
  font-size:1.85rem;
  font-weight:900;
  line-height:1.05;
  margin-top:.15rem;
}
.action-box{
  border-radius:16px;
  padding:.85rem 1rem;
  margin:.5rem 0;
  border:1px solid rgba(255,255,255,.10);
}
.action-existing{background:rgba(56,189,248,.10);}
.action-new{background:rgba(245,158,11,.10);}
.action-add{background:rgba(34,197,94,.10);}
.big-code pre{font-size:1.02rem !important; line-height:1.45 !important;}
.pill{
  display:inline-block;
  padding:.3rem .6rem;
  border-radius:999px;
  font-size:.82rem;
  font-weight:700;
  border:1px solid rgba(255,255,255,.12);
  margin-right:.35rem;
}
.pill-green{background:rgba(34,197,94,.16); color:#bbf7d0;}
.pill-yellow{background:rgba(245,158,11,.16); color:#fde68a;}
.pill-red{background:rgba(239,68,68,.16); color:#fecaca;}
.pill-blue{background:rgba(56,189,248,.16); color:#bae6fd;}
.section-label{
  font-size:1.02rem;
  font-weight:800;
  margin-bottom:.45rem;
}
.small-muted{
  color:#93a4cc;
  font-size:.88rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(
    """
    <div class="main-title">
      <div style="font-size:1.55rem;font-weight:900;">📈 RSP / URSP Breadth Model v4</div>
      <div class="small-muted">Transition-aware breadth engine with NYMO proxy governance, bounce quality, action hierarchy, polished decision visuals, and a canary-overlay backtest for RSP.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Paths / constants
# -----------------------------
APP_DIR = Path("breadth_model_store")
APP_DIR.mkdir(exist_ok=True)
BASELINE_PATH = APP_DIR / "baseline_hist.parquet"
BASELINE_META_PATH = APP_DIR / "baseline_meta.json"
UPLOAD_HISTORY_PATH = APP_DIR / "upload_history.csv"
SNAPSHOT_DIR = APP_DIR / "snapshots"
SNAPSHOT_DIR.mkdir(exist_ok=True)

OHLC_PATTERN = re.compile(
    r"(?P<day>Mon|Tue|Wed|Thu|Fri)\s+"
    r"(?P<date>\d{2}-\d{2}-\d{4})\s+"
    r"(?P<open>-?\d+(?:\.\d+)?)\s+"
    r"(?P<high>-?\d+(?:\.\d+)?)\s+"
    r"(?P<low>-?\d+(?:\.\d+)?)\s+"
    r"(?P<close>-?\d+(?:\.\d+)?)\s+"
    r"(?P<volume>-?\d+(?:\.\d+)?)"
)

DEFAULT_THRESHOLDS = {
    "BPSPX_BB_LIFT": 0.20,
    "SPXA50R_REPAIR": 30.0,
    "BPSPX_CONFIRM": 45.0,
    "NYMO_HEALTHY": -20.0,
    "NYMO_WASHOUT": -80.0,
    "NYSI_POSITIVE": 0.0,
    "NYSI_LEVERAGE_OK": -50.0,
    "NYAD_THRUST": 1500.0,
    "SPXADP_THRUST": 60.0,
    "CPCE_FEAR": 0.80,
    "VXX_BB_PROBE_MAX": 0.60,
    "URSP_TOTAL_MIN": 72.0,
    "RSP_TOTAL_MIN": 50.0,
    "DELTA_THRUST": 2.5,
    "DELTA_SHOCK": 5.0,
}
SETUP_MAX = 35
CONFIRM_MAX = 35
REGIME_MAX = 30
TOTAL_MAX = 100
DECISION_SYMBOLS = [
    "$BPSPX", "$SPXA50R", "$NYMO", "$NYSI", "$NYAD", "$SPXADP",
    "$CPCE", "$NYHL", "RSP:SPY", "VXX", "RSP", "URSP", "SPY",
]
CHART_SYMBOLS = ["$BPSPX", "$SPXA50R", "$NYMO", "$NYSI", "$NYAD", "$CPCE", "RSP", "SPY", "VXX"]

CANARY_TICKERS = ("RSP", "SPY", "HYG", "SHY", "SMH", "SOXX", "XLF", "IWM", "XLY", "SPXS", "SVOL", "VXX")
CANARY_RATIO_CONFIG = {
    "SPXS:SVOL (Stress/Carry)": {"num": "SPXS", "den": "SVOL", "invert": True,  "weight": 0.20},
    "HYG:SHY (Credit)":         {"num": "HYG",  "den": "SHY",  "invert": False, "weight": 0.18},
    "SMH:SPY (Semis Lead)":     {"num": "SMH",  "den": "SPY",  "invert": False, "weight": 0.16},
    "XLF:SPY (Financials)":     {"num": "XLF",  "den": "SPY",  "invert": False, "weight": 0.12},
    "RSP:SPY (Breadth Lead)":   {"num": "RSP",  "den": "SPY",  "invert": False, "weight": 0.12},
    "IWM:SPY (Small Caps)":     {"num": "IWM",  "den": "SPY",  "invert": False, "weight": 0.10},
    "SPY:VXX (Vol Confirm)":    {"num": "SPY",  "den": "VXX",  "invert": False, "weight": 0.07},
    "XLY:SPY (Discretionary)":  {"num": "XLY",  "den": "SPY",  "invert": False, "weight": 0.05},
}

# -----------------------------
# Helpers
# -----------------------------
def normalize_symbol(sym: str) -> str:
    return str(sym).strip()

def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan

def fmt_num(val, digits=2) -> str:
    if pd.isna(val):
        return "n/a"
    return f"{float(val):.{digits}f}"

def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default

def save_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2))

def load_upload_history() -> pd.DataFrame:
    if not UPLOAD_HISTORY_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(UPLOAD_HISTORY_PATH)
    except Exception:
        return pd.DataFrame()

def append_upload_history(row: Dict):
    hist = load_upload_history()
    hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    hist.to_csv(UPLOAD_HISTORY_PATH, index=False)

def save_snapshot_file(realtime_df: pd.DataFrame) -> Path:
    path = SNAPSHOT_DIR / f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    realtime_df.to_csv(path, index=False)
    return path

def load_latest_saved_snapshot() -> pd.DataFrame:
    hist = load_upload_history()
    if hist.empty or "snapshot_file" not in hist.columns:
        return pd.DataFrame()
    latest_path = hist.sort_values("upload_ts").iloc[-1]["snapshot_file"]
    if pd.isna(latest_path):
        return pd.DataFrame()
    p = Path(str(latest_path))
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

def load_selected_snapshot(snapshot_path: str) -> pd.DataFrame:
    if not snapshot_path:
        return pd.DataFrame()
    p = Path(snapshot_path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

def latest_upload_rows(n: int = 5) -> pd.DataFrame:
    hist = load_upload_history()
    if hist.empty:
        return hist
    return hist.sort_values("upload_ts").tail(n).reset_index(drop=True)

def uploads_in_lookback(history_df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    if history_df.empty or "upload_ts" not in history_df.columns:
        return pd.DataFrame()
    h = history_df.copy()
    h["upload_ts"] = pd.to_datetime(h["upload_ts"], errors="coerce")
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    return h[h["upload_ts"] >= cutoff].sort_values("upload_ts")

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1 / period, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))

def percent_b(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return (series - lower) / (upper - lower).replace(0, np.nan)

def roc(series: pd.Series, period: int = 3) -> pd.Series:
    return 100 * (series / series.shift(period) - 1)

def true_strength_index(series: pd.Series, long: int = 25, short: int = 13, signal: int = 7):
    m = series.diff()
    a = m.abs()
    m1 = ema(ema(m, long), short)
    a1 = ema(ema(a, long), short)
    tsi = 100 * (m1 / a1.replace(0, np.nan))
    sig = ema(tsi, signal)
    return tsi, sig

def slope_n(series: pd.Series, n: int = 3) -> pd.Series:
    return series - series.shift(n)

def write_snapshot_fields(snapshot: Dict[str, float], sym: str, values: Dict[str, float]):
    snapshot[sym] = values.get("close", np.nan)
    snapshot[f"{sym}_%B"] = values.get("pct_b20", np.nan)
    snapshot[f"{sym}_RSI14"] = values.get("rsi14", np.nan)
    snapshot[f"{sym}_CCI20"] = values.get("cci20", np.nan)
    snapshot[f"{sym}_TSI"] = values.get("tsi_fast", np.nan)
    snapshot[f"{sym}_ROC3"] = values.get("roc3", np.nan)
    snapshot[f"{sym}_SLOPE3"] = values.get("slope3", np.nan)

# -----------------------------
# Parsing / features
# -----------------------------
@st.cache_data(show_spinner=False)
def parse_stockcharts_historical_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    raw = pd.read_csv(io.BytesIO(file_bytes), header=None)
    records = []
    for cell in raw.iloc[:, 0].dropna().astype(str):
        symbol_match = re.match(r"\s*([^,]+),", cell)
        if not symbol_match:
            continue
        symbol = normalize_symbol(symbol_match.group(1))
        for m in OHLC_PATTERN.finditer(cell):
            dt = pd.to_datetime(m.group("date"), format="%m-%d-%Y", errors="coerce")
            if pd.isna(dt):
                continue
            records.append({
                "date": dt,
                "symbol": symbol,
                "open": float(m.group("open")),
                "high": float(m.group("high")),
                "low": float(m.group("low")),
                "close": float(m.group("close")),
                "volume": float(m.group("volume")),
            })
    if not records:
        raise ValueError("Historical baseline could not be parsed.")
    return pd.DataFrame(records).sort_values(["symbol", "date"]).reset_index(drop=True)

@st.cache_data(show_spinner=False)
def parse_realtime_snapshot_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    if "Symbol" not in df.columns:
        raise ValueError("Realtime snapshot must contain a 'Symbol' column.")
    df["Symbol"] = df["Symbol"].astype(str).map(normalize_symbol)
    close_candidates = ["Close", "Last", "Price", "Current", "Value", "Daily Close", "Close Price"]
    close_col = next((c for c in close_candidates if c in df.columns), None)
    if close_col is None:
        raise ValueError("Realtime snapshot must contain a close-like column.")
    df["Close"] = pd.to_numeric(df[close_col], errors="coerce")
    pct_candidates = ["Daily PctChange(1,Daily Close)", "% Change", "Pct Change", "Change %", "Daily Change %"]
    pct_col = next((c for c in pct_candidates if c in df.columns), None)
    df["PctChange"] = pd.to_numeric(df[pct_col], errors="coerce") if pct_col else np.nan
    return df

@st.cache_data(show_spinner=False)
def add_indicator_features(hist: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for sym, g in hist.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()
        g["rsi14"] = rsi(g["close"], 14)
        g["cci20"] = cci(g["high"], g["low"], g["close"], 20)
        g["pct_b20"] = percent_b(g["close"], 20, 2.0)
        g["roc3"] = roc(g["close"], 3)
        g["slope3"] = slope_n(g["close"], 3)
        g["tsi_fast"], g["tsi_fast_sig"] = true_strength_index(g["close"], 4, 2, 4)
        g["ma20"] = g["close"].rolling(20).mean()
        g["ma50"] = g["close"].rolling(50).mean()
        frames.append(g)
    return pd.concat(frames, ignore_index=True)

def get_feature_snapshot(hist_feat: pd.DataFrame, which: str = "latest") -> Dict[str, float]:
    snapshot = {}
    for sym, g in hist_feat.groupby("symbol", sort=False):
        g = g.sort_values("date")
        if which == "prior":
            if len(g) < 2:
                continue
            row = g.iloc[-2]
        else:
            row = g.iloc[-1]
        write_snapshot_fields(snapshot, sym, {
            "close": safe_float(row.get("close")),
            "pct_b20": safe_float(row.get("pct_b20")),
            "rsi14": safe_float(row.get("rsi14")),
            "cci20": safe_float(row.get("cci20")),
            "tsi_fast": safe_float(row.get("tsi_fast")),
            "roc3": safe_float(row.get("roc3")),
            "slope3": safe_float(row.get("slope3")),
        })
    return snapshot

def recompute_latest_indicators_from_realtime(hist_feat: pd.DataFrame, realtime_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out = {}
    rt_map = {str(row["Symbol"]).strip(): safe_float(row["Close"]) for _, row in realtime_df.iterrows() if pd.notna(safe_float(row["Close"]))}
    for sym, new_close in rt_map.items():
        g = hist_feat[hist_feat["symbol"] == sym].sort_values("date").copy()
        if g.empty:
            continue
        g.iloc[-1, g.columns.get_loc("close")] = new_close
        g.iloc[-1, g.columns.get_loc("high")] = max(g.iloc[-1]["high"], new_close)
        g.iloc[-1, g.columns.get_loc("low")] = min(g.iloc[-1]["low"], new_close)
        if len(g) > 1:
            g.iloc[-1, g.columns.get_loc("open")] = g.iloc[-2]["close"]
        g["rsi14"] = rsi(g["close"], 14)
        g["cci20"] = cci(g["high"], g["low"], g["close"], 20)
        g["pct_b20"] = percent_b(g["close"], 20, 2.0)
        g["roc3"] = roc(g["close"], 3)
        g["slope3"] = slope_n(g["close"], 3)
        g["tsi_fast"], g["tsi_fast_sig"] = true_strength_index(g["close"], 4, 2, 4)
        last = g.iloc[-1]
        out[sym] = {
            "close": safe_float(last["close"]),
            "pct_b20": safe_float(last["pct_b20"]),
            "rsi14": safe_float(last["rsi14"]),
            "cci20": safe_float(last["cci20"]),
            "tsi_fast": safe_float(last["tsi_fast"]),
            "roc3": safe_float(last["roc3"]),
            "slope3": safe_float(last["slope3"]),
        }
    return out

# -----------------------------
# Proxy governance / classification
# -----------------------------
def classify_delta(sym: str, cur: float, prev: float, thresholds: Dict[str, float]) -> Tuple[str, str]:
    if pd.isna(cur) or pd.isna(prev):
        return "n/a", "n/a"
    delta = cur - prev
    absd = abs(delta)
    if sym in {"$NYAD", "$SPXADP"}:
        if delta >= thresholds["NYAD_THRUST"] * 0.50 if sym == "$NYAD" else delta >= thresholds["SPXADP_THRUST"] * 0.60:
            return "Thrust", "surging"
        if delta <= -(thresholds["NYAD_THRUST"] * 0.35 if sym == "$NYAD" else thresholds["SPXADP_THRUST"] * 0.50):
            return "Collapse", "breaking lower"
    if sym in {"$BPSPX_%B", "$SPXA50R", "$BPSPX", "$NYMO", "$NYSI", "RSP:SPY"}:
        if absd >= thresholds["DELTA_SHOCK"]:
            return ("Shock+", "vertical repair") if delta > 0 else ("Shock-", "vertical damage")
        if absd >= thresholds["DELTA_THRUST"]:
            return ("Thrust", "repairing quickly") if delta > 0 else ("Collapse", "fading quickly")
    if delta > 0:
        return "Improve", "grinding higher"
    if delta < 0:
        return "Fade", "rolling over"
    return "Flat", "flat"

def compute_proxy_nymo(snapshot: Dict[str, float], prev_snapshot: Dict[str, float]) -> Dict[str, float]:
    nyad = safe_float(snapshot.get("$NYAD", np.nan))
    spxadp = safe_float(snapshot.get("$SPXADP", np.nan))
    prev_nyad = safe_float(prev_snapshot.get("$NYAD", np.nan))
    prev_spxadp = safe_float(prev_snapshot.get("$SPXADP", np.nan))
    if pd.isna(nyad) and pd.isna(spxadp):
        return {"proxy_raw": np.nan, "proxy_nymo": np.nan, "proxy_delta": np.nan, "proxy_state": "Unavailable"}
    # User-requested weighting
    cur_raw = 0.6 * (0 if pd.isna(nyad) else nyad) + 0.4 * (0 if pd.isna(spxadp) else spxadp)
    prev_raw = 0.6 * (0 if pd.isna(prev_nyad) else prev_nyad) + 0.4 * (0 if pd.isna(prev_spxadp) else prev_spxadp)
    # compress to NYMO-like range without pretending it is official
    proxy_nymo = 100 * np.tanh(cur_raw / 1600.0)
    prev_proxy_nymo = 100 * np.tanh(prev_raw / 1600.0)
    proxy_delta = proxy_nymo - prev_proxy_nymo
    if proxy_nymo <= -70:
        state = "Deep washout"
    elif proxy_nymo <= -20:
        state = "Negative but repairing" if proxy_delta > 0 else "Negative and weak"
    elif proxy_nymo <= 20:
        state = "Neutral / crossing"
    else:
        state = "Positive thrust"
    return {
        "proxy_raw": cur_raw,
        "proxy_nymo": proxy_nymo,
        "proxy_delta": proxy_delta,
        "proxy_state": state,
    }

def breadth_source_mode(use_proxy: bool) -> str:
    return "Proxy mode (intraday / unofficial)" if use_proxy else "Official mode (EOD / official NYMO)"

def get_nymo_effective(snapshot: Dict[str, float], prev_snapshot: Dict[str, float], use_proxy: bool) -> Dict[str, float]:
    official = safe_float(snapshot.get("$NYMO", np.nan))
    prior_official = safe_float(prev_snapshot.get("$NYMO", np.nan))
    proxy = compute_proxy_nymo(snapshot, prev_snapshot)
    if use_proxy or pd.isna(official):
        return {
            "mode": breadth_source_mode(True),
            "value": proxy["proxy_nymo"],
            "delta": proxy["proxy_delta"],
            "label": "NYMO Proxy",
            "state": proxy["proxy_state"],
        }
    return {
        "mode": breadth_source_mode(False),
        "value": official,
        "delta": official - prior_official if pd.notna(official) and pd.notna(prior_official) else np.nan,
        "label": "Official NYMO",
        "state": "Official series",
    }

# -----------------------------
# Scores
# -----------------------------
def qwen_repair_trigger(snapshot: Dict[str, float], thresholds: Dict[str, float]) -> bool:
    bpspx_bb = safe_float(snapshot.get("$BPSPX_%B", np.nan))
    spxa50r = safe_float(snapshot.get("$SPXA50R", np.nan))
    return bool(pd.notna(bpspx_bb) and pd.notna(spxa50r) and bpspx_bb > thresholds["BPSPX_BB_LIFT"] and spxa50r > thresholds["SPXA50R_REPAIR"])

def breadth_stage(snapshot: Dict[str, float], thresholds: Dict[str, float], nymo_eff: Optional[float] = None) -> Tuple[int, str]:
    bpspx_bb = safe_float(snapshot.get("$BPSPX_%B", np.nan))
    spxa50r = safe_float(snapshot.get("$SPXA50R", np.nan))
    bpspx = safe_float(snapshot.get("$BPSPX", np.nan))
    nysi = safe_float(snapshot.get("$NYSI", np.nan))
    nymo_eff = safe_float(nymo_eff)
    if pd.isna(bpspx_bb):
        return 0, "Unknown"
    if bpspx_bb < thresholds["BPSPX_BB_LIFT"]:
        return 0, "Oversold / probe"
    if bpspx_bb >= thresholds["BPSPX_BB_LIFT"] and spxa50r <= thresholds["SPXA50R_REPAIR"]:
        return 1, "Bounce starting"
    if bpspx_bb > thresholds["BPSPX_BB_LIFT"] and spxa50r > thresholds["SPXA50R_REPAIR"] and bpspx <= thresholds["BPSPX_CONFIRM"]:
        return 2, "Breadth repair"
    if bpspx > thresholds["BPSPX_CONFIRM"] and nysi <= thresholds["NYSI_POSITIVE"]:
        return 3, "Participation confirmation"
    if bpspx > thresholds["BPSPX_CONFIRM"] and nysi > thresholds["NYSI_POSITIVE"] and pd.notna(nymo_eff) and nymo_eff > thresholds["NYMO_HEALTHY"]:
        return 4, "Durable uptrend"
    return 3, "Participation confirmation"

def setup_score_components(snapshot: Dict[str, float], prev_snapshot: Dict[str, float], thresholds: Dict[str, float]):
    rows = []
    pts = 0
    bpspx_bb = safe_float(snapshot.get("$BPSPX_%B", np.nan))
    bpspx_bb_delta = bpspx_bb - safe_float(prev_snapshot.get("$BPSPX_%B", np.nan))
    spxa50r = safe_float(snapshot.get("$SPXA50R", np.nan))
    spxa50r_delta = spxa50r - safe_float(prev_snapshot.get("$SPXA50R", np.nan))
    cpce = safe_float(snapshot.get("$CPCE", np.nan))
    vxx_bb = safe_float(snapshot.get("VXX_%B", np.nan))
    nyad = safe_float(snapshot.get("$NYAD", np.nan))
    rsp_ratio = safe_float(snapshot.get("RSP:SPY", np.nan))
    rsp_ratio_delta = rsp_ratio - safe_float(prev_snapshot.get("RSP:SPY", np.nan))

    score = 10 if pd.notna(bpspx_bb) and bpspx_bb < 0.20 else 5 if pd.notna(bpspx_bb) and bpspx_bb < 0.35 else 0
    if pd.notna(bpspx_bb_delta) and bpspx_bb_delta > 0:
        score += 2
    pts += score
    rows.append({"Component": "BPSPX washout + lift", "Current": fmt_num(bpspx_bb), "Delta": fmt_num(bpspx_bb_delta), "Target": "<0.20 then rising", "Score": min(score, 12)})

    score = 8 if pd.notna(spxa50r) and spxa50r < 30 else 4 if pd.notna(spxa50r) and spxa50r < 35 else 0
    if pd.notna(spxa50r_delta) and spxa50r_delta > 1:
        score += 2
    pts += score
    rows.append({"Component": "SPXA50R collapse + repair", "Current": fmt_num(spxa50r), "Delta": fmt_num(spxa50r_delta), "Target": "<30 then improving", "Score": min(score, 10)})

    score = 5 if pd.notna(cpce) and cpce >= thresholds["CPCE_FEAR"] else 0
    pts += score
    rows.append({"Component": "CPCE fear support", "Current": fmt_num(cpce), "Delta": "n/a", "Target": f">={thresholds['CPCE_FEAR']:.2f}", "Score": score})

    score = 4 if pd.notna(vxx_bb) and vxx_bb < thresholds["VXX_BB_PROBE_MAX"] else 0
    pts += score
    rows.append({"Component": "VXX contained", "Current": fmt_num(vxx_bb), "Delta": "n/a", "Target": f"<{thresholds['VXX_BB_PROBE_MAX']:.2f}", "Score": score})

    score = 4 if pd.notna(nyad) and nyad > thresholds["NYAD_THRUST"] else 2 if pd.notna(nyad) and nyad > 500 else 0
    pts += score
    rows.append({"Component": "NYAD thrust support", "Current": fmt_num(nyad, 0), "Delta": "n/a", "Target": f">{thresholds['NYAD_THRUST']:.0f}", "Score": score})

    score = 4 if pd.notna(rsp_ratio_delta) and rsp_ratio_delta > 0 else 2 if pd.notna(rsp_ratio) else 0
    pts += score
    rows.append({"Component": "RSP:SPY leadership", "Current": fmt_num(rsp_ratio, 4), "Delta": fmt_num(rsp_ratio_delta, 4), "Target": "flat / rising", "Score": score})

    pts = int(min(SETUP_MAX, pts))
    return pd.DataFrame(rows), pts

def confirmation_score_components(snapshot: Dict[str, float], prev_snapshot: Dict[str, float], thresholds: Dict[str, float], nymo_effective: Dict[str, float]):
    rows = []
    pts = 0
    bpspx_bb = safe_float(snapshot.get("$BPSPX_%B", np.nan))
    spxa50r = safe_float(snapshot.get("$SPXA50R", np.nan))
    bpspx = safe_float(snapshot.get("$BPSPX", np.nan))
    spxadp = safe_float(snapshot.get("$SPXADP", np.nan))
    rsp_ratio = safe_float(snapshot.get("RSP:SPY", np.nan))
    rsp_ratio_delta = rsp_ratio - safe_float(prev_snapshot.get("RSP:SPY", np.nan))
    nymo = safe_float(nymo_effective.get("value", np.nan))
    nymo_delta = safe_float(nymo_effective.get("delta", np.nan))

    score = 10 if pd.notna(bpspx_bb) and bpspx_bb > thresholds["BPSPX_BB_LIFT"] else 0
    pts += score
    rows.append({"Component": "BPSPX %B repair lift", "Current": fmt_num(bpspx_bb), "Delta": fmt_num(bpspx_bb - safe_float(prev_snapshot.get("$BPSPX_%B", np.nan))), "Target": f">{thresholds['BPSPX_BB_LIFT']:.2f}", "Score": score})

    score = 10 if pd.notna(spxa50r) and spxa50r > thresholds["SPXA50R_REPAIR"] else 0
    pts += score
    rows.append({"Component": "SPXA50R repair", "Current": fmt_num(spxa50r), "Delta": fmt_num(spxa50r - safe_float(prev_snapshot.get("$SPXA50R", np.nan))), "Target": f">{thresholds['SPXA50R_REPAIR']:.0f}", "Score": score})

    score = 6 if pd.notna(bpspx) and bpspx > thresholds["BPSPX_CONFIRM"] else 3 if pd.notna(bpspx) and bpspx > 40 else 0
    pts += score
    rows.append({"Component": "BPSPX participation", "Current": fmt_num(bpspx), "Delta": fmt_num(bpspx - safe_float(prev_snapshot.get("$BPSPX", np.nan))), "Target": f">{thresholds['BPSPX_CONFIRM']:.0f}", "Score": score})

    score = 5 if pd.notna(nymo) and nymo > thresholds["NYMO_HEALTHY"] else 2 if pd.notna(nymo) and nymo > -50 else 0
    if pd.notna(nymo_delta) and nymo_delta > 5:
        score += 1
    pts += score
    rows.append({"Component": f"{nymo_effective['label']} momentum", "Current": fmt_num(nymo), "Delta": fmt_num(nymo_delta), "Target": f">{thresholds['NYMO_HEALTHY']:.0f}", "Score": min(score, 6)})

    score = 3 if pd.notna(spxadp) and spxadp > thresholds["SPXADP_THRUST"] else 1 if pd.notna(spxadp) and spxadp > 20 else 0
    pts += score
    rows.append({"Component": "SPXADP breadth volume", "Current": fmt_num(spxadp), "Delta": fmt_num(spxadp - safe_float(prev_snapshot.get("$SPXADP", np.nan))), "Target": f">{thresholds['SPXADP_THRUST']:.0f}", "Score": score})

    score = 2 if pd.notna(rsp_ratio_delta) and rsp_ratio_delta >= 0 else 0
    pts += score
    rows.append({"Component": "RSP:SPY not weakening", "Current": fmt_num(rsp_ratio, 4), "Delta": fmt_num(rsp_ratio_delta, 4), "Target": ">= prior", "Score": score})

    pts = int(min(CONFIRM_MAX, pts))
    return pd.DataFrame(rows), pts

def regime_score_components(snapshot: Dict[str, float], prev_snapshot: Dict[str, float], thresholds: Dict[str, float], nymo_effective: Dict[str, float]):
    rows = []
    pts = 0
    nysi = safe_float(snapshot.get("$NYSI", np.nan))
    nyhl = safe_float(snapshot.get("$NYHL", np.nan))
    repair = qwen_repair_trigger(snapshot, thresholds)
    stage_num, stage_name = breadth_stage(snapshot, thresholds, nymo_effective.get("value", np.nan))
    nysi_delta = nysi - safe_float(prev_snapshot.get("$NYSI", np.nan))

    score = 12 if pd.notna(nysi) and nysi > thresholds["NYSI_POSITIVE"] else 6 if pd.notna(nysi) and nysi > thresholds["NYSI_LEVERAGE_OK"] else 0
    if pd.notna(nysi_delta) and nysi_delta > 0:
        score += 1
    pts += score
    rows.append({"Component": "NYSI durability", "Current": fmt_num(nysi), "Delta": fmt_num(nysi_delta), "Target": f">{thresholds['NYSI_POSITIVE']:.0f}", "Score": min(score, 13)})

    score = 7 if repair else 0
    pts += score
    rows.append({"Component": "Repair trigger", "Current": "ON" if repair else "OFF", "Delta": "n/a", "Target": "ON", "Score": score})

    score = 5 if pd.notna(nyhl) and nyhl > 0 else 2 if pd.notna(nyhl) and nyhl > -50 else 0
    pts += score
    rows.append({"Component": "NYHL breadth tone", "Current": fmt_num(nyhl), "Delta": fmt_num(nyhl - safe_float(prev_snapshot.get("$NYHL", np.nan))), "Target": ">0", "Score": score})

    score = min(5, stage_num + 1)
    pts += score
    rows.append({"Component": "Regime stage", "Current": f"{stage_num} ({stage_name})", "Delta": "n/a", "Target": "higher", "Score": score})

    pts = int(min(REGIME_MAX, pts))
    return pd.DataFrame(rows), pts

def compute_bounce_quality(snapshot: Dict[str, float], prev_snapshot: Dict[str, float], thresholds: Dict[str, float], nymo_effective: Dict[str, float]) -> pd.DataFrame:
    rows = []
    bpspx_bb = safe_float(snapshot.get("$BPSPX_%B", np.nan))
    spxa50r = safe_float(snapshot.get("$SPXA50R", np.nan))
    rsp_ratio = safe_float(snapshot.get("RSP:SPY", np.nan))
    rsp_delta = rsp_ratio - safe_float(prev_snapshot.get("RSP:SPY", np.nan))
    vxx_bb = safe_float(snapshot.get("VXX_%B", np.nan))
    nyad = safe_float(snapshot.get("$NYAD", np.nan))
    nymo = safe_float(nymo_effective.get("value", np.nan))

    comps = [
        ("BPSPX lift", 2 if pd.notna(bpspx_bb) and bpspx_bb > thresholds["BPSPX_BB_LIFT"] else 0, fmt_num(bpspx_bb)),
        ("SPXA50R repair", 2 if pd.notna(spxa50r) and spxa50r > thresholds["SPXA50R_REPAIR"] else 0, fmt_num(spxa50r)),
        ("NYMO/Proxy health", 2 if pd.notna(nymo) and nymo > thresholds["NYMO_HEALTHY"] else 1 if pd.notna(nymo) and nymo > -50 else 0, fmt_num(nymo)),
        ("NYAD thrust", 2 if pd.notna(nyad) and nyad > thresholds["NYAD_THRUST"] else 1 if pd.notna(nyad) and nyad > 500 else 0, fmt_num(nyad,0)),
        ("Leadership + vol", 2 if pd.notna(rsp_delta) and rsp_delta > 0 and pd.notna(vxx_bb) and vxx_bb < thresholds["VXX_BB_PROBE_MAX"] else 1 if pd.notna(rsp_ratio) else 0, f"RSP:SPY Δ {fmt_num(rsp_delta,4)} / VXX%B {fmt_num(vxx_bb)}"),
    ]
    for name, score, current in comps:
        rows.append({"Component": name, "Current": current, "Score": score})
    return pd.DataFrame(rows)

def chronology_guardrail(upload_history_df: pd.DataFrame, snapshot: Dict[str, float], prev_snapshot: Dict[str, float]) -> Tuple[str, str]:
    if upload_history_df.empty:
        return "Normal", "No upload chronology conflict detected."
    hist = upload_history_df.copy()
    if "ConfluenceScore" not in hist.columns:
        return "Normal", "No score history available."
    hist["ConfluenceScore"] = pd.to_numeric(hist["ConfluenceScore"], errors="coerce")
    recent = hist.sort_values("upload_ts").tail(3)
    rising = recent["ConfluenceScore"].diff().dropna().gt(0).sum()
    bpspx_bb = safe_float(snapshot.get("$BPSPX_%B", np.nan))
    prev_bpspx_bb = safe_float(prev_snapshot.get("$BPSPX_%B", np.nan))
    if len(recent) >= 2 and rising == 0 and pd.notna(bpspx_bb) and pd.notna(prev_bpspx_bb) and bpspx_bb < prev_bpspx_bb:
        return "Historical conflict", "Recent upload history is not improving; treat current bounce as lower confidence."
    if len(recent) >= 2 and recent["ConfluenceScore"].iloc[-1] < recent["ConfluenceScore"].iloc[0]:
        return "Repair fading", "Recent uploads show score deterioration."
    return "Normal", "Recent upload sequence is not contradicting the current read."

# -----------------------------
# Actions / checklist / context
# -----------------------------
def model_action_hierarchy(total_score: int, setup_score: int, confirmation_score: int, regime_score: int,
                           snapshot: Dict[str, float], thresholds: Dict[str, float],
                           nymo_effective: Dict[str, float], guardrail_status: str):
    bpspx_bb = safe_float(snapshot.get("$BPSPX_%B", np.nan))
    spxa50r = safe_float(snapshot.get("$SPXA50R", np.nan))
    nysi = safe_float(snapshot.get("$NYSI", np.nan))
    proxy_ok = pd.notna(nymo_effective["value"]) and nymo_effective["value"] > thresholds["NYMO_HEALTHY"]

    existing = "Stay defensive / monitor"
    new = "No new long"
    add = "Do not add"
    rsp_size = 0.0
    ursp_size = 0.0
    rationale = "Conditions are not aligned enough for long exposure."

    if total_score >= thresholds["URSP_TOTAL_MIN"] and setup_score >= 20 and confirmation_score >= 24 and regime_score >= 16 \
       and pd.notna(bpspx_bb) and bpspx_bb >= thresholds["BPSPX_BB_LIFT"] \
       and pd.notna(spxa50r) and spxa50r >= thresholds["SPXA50R_REPAIR"] \
       and pd.notna(nysi) and nysi > thresholds["NYSI_LEVERAGE_OK"] and proxy_ok and guardrail_status == "Normal":
        existing = "Keep long bias"
        new = "New RSP okay; URSP allowed selectively"
        add = "Can add on confirmation holds"
        rsp_size = 0.30
        ursp_size = 0.10
        rationale = "Repair, confirmation, and regime all align. URSP is gated open."
    elif total_score >= thresholds["RSP_TOTAL_MIN"] and setup_score >= 18 and confirmation_score >= 16 \
         and pd.notna(bpspx_bb) and bpspx_bb >= thresholds["BPSPX_BB_LIFT"]:
        existing = "Hold / keep existing probe"
        new = "New RSP okay"
        add = "Add only if SPXA50R holds above threshold"
        rsp_size = 0.20
        rationale = "This is a reasonable RSP repair setup, but leverage is still off."
    elif setup_score >= 15:
        existing = "Small probe only if already engaged"
        new = "New probe RSP only"
        add = "Do not add yet"
        rsp_size = 0.10
        rationale = "Setup is present, but confirmation is incomplete."

    return {
        "existing": existing,
        "new": new,
        "add": add,
        "rsp_size": rsp_size,
        "ursp_size": ursp_size,
        "rationale": rationale,
    }

def target_for_symbol(sym: str, thresholds: Dict[str, float]) -> str:
    targets = {
        "$BPSPX_%B": f">{thresholds['BPSPX_BB_LIFT']:.2f}",
        "$BPSPX": f">{thresholds['BPSPX_CONFIRM']:.0f}",
        "$SPXA50R": f">{thresholds['SPXA50R_REPAIR']:.0f}",
        "$NYMO": f">{thresholds['NYMO_HEALTHY']:.0f}",
        "$NYSI": f">{thresholds['NYSI_POSITIVE']:.0f}",
        "$NYAD": f">{thresholds['NYAD_THRUST']:.0f}",
        "$SPXADP": f">{thresholds['SPXADP_THRUST']:.0f}",
        "$CPCE": f">={thresholds['CPCE_FEAR']:.2f}",
        "VXX_%B": f"<{thresholds['VXX_BB_PROBE_MAX']:.2f}",
        "RSP:SPY": "rising",
        "VXX": "falling",
    }
    return targets.get(sym, "improving")

def verbose_state(sym: str, cur: float, prev: float, thresholds: Dict[str, float]) -> str:
    if pd.isna(cur):
        return "missing"
    delta = cur - prev if pd.notna(prev) else np.nan
    if sym == "$BPSPX_%B":
        if cur < thresholds["BPSPX_BB_LIFT"]:
            return "oversold but trying to lift" if pd.notna(delta) and delta > 0 else "oversold / pressure"
        return "repair lifted" if cur < 0.50 else "healthy expansion"
    if sym == "$SPXA50R":
        if cur < 25:
            return "very weak breadth depth"
        if cur <= thresholds["SPXA50R_REPAIR"]:
            return "near repair line"
        return "repair passed"
    if sym == "$BPSPX":
        if cur < 40:
            return "participation weak"
        if cur <= thresholds["BPSPX_CONFIRM"]:
            return "participation improving"
        return "participation confirmed"
    if sym == "$NYMO":
        if cur < thresholds["NYMO_WASHOUT"]:
            return "deep washout"
        if cur < thresholds["NYMO_HEALTHY"]:
            return "negative but repairable"
        return "healthy momentum"
    if sym == "$NYSI":
        if cur < thresholds["NYSI_LEVERAGE_OK"]:
            return "not leverage-ready"
        if cur < thresholds["NYSI_POSITIVE"]:
            return "repairing backdrop"
        return "durable backdrop"
    if sym == "$NYAD":
        return "breadth thrust" if cur > thresholds["NYAD_THRUST"] else "mixed breadth"
    if sym == "$SPXADP":
        return "advancing-volume thrust" if cur > thresholds["SPXADP_THRUST"] else "mixed breadth volume"
    if sym == "$CPCE":
        return "fear support elevated" if cur >= thresholds["CPCE_FEAR"] else "fear support fading"
    if sym == "VXX":
        return "vol easing" if pd.notna(delta) and delta < 0 else "vol pressuring"
    if sym == "VXX_%B":
        return "contained vol" if cur < thresholds["VXX_BB_PROBE_MAX"] else "elevated vol"
    if sym == "RSP:SPY":
        return "leadership improving" if pd.notna(delta) and delta > 0 else "leadership soft"
    return "improving" if pd.notna(delta) and delta > 0 else "fading" if pd.notna(delta) and delta < 0 else "flat"

def build_momentum_context_table(symbols: List[str], snapshot: Dict[str, float], prev_snapshot: Dict[str, float], thresholds: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        cur = safe_float(snapshot.get(sym, np.nan))
        prev = safe_float(prev_snapshot.get(sym, np.nan))
        delta = cur - prev if pd.notna(cur) and pd.notna(prev) else np.nan
        label, class_state = classify_delta(sym, cur, prev, thresholds)
        rows.append({
            "Symbol": sym,
            "Current": None if pd.isna(cur) else round(float(cur), 2),
            "Prior": None if pd.isna(prev) else round(float(prev), 2),
            "Delta": None if pd.isna(delta) else round(float(delta), 2),
            "Δ Class": label,
            "Target": target_for_symbol(sym, thresholds),
            "State": f"{verbose_state(sym, cur, prev, thresholds)} / {class_state}",
        })
    return pd.DataFrame(rows)

def dynamic_trading_checklist(snapshot: Dict[str, float], thresholds: Dict[str, float], history_df: pd.DataFrame,
                              action_hierarchy: Dict[str, str], guardrail_msg: str, nymo_effective: Dict[str, float]) -> str:
    bpspx_bb = safe_float(snapshot.get("$BPSPX_%B", np.nan))
    spxa50r = safe_float(snapshot.get("$SPXA50R", np.nan))
    vxx_bb = safe_float(snapshot.get("VXX_%B", np.nan))
    nyad = safe_float(snapshot.get("$NYAD", np.nan))
    rsp_spy = safe_float(snapshot.get("RSP:SPY", np.nan))
    stage_num, stage_name = breadth_stage(snapshot, thresholds, nymo_effective.get("value", np.nan))
    week_hist = uploads_in_lookback(history_df, 7)
    spxa50r_intraday_highs = 0
    if not week_hist.empty and "SPXA50R" in week_hist.columns:
        vals = pd.to_numeric(week_hist["SPXA50R"], errors="coerce")
        spxa50r_intraday_highs = int((vals > thresholds["SPXA50R_REPAIR"]).sum())
    full_confirm = pd.notna(bpspx_bb) and bpspx_bb >= thresholds["BPSPX_BB_LIFT"] and pd.notna(spxa50r) and spxa50r >= thresholds["SPXA50R_REPAIR"]
    lines = [
        f"Mode: {nymo_effective['mode']}",
        f"Stage: {stage_num} ({stage_name})",
        f"Guardrail: {guardrail_msg}",
        "",
        "Action Hierarchy:",
        f"• Existing: {action_hierarchy['existing']}",
        f"• New: {action_hierarchy['new']}",
        f"• Add: {action_hierarchy['add']}",
        "",
        "Key Filters:",
        f"• BPSPX %B lift {'✅' if pd.notna(bpspx_bb) and bpspx_bb >= thresholds['BPSPX_BB_LIFT'] else '❌'} ({fmt_num(bpspx_bb)})",
        f"• SPXA50R repair {'✅' if pd.notna(spxa50r) and spxa50r >= thresholds['SPXA50R_REPAIR'] else '❌'} ({fmt_num(spxa50r)})",
        f"• VXX %B contained {'✅' if pd.notna(vxx_bb) and vxx_bb < thresholds['VXX_BB_PROBE_MAX'] else '❌'} ({fmt_num(vxx_bb)})",
        f"• NYAD thrust {'✅' if pd.notna(nyad) and nyad > thresholds['NYAD_THRUST'] else '❌'} ({fmt_num(nyad,0)})",
        f"• RSP:SPY leadership {'✅' if pd.notna(rsp_spy) else '❌'} ({fmt_num(rsp_spy,4)})",
        "",
        f"• Full confirmation {'MET' if full_confirm else 'NOT MET'}",
        f"• SPXA50R > threshold uploads in last week: {spxa50r_intraday_highs}",
        "• Re-check after official EOD breadth refresh if in proxy mode",
    ]
    return "\n".join(lines)

# -----------------------------
# Charts / backtest
# -----------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_canary_prices(tickers: tuple, years: int = 3) -> pd.DataFrame:
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=int(years * 365.25) + 40)
    data = yf.download(
        tickers=list(tickers),
        start=start,
        end=end + pd.Timedelta(days=1),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    close_cols = {}
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            try:
                close_cols[t] = pd.to_numeric(data[t]["Close"], errors="coerce")
            except Exception:
                pass
    else:
        try:
            close_cols[tickers[0]] = pd.to_numeric(data["Close"], errors="coerce")
        except Exception:
            pass
    out = pd.DataFrame(close_cols).sort_index().ffill().dropna(how="all")
    return out

def macd_hist_series(close: pd.Series, fast: int = 24, slow: int = 52, signal: int = 18) -> pd.Series:
    mline = ema(close, fast) - ema(close, slow)
    sig = ema(mline, signal)
    return mline - sig

def close_only_stoch(close: pd.Series, length: int = 14, smoothk: int = 3, smoothd: int = 3):
    lo = close.rolling(length).min()
    hi = close.rolling(length).max()
    denom = (hi - lo).replace(0, np.nan)
    k = 100.0 * (close - lo) / denom
    k = k.rolling(smoothk).mean()
    d = k.rolling(smoothd).mean()
    return k, d

def close_only_cci(close: pd.Series, length: int = 100) -> pd.Series:
    sma = close.rolling(length).mean()
    mad = (close - sma).abs().rolling(length).mean()
    return (close - sma) / (0.015 * mad.replace(0, np.nan))

def indicator_pack_ratio(close: pd.Series) -> pd.DataFrame:
    close = close.dropna()
    if len(close) < 220:
        return pd.DataFrame()
    macdh = macd_hist_series(close).rename("macdh")
    tsi_raw, _ = true_strength_index(close, 40, 20, 10)
    tsi_raw = tsi_raw.rename("tsi")
    stoch_k, _ = close_only_stoch(close, 14, 3, 3)
    stoch_k = stoch_k.rename("stochk")
    cci100 = close_only_cci(close, 100).rename("cci")
    return pd.concat([macdh, tsi_raw, stoch_k, cci100], axis=1).dropna()

def score_from_indicators_ratio(ind: pd.DataFrame) -> pd.Series:
    if ind.empty:
        return pd.Series(dtype=float)
    bull = (ind["macdh"] > 0) & (ind["tsi"] > 0) & (ind["stochk"] > 50) & (ind["cci"] > 0)
    bear = (ind["macdh"] < 0) & (ind["tsi"] < 0) & (ind["stochk"] < 50) & (ind["cci"] < 0)
    sc = pd.Series(0.0, index=ind.index)
    sc[bull] = 1.0
    sc[bear] = -1.0
    return sc

def make_ratio(close_df: pd.DataFrame, num: str, den: str) -> pd.Series:
    if num not in close_df.columns or den not in close_df.columns:
        return pd.Series(dtype=float)
    return (close_df[num] / close_df[den]).replace([np.inf, -np.inf], np.nan).dropna()

def build_canary_composite(close_df: pd.DataFrame):
    scores = {}
    for name, cfg in CANARY_RATIO_CONFIG.items():
        ratio = make_ratio(close_df, cfg["num"], cfg["den"])
        ind = indicator_pack_ratio(ratio)
        sc = score_from_indicators_ratio(ind)
        if sc.empty:
            continue
        if cfg["invert"]:
            sc = -sc
        scores[name] = sc
    if not scores:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=bool), pd.Series(dtype=bool)
    common_idx = None
    for sc in scores.values():
        common_idx = sc.index if common_idx is None else common_idx.intersection(sc.index)
    if common_idx is None or len(common_idx) == 0:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=bool), pd.Series(dtype=bool)
    scores_df = pd.DataFrame({k: v.loc[common_idx] for k, v in scores.items()}).dropna()
    if scores_df.empty:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=bool), pd.Series(dtype=bool)
    weights = pd.Series({k: CANARY_RATIO_CONFIG[k]["weight"] for k in scores_df.columns})
    weights = weights / weights.sum()
    composite = (scores_df * weights).sum(axis=1)
    comp_sign = np.sign(composite.replace(0, np.nan))
    align = (np.sign(scores_df).replace(0, np.nan).eq(comp_sign, axis=0)).mean(axis=1).fillna(0)
    strength = scores_df.abs().mean(axis=1)
    confidence = ((0.6 * align) + (0.4 * strength)) * 100.0
    credit_gate = scores_df.get("HYG:SHY (Credit)", pd.Series(index=scores_df.index, data=np.nan)) > 0
    stress_gate = scores_df.get("SPXS:SVOL (Stress/Carry)", pd.Series(index=scores_df.index, data=np.nan)) > 0
    return scores_df, composite, confidence, credit_gate.fillna(False), stress_gate.fillna(False)

def attach_canary_overlay(backtest_df: pd.DataFrame, years: int = 3, buy_thr: float = 0.05, conf_thr: float = 55.0, trend_window: int = 3):
    try:
        close_df = fetch_canary_prices(CANARY_TICKERS, years=years)
    except Exception as e:
        return backtest_df.copy(), {"enabled": False, "warning": f"Canary fetch failed: {e}"}, pd.DataFrame()
    scores_df, composite, confidence, credit_gate, stress_gate = build_canary_composite(close_df)
    if composite.empty:
        return backtest_df.copy(), {"enabled": False, "warning": "Canary overlay unavailable; insufficient ratio history."}, pd.DataFrame()
    overlay = pd.DataFrame({
        "date": pd.to_datetime(composite.index),
        "canary_comp": composite.values,
        "canary_conf": confidence.reindex(composite.index).values,
        "canary_credit": credit_gate.reindex(composite.index).astype(int).values,
        "canary_stress": stress_gate.reindex(composite.index).astype(int).values,
    }).sort_values("date")
    overlay["canary_trend"] = overlay["canary_comp"] - overlay["canary_comp"].shift(trend_window)
    overlay["canary_ok_raw"] = (overlay["canary_comp"] > buy_thr) & (overlay["canary_trend"] > 0) & (overlay["canary_conf"] >= conf_thr) & (overlay["canary_credit"] == 1) & (overlay["canary_stress"] == 1)
    x = pd.merge_asof(backtest_df.sort_values("date"), overlay, on="date", direction="backward")
    x["canary_ok_raw"] = x["canary_ok_raw"].fillna(False)
    x["held_blended"] = ((x["regime"] > 0) & x["canary_ok_raw"]).shift(1).fillna(False).astype(int)
    return x, {"enabled": True, "warning": None, "scores_df": scores_df, "overlay": overlay}, scores_df.tail(1).T.reset_index().rename(columns={"index":"Ratio", scores_df.index[-1]:"Signal"})
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

def make_symbol_chart(hist_feat: pd.DataFrame, sym: str, bb_lift: float, realtime_overrides: Optional[Dict[str, Dict[str, float]]] = None):
    g = hist_feat[hist_feat["symbol"] == sym].sort_values("date").copy()
    if g.empty:
        return None, None
    if realtime_overrides and sym in realtime_overrides:
        vals = realtime_overrides[sym]
        g.iloc[-1, g.columns.get_loc("close")] = vals["close"]
        g.iloc[-1, g.columns.get_loc("high")] = max(g.iloc[-1]["high"], vals["close"])
        g.iloc[-1, g.columns.get_loc("low")] = min(g.iloc[-1]["low"], vals["close"])
        g["rsi14"] = rsi(g["close"], 14)
        g["cci20"] = cci(g["high"], g["low"], g["close"], 20)
        g["pct_b20"] = percent_b(g["close"], 20, 2.0)
        g["ma20"] = g["close"].rolling(20).mean()
        g["ma50"] = g["close"].rolling(50).mean()
    g["candle_pattern"] = g.apply(detect_candle_pattern, axis=1)
    last = g.iloc[-1]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.62, 0.38], specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    fig.add_trace(go.Candlestick(x=g["date"], open=g["open"], high=g["high"], low=g["low"], close=g["close"], name=sym), row=1, col=1)
    if g["ma20"].notna().any():
        fig.add_trace(go.Scatter(x=g["date"], y=g["ma20"], name="MA20", line=dict(width=1.6)), row=1, col=1)
    if g["ma50"].notna().any():
        fig.add_trace(go.Scatter(x=g["date"], y=g["ma50"], name="MA50", line=dict(width=1.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=g["date"], y=g["pct_b20"], name="%B(20,2)", line=dict(width=2.1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=g["date"], y=g["rsi14"], name="RSI14", visible="legendonly"), row=2, col=1)
    fig.add_trace(go.Scatter(x=g["date"], y=g["cci20"], name="CCI20", visible="legendonly"), row=2, col=1)
    fig.add_hline(y=1.0, line_dash="dash", row=2, col=1)
    fig.add_hline(y=0.5, line_dash="dot", row=2, col=1)
    fig.add_hline(y=bb_lift, line_dash="dot", annotation_text=f"%B lift {bb_lift:.2f}", row=2, col=1)
    fig.add_hline(y=0.0, line_dash="dash", row=2, col=1)
    fig.update_layout(
        height=560, xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", y=1.05, x=0),
        template="plotly_dark",
    )
    diag = {
        "Date": str(last["date"].date()),
        "Close": None if pd.isna(last["close"]) else round(float(last["close"]), 2),
        "Bollinger %B(20,2)": None if pd.isna(last["pct_b20"]) else round(float(last["pct_b20"]), 2),
        "RSI14": None if pd.isna(last["rsi14"]) else round(float(last["rsi14"]), 2),
        "CCI20": None if pd.isna(last["cci20"]) else round(float(last["cci20"]), 2),
        "ROC3": None if pd.isna(last.get("roc3")) else round(float(last["roc3"]), 2),
        "Candle": last["candle_pattern"],
    }
    return fig, diag

def build_historical_score_series(hist_feat: pd.DataFrame, thresholds: Dict[str, float], use_proxy_backtest: bool) -> pd.DataFrame:
    dates = sorted(pd.to_datetime(hist_feat["date"].dropna().unique()))
    rows = []
    for dt in dates:
        g = hist_feat[hist_feat["date"] == dt]
        snap = {}
        for _, row in g.iterrows():
            sym = row["symbol"]
            write_snapshot_fields(snap, sym, {
                "close": safe_float(row.get("close")),
                "pct_b20": safe_float(row.get("pct_b20")),
                "rsi14": safe_float(row.get("rsi14")),
                "cci20": safe_float(row.get("cci20")),
                "tsi_fast": safe_float(row.get("tsi_fast")),
                "roc3": safe_float(row.get("roc3")),
                "slope3": safe_float(row.get("slope3")),
            })
        if "$BPSPX" not in snap or "$SPXA50R" not in snap:
            continue
        prev_snap = rows[-1]["snapshot"] if rows else {}
        nymo_eff = get_nymo_effective(snap, prev_snap, use_proxy_backtest)
        _, setup = setup_score_components(snap, prev_snap, thresholds)
        _, conf = confirmation_score_components(snap, prev_snap, thresholds, nymo_eff)
        _, regime = regime_score_components(snap, prev_snap, thresholds, nymo_eff)
        total = int(min(TOTAL_MAX, setup + conf + regime))
        rows.append({
            "date": pd.Timestamp(dt),
            "setup_score": setup,
            "confirmation_score": conf,
            "regime_score": regime,
            "breadth_score": total,
            "RSP": safe_float(snap.get("RSP", np.nan)),
            "SPY": safe_float(snap.get("SPY", np.nan)),
            "snapshot": snap,
        })
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return out.drop(columns=["snapshot"], errors="ignore")

def build_oscillator(df: pd.DataFrame, fast: int, slow: int, signal: int, deadband: float) -> pd.DataFrame:
    x = df.copy()
    x["ema_fast"] = ema(x["breadth_score"], fast)
    x["ema_slow"] = ema(x["breadth_score"], slow)
    denom = x["ema_slow"].abs().replace(0, np.nan)
    x["osc_raw"] = 100 * (x["ema_fast"] - x["ema_slow"]) / denom
    x["osc"] = ema(x["osc_raw"], signal)
    x["osc_signal"] = ema(x["osc"], signal)
    x["regime"] = np.where(x["osc"] > deadband, 1, 0)
    x["held"] = x["regime"].shift(1).fillna(0)
    return x

def run_backtest(df: pd.DataFrame, asset_col: str = "RSP", held_col: str = "held", switch_cost_bps: float = 0.0):
    x = df.copy()
    if asset_col not in x.columns:
        raise ValueError(f"Missing asset column for backtest: {asset_col}")
    x["asset_ret"] = x[asset_col].pct_change().fillna(0)
    x["switch"] = x[held_col].diff().abs().fillna(0)
    x["cost"] = (switch_cost_bps / 10000.0) * x["switch"]
    x["strategy_ret"] = x[held_col] * x["asset_ret"] - x["cost"]
    x["equity_strategy"] = (1 + x["strategy_ret"]).cumprod()
    x["equity_buyhold"] = (1 + x["asset_ret"]).cumprod()
    dd_strategy = (x["equity_strategy"] / x["equity_strategy"].cummax()) - 1
    dd_bh = (x["equity_buyhold"] / x["equity_buyhold"].cummax()) - 1
    stats = {
        "Strategy Return %": round(float((x["equity_strategy"].iloc[-1] - 1) * 100), 2) if len(x) else np.nan,
        "BuyHold Return %": round(float((x["equity_buyhold"].iloc[-1] - 1) * 100), 2) if len(x) else np.nan,
        "Strategy Max DD %": round(float(dd_strategy.min() * 100), 2) if len(x) else np.nan,
        "BuyHold Max DD %": round(float(dd_bh.min() * 100), 2) if len(x) else np.nan,
        "Switches": int(x["switch"].sum()) if len(x) else 0,
        "Exposure %": round(float(100 * x[held_col].mean()), 1) if len(x) else np.nan,
    }
    return x, stats

def metric_card(title: str, value: str, subtitle: str = "", tone: str = "blue", pct: float = 0.0):
    fill_cls = {"green":"score-fill-green","yellow":"score-fill-yellow","red":"score-fill-red","blue":"score-fill-blue"}.get(tone,"score-fill-blue")
    pct = max(0.0, min(100.0, pct))
    st.markdown(
        f"""
        <div class="soft-card score-card">
          <div>
            <div class="score-title">{title}</div>
            <div class="score-value">{value}</div>
            <div class="score-subtitle">{subtitle}</div>
          </div>
          <div>
            <div class="score-bar"><div class="score-fill {fill_cls}" style="width:{pct}%;"></div></div>
            <div class="small-muted" style="margin-top:.45rem;">{pct:.0f}% of max score</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_box(label: str, value: str):
    st.markdown(
        f"""
        <div class="kpi-box">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def score_tone(score: int) -> str:
    if score >= 72:
        return "green"
    if score >= 50:
        return "blue"
    if score >= 30:
        return "yellow"
    return "red"

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Baseline / Uploads")
    baseline_file = st.file_uploader("Upload historical baseline CSV", type=["csv"])
    realtime_file = st.file_uploader("Upload realtime / EOD snapshot CSV", type=["csv"])
    upload_tag = st.text_input("Upload tag", value="manual upload")

    st.markdown("---")
    st.subheader("Signal Governance")
    use_proxy_mode = st.toggle("Use NYMO proxy mode", value=True, help="Uses (NYAD×0.6 + SPXADP×0.4) compressed to a NYMO-like range for intraday reads.")
    enforce_guardrail = st.toggle("Enforce chronology guardrail", value=True)

    st.markdown("---")
    st.subheader("Restore Prior Snapshot")
    history_df_for_select = load_upload_history()
    selected_snapshot_ts = None
    if not history_df_for_select.empty and "upload_ts" in history_df_for_select.columns:
        options = ["Latest saved snapshot"] + history_df_for_select.sort_values("upload_ts", ascending=False)["upload_ts"].astype(str).tolist()
        selected_snapshot_ts = st.selectbox("Load saved snapshot", options=options, index=0)

    st.markdown("---")
    st.subheader("Thresholds")
    thresholds = DEFAULT_THRESHOLDS.copy()
    thresholds["BPSPX_BB_LIFT"] = st.number_input("BPSPX %B repair lift", value=float(thresholds["BPSPX_BB_LIFT"]), step=0.01, format="%.2f")
    thresholds["SPXA50R_REPAIR"] = st.number_input("SPXA50R repair threshold", value=float(thresholds["SPXA50R_REPAIR"]), step=1.0, format="%.1f")
    thresholds["BPSPX_CONFIRM"] = st.number_input("BPSPX confirm threshold", value=float(thresholds["BPSPX_CONFIRM"]), step=1.0, format="%.1f")
    thresholds["NYMO_HEALTHY"] = st.number_input("NYMO healthy threshold", value=float(thresholds["NYMO_HEALTHY"]), step=5.0, format="%.1f")
    thresholds["NYSI_LEVERAGE_OK"] = st.number_input("NYSI leverage threshold", value=float(thresholds["NYSI_LEVERAGE_OK"]), step=5.0, format="%.1f")
    thresholds["VXX_BB_PROBE_MAX"] = st.number_input("VXX %B probe max", value=float(thresholds["VXX_BB_PROBE_MAX"]), step=0.05, format="%.2f")

    st.markdown("---")
    st.subheader("Oscillator Backtest")
    fast_ema = st.number_input("Fast EMA", min_value=2, max_value=50, value=5, step=1)
    slow_ema = st.number_input("Slow EMA", min_value=3, max_value=80, value=13, step=1)
    signal_ema = st.number_input("Signal EMA", min_value=2, max_value=30, value=5, step=1)
    deadband = st.number_input("Deadband", value=0.0, step=0.1, format="%.2f")
    switch_cost_bps = st.number_input("Switch cost (bps)", value=0.0, step=1.0, format="%.1f")
    use_proxy_backtest = st.toggle("Use proxy logic in backtest", value=False)
    use_canary_overlay = st.toggle("Blend canary overlay into Tab 2", value=True, help="Adds a ratio-canary regime filter to the breadth oscillator backtest and trades RSP instead of SPY.")
    canary_buy_thr = st.number_input("Canary composite buy threshold", value=0.05, step=0.01, format="%.2f")
    canary_conf_thr = st.number_input("Canary confidence threshold", value=55.0, step=1.0, format="%.1f")
    canary_trend_window = st.number_input("Canary trend window", min_value=1, max_value=10, value=3, step=1)

    st.markdown("---")
    st.subheader("Decision Charts")
    show_charts = st.multiselect("Symbols", options=CHART_SYMBOLS, default=["$BPSPX", "$SPXA50R", "$NYMO", "$NYSI", "RSP"])

# -----------------------------
# Baseline save/load
# -----------------------------
baseline_meta = load_json(BASELINE_META_PATH, {})
if baseline_file is not None:
    parsed_baseline = parse_stockcharts_historical_from_bytes(baseline_file.getvalue())
    parsed_baseline.to_parquet(BASELINE_PATH, index=False)
    baseline_meta = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "source_filename": baseline_file.name,
        "rows": int(len(parsed_baseline)),
        "symbols": int(parsed_baseline["symbol"].nunique())
    }
    save_json(BASELINE_META_PATH, baseline_meta)
    st.success("Historical baseline saved.")

if not BASELINE_PATH.exists():
    st.info("Upload and save the historical baseline first.")
    st.stop()

# -----------------------------
# Load data
# -----------------------------
hist_long = pd.read_parquet(BASELINE_PATH)
hist_feat = add_indicator_features(hist_long)
baseline_snapshot = get_feature_snapshot(hist_feat, "latest")
baseline_prior_snapshot = get_feature_snapshot(hist_feat, "prior")
upload_history_df = load_upload_history()

snapshot = baseline_snapshot.copy()
prev_snapshot = baseline_prior_snapshot.copy()
realtime_df = pd.DataFrame()
realtime_overrides = {}

if realtime_file is not None:
    realtime_df = parse_realtime_snapshot_from_bytes(realtime_file.getvalue())
    saved_path = save_snapshot_file(realtime_df)
else:
    if selected_snapshot_ts and selected_snapshot_ts != "Latest saved snapshot":
        row = upload_history_df[upload_history_df["upload_ts"].astype(str) == str(selected_snapshot_ts)]
        if not row.empty and "snapshot_file" in row.columns:
            realtime_df = load_selected_snapshot(str(row.iloc[0]["snapshot_file"]))
    else:
        realtime_df = load_latest_saved_snapshot()

if not realtime_df.empty:
    realtime_overrides = recompute_latest_indicators_from_realtime(hist_feat, realtime_df)
    for sym, vals in realtime_overrides.items():
        write_snapshot_fields(snapshot, sym, vals)

nymo_effective = get_nymo_effective(snapshot, prev_snapshot, use_proxy_mode)
setup_df, setup_score = setup_score_components(snapshot, prev_snapshot, thresholds)
confirm_df, confirmation_score = confirmation_score_components(snapshot, prev_snapshot, thresholds, nymo_effective)
regime_df, regime_score = regime_score_components(snapshot, prev_snapshot, thresholds, nymo_effective)
total_score = int(min(TOTAL_MAX, setup_score + confirmation_score + regime_score))
guardrail_status, guardrail_msg = chronology_guardrail(upload_history_df, snapshot, prev_snapshot)
if not enforce_guardrail:
    guardrail_status, guardrail_msg = "Normal", "Guardrail disabled."
action_hierarchy = model_action_hierarchy(total_score, setup_score, confirmation_score, regime_score, snapshot, thresholds, nymo_effective, guardrail_status)
core_momentum_df = build_momentum_context_table(["$BPSPX_%B", "$BPSPX", "$SPXA50R", "$NYMO", "$NYSI", "$NYAD", "$SPXADP", "$CPCE", "RSP:SPY", "VXX", "VXX_%B"], snapshot, prev_snapshot, thresholds)
extended_momentum_df = build_momentum_context_table(["RSP", "SPY", "URSP"], snapshot, prev_snapshot, thresholds)
bounce_quality_df = compute_bounce_quality(snapshot, prev_snapshot, thresholds, nymo_effective)
bounce_quality_score = int(bounce_quality_df["Score"].sum())
recent_uploads_df = latest_upload_rows(5)
checklist_text = dynamic_trading_checklist(snapshot, thresholds, upload_history_df, action_hierarchy, guardrail_msg, nymo_effective)

if realtime_file is not None and not realtime_df.empty:
    stage_num, stage_name = breadth_stage(snapshot, thresholds, nymo_effective.get("value", np.nan))
    append_upload_history({
        "snapshot_file": str(saved_path),
        "upload_ts": datetime.now().isoformat(timespec="seconds"),
        "tag": upload_tag,
        "BPSPX": snapshot.get("$BPSPX", np.nan),
        "BPSPX_%B": snapshot.get("$BPSPX_%B", np.nan),
        "SPXA50R": snapshot.get("$SPXA50R", np.nan),
        "ConfluenceScore": total_score,
        "Stage": stage_num,
        "StageName": stage_name,
        "RepairTrigger": qwen_repair_trigger(snapshot, thresholds),
        "BounceQuality": bounce_quality_score,
    })
    upload_history_df = load_upload_history()
    recent_uploads_df = latest_upload_rows(5)

score_series = build_historical_score_series(hist_feat, thresholds, use_proxy_backtest)
if not score_series.empty:
    max_date = score_series["date"].max()
    score_series = score_series[score_series["date"] >= (max_date - pd.Timedelta(days=365))].reset_index(drop=True)
    score_series = build_oscillator(score_series, fast_ema, slow_ema, signal_ema, deadband)
    backtest_df, stats = run_backtest(score_series, asset_col="RSP", held_col="held", switch_cost_bps=switch_cost_bps)
    if use_canary_overlay and not backtest_df.empty:
        backtest_df, canary_meta, canary_latest_components = attach_canary_overlay(backtest_df, years=3, buy_thr=canary_buy_thr, conf_thr=canary_conf_thr, trend_window=canary_trend_window)
        if canary_meta.get("enabled"):
            backtest_df["switch_blended"] = backtest_df["held_blended"].diff().abs().fillna(0)
            backtest_df["cost_blended"] = (switch_cost_bps / 10000.0) * backtest_df["switch_blended"]
            backtest_df["strategy_ret_blended"] = backtest_df["held_blended"] * backtest_df["asset_ret"] - backtest_df["cost_blended"]
            backtest_df["equity_blended"] = (1 + backtest_df["strategy_ret_blended"]).cumprod()
            dd_blended = (backtest_df["equity_blended"] / backtest_df["equity_blended"].cummax()) - 1
            blended_stats = {
                "Blended Return %": round(float((backtest_df["equity_blended"].iloc[-1] - 1) * 100), 2),
                "Blended Max DD %": round(float(dd_blended.min() * 100), 2),
                "Blended Switches": int(backtest_df["switch_blended"].sum()),
                "Blended Exposure %": round(float(100 * backtest_df["held_blended"].mean()), 1),
            }
        else:
            blended_stats = {}
    else:
        canary_meta = {"enabled": False, "warning": None}
        canary_latest_components = pd.DataFrame()
        blended_stats = {}
else:
    backtest_df = pd.DataFrame()
    stats = {}
    canary_meta = {"enabled": False, "warning": None}
    canary_latest_components = pd.DataFrame()
    blended_stats = {}

# -----------------------------
# Main UI
# -----------------------------
tab1, tab2 = st.tabs(["Decision Dashboard", "Score Oscillator Backtest"])

with tab1:
    row1c1, row1c2 = st.columns(2)
    with row1c1:
        metric_card("Setup", f"{setup_score}/{SETUP_MAX}", "washout + lift", score_tone(setup_score * 2), (setup_score / SETUP_MAX) * 100)
    with row1c2:
        metric_card("Confirmation", f"{confirmation_score}/{CONFIRM_MAX}", "repair quality", score_tone(confirmation_score * 2), (confirmation_score / CONFIRM_MAX) * 100)

    row2c1, row2c2 = st.columns(2)
    with row2c1:
        metric_card("Regime", f"{regime_score}/{REGIME_MAX}", "durability", score_tone(regime_score * 2), (regime_score / REGIME_MAX) * 100)
    with row2c2:
        metric_card("Total", f"{total_score}/{TOTAL_MAX}", f"bounce quality {bounce_quality_score}/10", score_tone(total_score), (total_score / TOTAL_MAX) * 100)

    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">At-a-Glance Numbers</div>', unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3)
    stage_num, stage_name = breadth_stage(snapshot, thresholds, nymo_effective.get("value", np.nan))
    with k1:
        kpi_box("Bounce Quality", f"{bounce_quality_score}/10")
    with k2:
        kpi_box("NYMO Source", nymo_effective["label"] + f": {fmt_num(nymo_effective['value'])}")
    with k3:
        kpi_box("Stage", f"{stage_num} - {stage_name}")
    st.markdown('</div>', unsafe_allow_html=True)

    a1, a2 = st.columns([0.62, 0.38])
    with a1:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Action Hierarchy</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="action-box action-existing"><b>Existing:</b> {action_hierarchy['existing']}</div>
        <div class="action-box action-new"><b>New:</b> {action_hierarchy['new']}</div>
        <div class="action-box action-add"><b>Add:</b> {action_hierarchy['add']}</div>
        <div class="small-muted" style="margin-top:.55rem; font-size:1rem;">{action_hierarchy['rationale']}</div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="kpi-row">', unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        with s1:
            kpi_box("Suggested RSP", f"{int(action_hierarchy['rsp_size']*100)}%")
        with s2:
            kpi_box("Suggested URSP", f"{int(action_hierarchy['ursp_size']*100)}%")
        with s3:
            kpi_box("Stage", f"{stage_num} - {stage_name}")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("")
        st.markdown('<div class="soft-card big-code">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Trading Checklist</div>', unsafe_allow_html=True)
        st.code(checklist_text, language="text")
        st.markdown('</div>', unsafe_allow_html=True)

    with a2:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Breadth Source & Guardrails</div>', unsafe_allow_html=True)
        tone = "pill-green" if guardrail_status == "Normal" else "pill-yellow"
        st.markdown(f"""
        <span class="pill pill-blue">{nymo_effective['mode']}</span>
        <span class="pill {tone}">{guardrail_status}</span>
        <span class="pill pill-blue">{nymo_effective['label']}: {fmt_num(nymo_effective['value'])}</span>
        """, unsafe_allow_html=True)
        st.write(guardrail_msg)
        st.write(f"Proxy/official state: **{nymo_effective['state']}**")
        st.subheader("Current Input Snapshot")
        input_rows = []
        for sym in DECISION_SYMBOLS:
            val = snapshot.get(sym, np.nan)
            if pd.notna(val):
                input_rows.append({"Symbol": sym, "Value": round(float(val), 4)})
        st.dataframe(pd.DataFrame(input_rows), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    r1, r2, r3 = st.columns(3)
    with r1:
        st.subheader("Setup Components")
        st.dataframe(setup_df, use_container_width=True, hide_index=True)
    with r2:
        st.subheader("Confirmation Components")
        st.dataframe(confirm_df, use_container_width=True, hide_index=True)
    with r3:
        st.subheader("Regime Components")
        st.dataframe(regime_df, use_container_width=True, hide_index=True)

    b1, b2 = st.columns([0.58, 0.42])
    with b1:
        st.subheader("Momentum Context — Core")
        st.dataframe(core_momentum_df, use_container_width=True, hide_index=True)
    with b2:
        st.subheader("Bounce Quality (0-10)")
        st.dataframe(bounce_quality_df, use_container_width=True, hide_index=True)

    st.subheader("Momentum Context — Extended")
    st.dataframe(extended_momentum_df, use_container_width=True, hide_index=True)

    st.subheader("Current Realtime Snapshot")
    if realtime_df.empty:
        st.info("No new realtime snapshot uploaded this run. Latest saved snapshot restored if available.")
    else:
        show_cols = [c for c in ["Symbol", "Close", "PctChange"] if c in realtime_df.columns]
        st.dataframe(realtime_df[show_cols], use_container_width=True, hide_index=True)

    if not recent_uploads_df.empty:
        st.subheader("Upload History")
        display_cols = [c for c in ["upload_ts", "tag", "Stage", "StageName", "ConfluenceScore", "BounceQuality", "snapshot_file"] if c in recent_uploads_df.columns]
        st.dataframe(recent_uploads_df[display_cols], use_container_width=True, hide_index=True)

    st.subheader("Decision Charts")
    for sym in show_charts:
        fig, diag = make_symbol_chart(hist_feat, sym, thresholds["BPSPX_BB_LIFT"], realtime_overrides=realtime_overrides)
        if fig is None:
            continue
        with st.expander(sym, expanded=sym in ["$BPSPX", "$SPXA50R"]):
            l, r = st.columns([1.65, 0.35])
            with l:
                st.plotly_chart(fig, use_container_width=True)
            with r:
                st.json(diag)

with tab2:
    st.subheader("1-Year Breadth Score Oscillator Backtest")
    if backtest_df.empty:
        st.info("Not enough historical data to build the backtest.")
    else:
        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("Strategy Return", f"{stats.get('Strategy Return %', np.nan)}%")
        p2.metric("Buy & Hold Return", f"{stats.get('BuyHold Return %', np.nan)}%")
        p3.metric("Strategy Max DD", f"{stats.get('Strategy Max DD %', np.nan)}%")
        p4.metric("Switches", f"{stats.get('Switches', 0)}")
        p5.metric("Exposure", f"{stats.get('Exposure %', np.nan)}%")

        fig_price = go.Figure()
        asset_label = "RSP" if "RSP" in backtest_df.columns else ("SPY" if "SPY" in backtest_df.columns else None)
        if asset_label is not None:
            fig_price.add_trace(go.Scatter(x=backtest_df["date"], y=backtest_df[asset_label], name=asset_label, line=dict(width=2.2)))
        fig_price.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), title=f"{asset_label or 'Asset'}", template="plotly_dark")

        fig_osc = go.Figure()
        fig_osc.add_trace(go.Scatter(x=backtest_df["date"], y=backtest_df["osc"], name="Breadth Oscillator", line=dict(width=2.2)))
        fig_osc.add_trace(go.Scatter(x=backtest_df["date"], y=backtest_df["osc_signal"], name="Signal", visible="legendonly"))
        fig_osc.add_hline(y=0.0, line_dash="dash")
        fig_osc.add_hline(y=deadband, line_dash="dot", annotation_text=f"deadband {deadband:.2f}")
        fig_osc.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=10), title="PPO-Style Breadth Oscillator", template="plotly_dark")

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=backtest_df["date"], y=backtest_df["equity_strategy"], name="Strategy", line=dict(width=2.4)))
        buyhold_col = "equity_buyhold" if "equity_buyhold" in backtest_df.columns else ("equity_spy" if "equity_spy" in backtest_df.columns else None)
        if buyhold_col is not None:
            fig_eq.add_trace(go.Scatter(x=backtest_df["date"], y=backtest_df[buyhold_col], name="Buy & Hold", line=dict(width=2.0)))
        fig_eq.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), title="Equity Curves", template="plotly_dark")

        st.plotly_chart(fig_price, use_container_width=True)
        st.plotly_chart(fig_osc, use_container_width=True)
        st.plotly_chart(fig_eq, use_container_width=True)

        stat_rows = [{"Metric": k, "Value": v} for k, v in stats.items()]
        st.subheader("Backtest Stats")
        st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)
        st.subheader("Breadth Score History")
        st.dataframe(backtest_df[["date", "setup_score", "confirmation_score", "regime_score", "breadth_score", "osc", "held"]], use_container_width=True, hide_index=True)

st.subheader("Stored Baseline Status")
st.json(baseline_meta if baseline_meta else {"status": "No metadata found"})
st.subheader("Downloads")
parsed_csv = hist_feat.to_csv(index=False).encode("utf-8")
st.download_button("Download parsed historical features", data=parsed_csv, file_name="parsed_breadth_features.csv", mime="text/csv")
upload_hist_df = load_upload_history()
if not upload_hist_df.empty:
    st.download_button("Download upload history", data=upload_hist_df.to_csv(index=False).encode("utf-8"), file_name="breadth_upload_history.csv", mime="text/csv")

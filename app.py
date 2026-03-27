

import io
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="RSP / URSP Breadth Model", layout="wide")
st.title("RSP / URSP Breadth Model")
st.caption("Decision dashboard + 1-year breadth score oscillator backtest")

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

DECISION_SYMBOLS = [
    "$BPSPX", "$SPXA50R", "$NYMO", "$NYSI", "$NYAD", "$SPXADP",
    "$CPCE", "$NYHL", "RSP:SPY", "VXX", "RSP", "URSP", "SPY",
]
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
}
SETUP_MAX = 35
CONFIRM_MAX = 35
REGIME_MAX = 30
TOTAL_MAX = 100
DEFAULT_MINIMAL_CHARTS = ["$BPSPX", "$SPXA50R", "$NYMO", "$NYSI", "RSP"]
CHART_SYMBOLS = ["$BPSPX", "$SPXA50R", "$NYMO", "$NYSI", "RSP", "SPY", "VXX"]


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


def true_strength_index(series: pd.Series, long: int = 25, short: int = 13, signal: int = 7):
    m = series.diff()
    a = m.abs()
    m1 = ema(ema(m, long), short)
    a1 = ema(ema(a, long), short)
    tsi = 100 * (m1 / a1.replace(0, np.nan))
    sig = ema(tsi, signal)
    return tsi, sig


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
        g["tsi_fast"], g["tsi_fast_sig"] = true_strength_index(g["close"], 4, 2, 4)
        g["ma20"] = g["close"].rolling(20).mean()
        g["ma50"] = g["close"].rolling(50).mean()
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def get_latest_feature_snapshot(hist_feat: pd.DataFrame) -> Dict[str, float]:
    snapshot = {}
    latest_rows = hist_feat.sort_values("date").groupby("symbol", as_index=False).tail(1)
    for _, row in latest_rows.iterrows():
        sym = row["symbol"]
        snapshot[sym] = safe_float(row.get("close"))
        snapshot[f"{sym}_%B"] = safe_float(row.get("pct_b20"))
        snapshot[f"{sym}_RSI14"] = safe_float(row.get("rsi14"))
        snapshot[f"{sym}_CCI20"] = safe_float(row.get("cci20"))
        snapshot[f"{sym}_TSI"] = safe_float(row.get("tsi_fast"))
    return snapshot


def get_prior_feature_snapshot(hist_feat: pd.DataFrame) -> Dict[str, float]:
    snapshot = {}
    for sym, g in hist_feat.groupby("symbol", sort=False):
        g = g.sort_values("date")
        if len(g) < 2:
            continue
        row = g.iloc[-2]
        snapshot[sym] = safe_float(row.get("close"))
        snapshot[f"{sym}_%B"] = safe_float(row.get("pct_b20"))
        snapshot[f"{sym}_RSI14"] = safe_float(row.get("rsi14"))
        snapshot[f"{sym}_CCI20"] = safe_float(row.get("cci20"))
        snapshot[f"{sym}_TSI"] = safe_float(row.get("tsi_fast"))
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
        g["tsi_fast"], g["tsi_fast_sig"] = true_strength_index(g["close"], 4, 2, 4)
        last = g.iloc[-1]
        out[sym] = {
            "close": safe_float(last["close"]),
            "pct_b20": safe_float(last["pct_b20"]),
            "rsi14": safe_float(last["rsi14"]),
            "cci20": safe_float(last["cci20"]),
            "tsi_fast": safe_float(last["tsi_fast"]),
        }
    return out


def qwen_repair_trigger(snapshot: Dict[str, float], thresholds: Dict[str, float]) -> bool:
    bpspx_bb = snapshot.get("$BPSPX_%B", np.nan)
    spxa50r = snapshot.get("$SPXA50R", np.nan)
    return bool(pd.notna(bpspx_bb) and pd.notna(spxa50r) and bpspx_bb > thresholds["BPSPX_BB_LIFT"] and spxa50r > thresholds["SPXA50R_REPAIR"])


def breadth_stage(snapshot: Dict[str, float], thresholds: Dict[str, float]) -> Tuple[int, str]:
    bpspx_bb = safe_float(snapshot.get("$BPSPX_%B", np.nan))
    spxa50r = safe_float(snapshot.get("$SPXA50R", np.nan))
    bpspx = safe_float(snapshot.get("$BPSPX", np.nan))
    nysi = safe_float(snapshot.get("$NYSI", np.nan))
    if pd.isna(bpspx_bb):
        return 0, "Unknown"
    if bpspx_bb < thresholds["BPSPX_BB_LIFT"]:
        return 0, "Probe / lower-band pressure"
    if bpspx_bb >= thresholds["BPSPX_BB_LIFT"] and spxa50r <= thresholds["SPXA50R_REPAIR"]:
        return 1, "Bounce starting"
    if bpspx_bb > thresholds["BPSPX_BB_LIFT"] and spxa50r > thresholds["SPXA50R_REPAIR"] and bpspx <= thresholds["BPSPX_CONFIRM"]:
        return 2, "Repair building"
    if bpspx > thresholds["BPSPX_CONFIRM"] and nysi <= thresholds["NYSI_POSITIVE"]:
        return 3, "Participation confirmation"
    if bpspx > thresholds["BPSPX_CONFIRM"] and nysi > thresholds["NYSI_POSITIVE"]:
        return 4, "Durable up-regime"
    return 0, "Unknown"


def setup_score_components(snapshot: Dict[str, float], thresholds: Dict[str, float]):
    rows = []
    pts = 0
    bpspx_bb = safe_float(snapshot.get("$BPSPX_%B", np.nan))
    spxa50r = safe_float(snapshot.get("$SPXA50R", np.nan))
    cpce = safe_float(snapshot.get("$CPCE", np.nan))
    vxx_bb = safe_float(snapshot.get("VXX_%B", np.nan))
    nyad = safe_float(snapshot.get("$NYAD", np.nan))
    rsp_spy = safe_float(snapshot.get("RSP:SPY", np.nan))
    score = 10 if pd.notna(bpspx_bb) and bpspx_bb < 0.20 else 5 if pd.notna(bpspx_bb) and bpspx_bb < 0.35 else 0
    pts += score
    rows.append({"Component": "BPSPX %B washout", "Current": fmt_num(bpspx_bb), "Target": "< 0.20", "Score": score})
    score = 8 if pd.notna(spxa50r) and spxa50r < 30 else 4 if pd.notna(spxa50r) and spxa50r < 35 else 0
    pts += score
    rows.append({"Component": "SPXA50R weak depth", "Current": fmt_num(spxa50r), "Target": "< 30", "Score": score})
    score = 5 if pd.notna(cpce) and cpce >= thresholds["CPCE_FEAR"] else 0
    pts += score
    rows.append({"Component": "CPCE fear support", "Current": fmt_num(cpce), "Target": f">= {thresholds['CPCE_FEAR']:.2f}", "Score": score})
    score = 5 if pd.notna(vxx_bb) and vxx_bb < thresholds["VXX_BB_PROBE_MAX"] else 0
    pts += score
    rows.append({"Component": "VXX %B contained", "Current": fmt_num(vxx_bb), "Target": f"< {thresholds['VXX_BB_PROBE_MAX']:.2f}", "Score": score})
    score = 4 if pd.notna(nyad) and nyad > thresholds["NYAD_THRUST"] else 2 if pd.notna(nyad) and nyad > 500 else 0
    pts += score
    rows.append({"Component": "NYAD positive thrust", "Current": fmt_num(nyad, 0), "Target": f"> {thresholds['NYAD_THRUST']:.0f}", "Score": score})
    score = 3 if pd.notna(rsp_spy) else 0
    pts += score
    rows.append({"Component": "RSP:SPY available", "Current": fmt_num(rsp_spy, 3), "Target": "flat / rising", "Score": score})
    return pd.DataFrame(rows), int(min(SETUP_MAX, pts))


def confirmation_score_components(snapshot: Dict[str, float], thresholds: Dict[str, float]):
    rows = []
    pts = 0
    bpspx_bb = safe_float(snapshot.get("$BPSPX_%B", np.nan))
    spxa50r = safe_float(snapshot.get("$SPXA50R", np.nan))
    bpspx = safe_float(snapshot.get("$BPSPX", np.nan))
    nymo = safe_float(snapshot.get("$NYMO", np.nan))
    spxadp = safe_float(snapshot.get("$SPXADP", np.nan))
    rsp_spy = safe_float(snapshot.get("RSP:SPY", np.nan))
    score = 10 if pd.notna(bpspx_bb) and bpspx_bb > thresholds["BPSPX_BB_LIFT"] else 0
    pts += score
    rows.append({"Component": "BPSPX %B repair lift", "Current": fmt_num(bpspx_bb), "Target": f"> {thresholds['BPSPX_BB_LIFT']:.2f}", "Score": score})
    score = 10 if pd.notna(spxa50r) and spxa50r > thresholds["SPXA50R_REPAIR"] else 0
    pts += score
    rows.append({"Component": "SPXA50R > 30", "Current": fmt_num(spxa50r), "Target": f"> {thresholds['SPXA50R_REPAIR']:.0f}", "Score": score})
    score = 6 if pd.notna(bpspx) and bpspx > thresholds["BPSPX_CONFIRM"] else 3 if pd.notna(bpspx) and bpspx > 40 else 0
    pts += score
    rows.append({"Component": "BPSPX participation", "Current": fmt_num(bpspx), "Target": f"> {thresholds['BPSPX_CONFIRM']:.0f}", "Score": score})
    score = 4 if pd.notna(nymo) and nymo > thresholds["NYMO_HEALTHY"] else 2 if pd.notna(nymo) and nymo > -50 else 0
    pts += score
    rows.append({"Component": "NYMO momentum", "Current": fmt_num(nymo), "Target": f"> {thresholds['NYMO_HEALTHY']:.0f}", "Score": score})
    score = 3 if pd.notna(spxadp) and spxadp > thresholds["SPXADP_THRUST"] else 1 if pd.notna(spxadp) and spxadp > 20 else 0
    pts += score
    rows.append({"Component": "SPXADP breadth volume", "Current": fmt_num(spxadp), "Target": f"> {thresholds['SPXADP_THRUST']:.0f}", "Score": score})
    score = 2 if pd.notna(rsp_spy) else 0
    pts += score
    rows.append({"Component": "RSP:SPY no weakness", "Current": fmt_num(rsp_spy, 3), "Target": "flat / rising", "Score": score})
    return pd.DataFrame(rows), int(min(CONFIRM_MAX, pts))


def regime_score_components(snapshot: Dict[str, float], thresholds: Dict[str, float]):
    rows = []
    pts = 0
    nysi = safe_float(snapshot.get("$NYSI", np.nan))
    nyhl = safe_float(snapshot.get("$NYHL", np.nan))
    repair = qwen_repair_trigger(snapshot, thresholds)
    stage_num, stage_name = breadth_stage(snapshot, thresholds)
    score = 12 if pd.notna(nysi) and nysi > thresholds["NYSI_POSITIVE"] else 6 if pd.notna(nysi) and nysi > thresholds["NYSI_LEVERAGE_OK"] else 0
    pts += score
    rows.append({"Component": "NYSI durability", "Current": fmt_num(nysi), "Target": f"> {thresholds['NYSI_POSITIVE']:.0f}", "Score": score})
    score = 8 if repair else 0
    pts += score
    rows.append({"Component": "Repair trigger", "Current": "ON" if repair else "OFF", "Target": "ON", "Score": score})
    score = 5 if pd.notna(nyhl) and nyhl > 0 else 0
    pts += score
    rows.append({"Component": "NYHL positive", "Current": fmt_num(nyhl), "Target": "> 0", "Score": score})
    score = min(5, stage_num)
    pts += score
    rows.append({"Component": "Regime stage", "Current": f"{stage_num} ({stage_name})", "Target": "higher", "Score": score})
    return pd.DataFrame(rows), int(min(REGIME_MAX, pts))


def model_recommendation(total_score: int, setup_score: int, confirmation_score: int, regime_score: int, snapshot: Dict[str, float], thresholds: Dict[str, float]):
    bpspx_bb = safe_float(snapshot.get("$BPSPX_%B", np.nan))
    spxa50r = safe_float(snapshot.get("$SPXA50R", np.nan))
    nysi = safe_float(snapshot.get("$NYSI", np.nan))
    if total_score >= 72 and setup_score >= 20 and confirmation_score >= 24 and regime_score >= 16 and bpspx_bb >= thresholds["BPSPX_BB_LIFT"] and spxa50r >= thresholds["SPXA50R_REPAIR"] and nysi > thresholds["NYSI_LEVERAGE_OK"]:
        return "Aggressive Long / URSP Allowed", "URSP can be used selectively; breadth and regime are strong enough.", 0.35, 0.12
    if total_score >= 50 and setup_score >= 20 and confirmation_score >= 16:
        return "Long Bias / RSP Preferred", "Use RSP sizing first; keep URSP off unless regime improves further.", 0.20, 0.00
    if setup_score >= 15 and confirmation_score < 16:
        return "Probe Only", "Small RSP probe only; wait for confirmation before sizing up.", 0.10, 0.00
    return "Stand Down / Defensive", "Conditions are not strong enough for meaningful long exposure.", 0.00, 0.00


def dynamic_trading_checklist(snapshot: Dict[str, float], thresholds: Dict[str, float], history_df: pd.DataFrame, recommendation: str) -> str:
    bpspx_bb = safe_float(snapshot.get("$BPSPX_%B", np.nan))
    spxa50r = safe_float(snapshot.get("$SPXA50R", np.nan))
    vxx_bb = safe_float(snapshot.get("VXX_%B", np.nan))
    nyad = safe_float(snapshot.get("$NYAD", np.nan))
    rsp_spy = safe_float(snapshot.get("RSP:SPY", np.nan))
    stage_num, stage_name = breadth_stage(snapshot, thresholds)
    week_hist = uploads_in_lookback(history_df, 7)
    spxa50r_intraday_highs = 0
    spxa50r_week_high = np.nan
    if not week_hist.empty and "SPXA50R" in week_hist.columns:
        vals = pd.to_numeric(week_hist["SPXA50R"], errors="coerce")
        spxa50r_intraday_highs = int((vals > thresholds["SPXA50R_REPAIR"]).sum())
        spxa50r_week_high = vals.max()
    full_confirm = pd.notna(bpspx_bb) and bpspx_bb >= thresholds["BPSPX_BB_LIFT"] and pd.notna(spxa50r) and spxa50r >= thresholds["SPXA50R_REPAIR"]
    vxx_ok = pd.notna(vxx_bb) and vxx_bb < thresholds["VXX_BB_PROBE_MAX"]
    nyad_ok = pd.notna(nyad) and nyad > thresholds["NYAD_THRUST"]
    rsp_ok = pd.notna(rsp_spy)
    lines = [f"Current Regime: {recommendation} (Stage {stage_num}: {stage_name})", "", "Context:"]
    if spxa50r_intraday_highs >= 2:
        lines.append(f"• SPXA50R printed {spxa50r_intraday_highs} intraday readings above {thresholds['SPXA50R_REPAIR']:.0f} during the last week")
    elif pd.notna(spxa50r_week_high):
        lines.append(f"• SPXA50R week high is {spxa50r_week_high:.2f}")
    else:
        lines.append("• Limited upload history for intraday threshold context")
    lines += [
        "",
        "Entry Filters:",
        f"• VXX %B < {thresholds['VXX_BB_PROBE_MAX']:.2f} {'✅' if vxx_ok else '❌'}" + (f" (current {vxx_bb:.2f})" if pd.notna(vxx_bb) else ""),
        f"• NYAD thrust / strong positive {'✅' if nyad_ok else '❌'}" + (f" (current {nyad:,.0f})" if pd.notna(nyad) else ""),
        f"• RSP:SPY flat or rising {'✅' if rsp_ok else '❌'}" + (f" (current {rsp_spy:.3f})" if pd.notna(rsp_spy) else ""),
        "",
        f"{'✅' if full_confirm else '❌'} Full Confirmation {'Met' if full_confirm else 'NOT Met'}:",
        f"• BPSPX %B {'>=' if pd.notna(bpspx_bb) and bpspx_bb >= thresholds['BPSPX_BB_LIFT'] else '<'} {thresholds['BPSPX_BB_LIFT']:.2f}",
        f"• SPXA50R {'>=' if pd.notna(spxa50r) and spxa50r >= thresholds['SPXA50R_REPAIR'] else '<'} {thresholds['SPXA50R_REPAIR']:.0f}",
        "",
        "Execution Rules:",
        "• Probe only when confirmation is incomplete",
        "• Full RSP sizing only after confirmation thresholds are met",
        "• URSP only when regime and durability are strong",
        "• Re-evaluate after official EOD breadth refresh",
    ]
    return "\n".join(lines)


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
    delta = cur - prev if pd.notna(cur) and pd.notna(prev) else np.nan
    improving = pd.notna(delta) and delta > 0
    worsening = pd.notna(delta) and delta < 0
    if pd.isna(cur):
        return "missing"
    if sym == "$BPSPX_%B":
        if cur < thresholds["BPSPX_BB_LIFT"]:
            return "pinned low but creeping up" if improving else "pinned low and not yet repairing"
        return "lifted off the lows and repairing" if cur < 0.50 else "well off the lows and broadening"
    if sym == "$SPXA50R":
        if cur < 25:
            return "weak breadth depth but improving" if improving else "weak breadth depth and not yet repaired"
        if cur <= thresholds["SPXA50R_REPAIR"]:
            return "near repair threshold and grinding higher" if improving else "near repair threshold but stalling"
        return "repair passed and improving" if improving else "repair passed but flattening"
    if sym == "$BPSPX":
        if cur < 40:
            return "washed out participation but repairing" if improving else "washed out participation and still weak"
        if cur <= thresholds["BPSPX_CONFIRM"]:
            return "participation improving but not yet confirmed"
        return "participation confirmed and healthier"
    if sym == "$NYMO":
        if cur < thresholds["NYMO_WASHOUT"]:
            return "deep washout momentum"
        if cur < thresholds["NYMO_HEALTHY"]:
            return "negative momentum but repairing" if improving else "negative momentum and still vulnerable"
        return "momentum recovered above healthy threshold"
    if sym == "$NYSI":
        if cur < thresholds["NYSI_LEVERAGE_OK"]:
            return "structurally damaged but improving" if improving else "structurally damaged and not leverage-ready"
        if cur < thresholds["NYSI_POSITIVE"]:
            return "still negative but repair is building" if improving else "still negative and fragile"
        return "trend backdrop positive and more durable"
    if sym == "$NYAD":
        return "strong breadth thrust day" if cur > thresholds["NYAD_THRUST"] else "breadth positive but not a thrust" if cur > 300 else "negative breadth pressure" if cur < -300 else "flat to mixed breadth"
    if sym == "$SPXADP":
        return "strong advancing-volume thrust" if cur > thresholds["SPXADP_THRUST"] else "positive breadth volume but not a thrust" if cur > 20 else "negative breadth volume pressure" if cur < -20 else "mixed breadth volume"
    if sym == "$CPCE":
        return "fear elevated, which can support a bounce" if cur >= thresholds["CPCE_FEAR"] else "fear support fading"
    if sym == "VXX":
        return "volatility easing, which helps repair" if worsening else "volatility rising, which pressures the bounce" if improving else "volatility mixed"
    if sym == "VXX_%B":
        return "volatility %B contained" if cur < thresholds["VXX_BB_PROBE_MAX"] else "volatility %B elevated"
    if sym == "RSP:SPY":
        return "leadership ratio improving" if improving else "leadership ratio weakening" if worsening else "leadership ratio flat"
    return "improving" if improving else "worsening" if worsening else "flat"


def build_momentum_context_table(symbols: List[str], snapshot: Dict[str, float], prev_snapshot: Dict[str, float], thresholds: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        cur = snapshot.get(sym, np.nan)
        prev = prev_snapshot.get(sym, np.nan)
        delta = cur - prev if pd.notna(cur) and pd.notna(prev) else np.nan
        rows.append({
            "Symbol": sym,
            "Current": None if pd.isna(cur) else round(float(cur), 2),
            "Prior": None if pd.isna(prev) else round(float(prev), 2),
            "Delta": None if pd.isna(delta) else round(float(delta), 2),
            "Target": target_for_symbol(sym, thresholds),
            "State": verbose_state(sym, cur, prev, thresholds),
        })
    return pd.DataFrame(rows)


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
        return None, None, None
    if realtime_overrides and sym in realtime_overrides:
        vals = realtime_overrides[sym]
        g.iloc[-1, g.columns.get_loc("close")] = vals["close"]
        g.iloc[-1, g.columns.get_loc("high")] = max(g.iloc[-1]["high"], vals["close"])
        g.iloc[-1, g.columns.get_loc("low")] = min(g.iloc[-1]["low"], vals["close"])
        g["rsi14"] = rsi(g["close"], 14)
        g["cci20"] = cci(g["high"], g["low"], g["close"], 20)
        g["pct_b20"] = percent_b(g["close"], 20, 2.0)
    g["candle_pattern"] = g.apply(detect_candle_pattern, axis=1)
    last = g.iloc[-1]
    price_fig = go.Figure()
    price_fig.add_trace(go.Candlestick(x=g["date"], open=g["open"], high=g["high"], low=g["low"], close=g["close"], name=sym))
    if g["ma20"].notna().any():
        price_fig.add_trace(go.Scatter(x=g["date"], y=g["ma20"], name="MA20"))
    if g["ma50"].notna().any():
        price_fig.add_trace(go.Scatter(x=g["date"], y=g["ma50"], name="MA50"))
    price_fig.update_layout(height=300, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=15, b=10))
    osc_fig = go.Figure()
    osc_fig.add_trace(go.Scatter(x=g["date"], y=g["pct_b20"], name="Bollinger %B(20,2)"))
    osc_fig.add_trace(go.Scatter(x=g["date"], y=g["rsi14"], name="RSI14", visible="legendonly"))
    osc_fig.add_trace(go.Scatter(x=g["date"], y=g["cci20"], name="CCI20", visible="legendonly"))
    osc_fig.add_hline(y=1.0, line_dash="dash")
    osc_fig.add_hline(y=0.5, line_dash="dot")
    osc_fig.add_hline(y=bb_lift, line_dash="dot", annotation_text=f"%B {bb_lift:.2f}")
    osc_fig.add_hline(y=0.0, line_dash="dash")
    osc_fig.update_layout(height=240, margin=dict(l=10, r=10, t=15, b=10))
    diag = {"Date": str(last["date"].date()), "Close": None if pd.isna(last["close"]) else round(float(last["close"]), 2), "Bollinger %B(20,2)": None if pd.isna(last["pct_b20"]) else round(float(last["pct_b20"]), 2), "RSI14": None if pd.isna(last["rsi14"]) else round(float(last["rsi14"]), 2), "CCI20": None if pd.isna(last["cci20"]) else round(float(last["cci20"]), 2), "Candle": last["candle_pattern"]}
    return price_fig, osc_fig, diag


def build_historical_score_series(hist_feat: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
    dates = sorted(pd.to_datetime(hist_feat["date"].dropna().unique()))
    rows = []
    for dt in dates:
        g = hist_feat[hist_feat["date"] == dt]
        snap = {}
        for _, row in g.iterrows():
            sym = row["symbol"]
            snap[sym] = safe_float(row["close"])
            snap[f"{sym}_%B"] = safe_float(row.get("pct_b20"))
        if "$BPSPX" not in snap or "$SPXA50R" not in snap:
            continue
        _, setup = setup_score_components(snap, thresholds)
        _, conf = confirmation_score_components(snap, thresholds)
        _, regime = regime_score_components(snap, thresholds)
        total = int(min(TOTAL_MAX, setup + conf + regime))
        rows.append({"date": pd.Timestamp(dt), "setup_score": setup, "confirmation_score": conf, "regime_score": regime, "breadth_score": total, "SPY": safe_float(snap.get("SPY", np.nan))})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


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


def run_backtest(df: pd.DataFrame):
    x = df.copy()
    x["spy_ret"] = x["SPY"].pct_change().fillna(0)
    x["strategy_ret"] = x["held"] * x["spy_ret"]
    x["equity_strategy"] = (1 + x["strategy_ret"]).cumprod()
    x["equity_spy"] = (1 + x["spy_ret"]).cumprod()
    x["switch"] = x["held"].diff().abs().fillna(0)
    stats = {
        "Strategy Return %": round(float((x["equity_strategy"].iloc[-1] - 1) * 100), 2) if len(x) else np.nan,
        "BuyHold Return %": round(float((x["equity_spy"].iloc[-1] - 1) * 100), 2) if len(x) else np.nan,
        "Strategy Max DD %": round(float((((x["equity_strategy"] / x["equity_strategy"].cummax()) - 1).min()) * 100), 2) if len(x) else np.nan,
        "BuyHold Max DD %": round(float((((x["equity_spy"] / x["equity_spy"].cummax()) - 1).min()) * 100), 2) if len(x) else np.nan,
        "Switches": int(x["switch"].sum()) if len(x) else 0,
        "Exposure %": round(float(100 * x["held"].mean()), 1) if len(x) else np.nan,
    }
    return x, stats


with st.sidebar:
    st.header("Baseline / Uploads")
    baseline_file = st.file_uploader("Upload historical baseline CSV", type=["csv"])
    realtime_file = st.file_uploader("Upload realtime / EOD snapshot CSV", type=["csv"])
    upload_tag = st.text_input("Upload tag", value="manual upload")
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
    slow_ema = st.number_input("Slow EMA", min_value=3, max_value=100, value=13, step=1)
    signal_ema = st.number_input("Signal EMA", min_value=1, max_value=30, value=5, step=1)
    deadband = st.number_input("Deadband", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    st.markdown("---")
    show_charts = st.multiselect("Decision tab charts", CHART_SYMBOLS, default=DEFAULT_MINIMAL_CHARTS)

baseline_meta = load_json(BASELINE_META_PATH, default={})
if baseline_file is not None:
    parsed_baseline = parse_stockcharts_historical_from_bytes(baseline_file.getvalue())
    parsed_baseline.to_parquet(BASELINE_PATH, index=False)
    baseline_meta = {"saved_at": datetime.now().isoformat(timespec="seconds"), "source_filename": baseline_file.name, "rows": int(len(parsed_baseline)), "symbols": int(parsed_baseline["symbol"].nunique())}
    save_json(BASELINE_META_PATH, baseline_meta)
    st.success("Historical baseline saved.")

if not BASELINE_PATH.exists():
    st.info("Upload and save the historical baseline first.")
    st.stop()

hist_long = pd.read_parquet(BASELINE_PATH)
hist_feat = add_indicator_features(hist_long)
baseline_snapshot = get_latest_feature_snapshot(hist_feat)
baseline_prior_snapshot = get_prior_feature_snapshot(hist_feat)
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
    for _, row in realtime_df.iterrows():
        sym = normalize_symbol(row["Symbol"])
        close_val = safe_float(row["Close"])
        if pd.notna(close_val):
            snapshot[sym] = close_val
    realtime_overrides = recompute_latest_indicators_from_realtime(hist_feat, realtime_df)
    for sym, vals in realtime_overrides.items():
        snapshot[sym] = vals["close"]
        snapshot[f"{sym}_%B"] = vals["pct_b20"]
        snapshot[f"{sym}_RSI14"] = vals["rsi14"]
        snapshot[f"{sym}_CCI20"] = vals["cci20"]
        snapshot[f"{sym}_TSI"] = vals["tsi_fast"]

setup_df, setup_score = setup_score_components(snapshot, thresholds)
confirm_df, confirmation_score = confirmation_score_components(snapshot, thresholds)
regime_df, regime_score = regime_score_components(snapshot, thresholds)
total_score = int(min(TOTAL_MAX, setup_score + confirmation_score + regime_score))
recommendation, recommendation_detail, rsp_size, ursp_size = model_recommendation(total_score, setup_score, confirmation_score, regime_score, snapshot, thresholds)
core_momentum_df = build_momentum_context_table(["$BPSPX_%B", "$BPSPX", "$SPXA50R", "$NYMO", "$NYSI", "$NYAD", "$SPXADP", "$CPCE", "RSP:SPY", "VXX", "VXX_%B"], snapshot, prev_snapshot, thresholds)
extended_momentum_df = build_momentum_context_table(["RSP", "SPY"], snapshot, prev_snapshot, thresholds)
recent_uploads_df = latest_upload_rows(5)
checklist_text = dynamic_trading_checklist(snapshot, thresholds, upload_history_df, recommendation)

if realtime_file is not None and not realtime_df.empty:
    append_upload_history({
        "snapshot_file": str(saved_path),
        "upload_ts": datetime.now().isoformat(timespec="seconds"),
        "tag": upload_tag,
        "BPSPX": snapshot.get("$BPSPX", np.nan),
        "BPSPX_%B": snapshot.get("$BPSPX_%B", np.nan),
        "SPXA50R": snapshot.get("$SPXA50R", np.nan),
        "ConfluenceScore": total_score,
        "Stage": breadth_stage(snapshot, thresholds)[0],
        "StageName": breadth_stage(snapshot, thresholds)[1],
        "RepairTrigger": qwen_repair_trigger(snapshot, thresholds),
    })
    upload_history_df = load_upload_history()
    recent_uploads_df = latest_upload_rows(5)

score_series = build_historical_score_series(hist_feat, thresholds)
if not score_series.empty:
    max_date = score_series["date"].max()
    score_series = score_series[score_series["date"] >= (max_date - pd.Timedelta(days=365))].reset_index(drop=True)
    score_series = build_oscillator(score_series, fast_ema, slow_ema, signal_ema, deadband)
    backtest_df, stats = run_backtest(score_series)
else:
    backtest_df = pd.DataFrame()
    stats = {}

tab1, tab2 = st.tabs(["Model Dashboard", "Score Oscillator Backtest"])

with tab1:
    st.subheader("Decision Dashboard")
    left_col, right_col = st.columns([0.65, 0.35])
    with left_col:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Setup", f"{setup_score}/{SETUP_MAX}")
        c2.metric("Confirmation", f"{confirmation_score}/{CONFIRM_MAX}")
        c3.metric("Regime", f"{regime_score}/{REGIME_MAX}")
        c4.metric("Total", f"{total_score}/{TOTAL_MAX}")
        st.markdown(f"## Recommendation: {recommendation}")
        st.write(recommendation_detail)
        pm1, pm2, pm3 = st.columns(3)
        pm1.metric("Suggested RSP Size", f"{int(rsp_size * 100)}%")
        pm2.metric("Suggested URSP Size", f"{int(ursp_size * 100)}%")
        stage_num, stage_name = breadth_stage(snapshot, thresholds)
        pm3.metric("Regime", f"{stage_num} - {stage_name}")
        st.subheader("Trading Checklist")
        st.code(checklist_text, language="text")
    with right_col:
        st.subheader("Current Input Snapshot")
        input_rows = []
        for sym in DECISION_SYMBOLS:
            val = snapshot.get(sym, np.nan)
            if pd.notna(val):
                input_rows.append({"Symbol": sym, "Value": round(float(val), 3)})
        st.dataframe(pd.DataFrame(input_rows), use_container_width=True, hide_index=True)
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
    st.subheader("Momentum Context — Core")
    st.dataframe(core_momentum_df, use_container_width=True, hide_index=True)
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
        display_cols = [c for c in ["upload_ts", "tag", "Stage", "StageName", "ConfluenceScore", "snapshot_file"] if c in recent_uploads_df.columns]
        st.dataframe(recent_uploads_df[display_cols], use_container_width=True, hide_index=True)
    st.subheader("Decision Charts")
    for sym in show_charts:
        price_fig, osc_fig, diag = make_symbol_chart(hist_feat, sym, thresholds["BPSPX_BB_LIFT"], realtime_overrides=realtime_overrides)
        if price_fig is None:
            continue
        with st.expander(sym, expanded=sym in ["$BPSPX", "$SPXA50R"]):
            l, r = st.columns([1.7, 1.0])
            with l:
                st.plotly_chart(price_fig, use_container_width=True)
                st.plotly_chart(osc_fig, use_container_width=True)
            with r:
                st.json(diag)

with tab2:
    st.subheader("1-Year Breadth Score Oscillator Backtest")
    if backtest_df.empty:
        st.info("Not enough historical data to build the backtest.")
    else:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Strategy Return", f"{stats.get('Strategy Return %', np.nan)}%")
        p2.metric("Buy & Hold Return", f"{stats.get('BuyHold Return %', np.nan)}%")
        p3.metric("Strategy Max DD", f"{stats.get('Strategy Max DD %', np.nan)}%")
        p4.metric("Switches", f"{stats.get('Switches', 0)}")
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=backtest_df["date"], y=backtest_df["SPY"], name="SPY"))
        fig_price.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), title="SPY")
        fig_osc = go.Figure()
        fig_osc.add_trace(go.Scatter(x=backtest_df["date"], y=backtest_df["osc"], name="Breadth Oscillator"))
        fig_osc.add_trace(go.Scatter(x=backtest_df["date"], y=backtest_df["osc_signal"], name="Signal", visible="legendonly"))
        fig_osc.add_hline(y=0.0, line_dash="dash")
        fig_osc.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=10), title="PPO-Style Breadth Oscillator")
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=backtest_df["date"], y=backtest_df["equity_strategy"], name="Strategy"))
        fig_eq.add_trace(go.Scatter(x=backtest_df["date"], y=backtest_df["equity_spy"], name="Buy & Hold"))
        fig_eq.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), title="Equity Curves")
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

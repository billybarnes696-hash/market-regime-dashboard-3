import io
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ============================================================
# PAGE
# ============================================================
st.set_page_config(page_title="Breadth Engine vNext", layout="wide")
st.title("Breadth Engine vNext")
st.caption("Stateful breadth workstation: baseline memory + rolling uploads + weekly + optional ChatGPT API")


# ============================================================
# STORAGE
# ============================================================
APP_DIR = Path("breadth_engine_store")
APP_DIR.mkdir(exist_ok=True)

BASELINE_PATH = APP_DIR / "baseline_hist.parquet"
BASELINE_META_PATH = APP_DIR / "baseline_meta.json"
UPLOAD_HISTORY_PATH = APP_DIR / "upload_history.csv"
LLM_HISTORY_PATH = APP_DIR / "llm_history.jsonl"


# ============================================================
# THRESHOLDS
# ============================================================
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
    "NYHL_POSITIVE": 0.0,
}


# ============================================================
# SYMBOLS
# ============================================================
CORE_SYMBOLS = [
    "$BPSPX",
    "$SPXA50R",
    "$NYMO",
    "$NYSI",
    "$NYAD",
    "$SPXADP",
    "$CPCE",
    "$NYHL",
    "RSP:SPY",
    "VXX",
]

EXTENDED_SYMBOLS = [
    "$BPNYA",
    "$OEXA150R",
    "$OEXA200R",
    "$OEXA50R",
    "$SPX",
    "RSP",
    "SMH:SPY",
    "XLF:SPY",
    "IWM:SPY",
    "HYG:IEF",
    "HYG:TLT",
    "SPXS:SVOL",
    "URSP",
]

ALL_SYMBOLS = CORE_SYMBOLS + [s for s in EXTENDED_SYMBOLS if s not in CORE_SYMBOLS]
DEFAULT_MINIMAL_CHARTS = ["$BPSPX", "$SPXA50R", "$NYMO", "$NYSI", "RSP"]


OHLC_PATTERN = re.compile(
    r"(?P<day>Mon|Tue|Wed|Thu|Fri)\s+"
    r"(?P<date>\d{2}-\d{2}-\d{4})\s+"
    r"(?P<open>-?\d+(?:\.\d+)?)\s+"
    r"(?P<high>-?\d+(?:\.\d+)?)\s+"
    r"(?P<low>-?\d+(?:\.\d+)?)\s+"
    r"(?P<close>-?\d+(?:\.\d+)?)\s+"
    r"(?P<volume>-?\d+(?:\.\d+)?)"
)


# ============================================================
# UTILS
# ============================================================
def normalize_symbol(sym: str) -> str:
    return str(sym).strip()


def fmt_num(val, digits=2) -> str:
    if pd.isna(val):
        return "n/a"
    return f"{float(val):.{digits}f}"


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def save_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2))


def append_jsonl(path: Path, row: Dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


# ============================================================
# OPENAI
# ============================================================
def get_manual_api_key() -> Optional[str]:
    return st.session_state.get("manual_api_key") or None


def get_openai_client_and_model() -> Tuple[Optional[object], Optional[str], str]:
    model_name = "gpt-5.4"
    api_key = None

    try:
        if "OPENAI_MODEL" in st.secrets:
            model_name = st.secrets["OPENAI_MODEL"]
        if "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        api_key = get_manual_api_key()

    if not api_key or OpenAI is None:
        return None, api_key, model_name

    try:
        client = OpenAI(api_key=api_key)
        return client, api_key, model_name
    except Exception:
        return None, api_key, model_name


def call_openai_analysis(client, model_name: str, prompt: str, retries: int = 2) -> str:
    last_err = None
    for attempt in range(retries + 1):
        try:
            response = client.responses.create(
                model=model_name,
                input=prompt,
            )
            return response.output_text
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
            else:
                raise last_err


# ============================================================
# INDICATORS
# ============================================================
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


def true_strength_index(series: pd.Series, long: int = 25, short: int = 13, signal: int = 7):
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
    return (series - lower) / (upper - lower).replace(0, np.nan)


def roc(series: pd.Series, period: int = 3) -> pd.Series:
    return (series / series.shift(period) - 1) * 100


# ============================================================
# PARSERS
# ============================================================
@st.cache_data(show_spinner=False)
def parse_stockcharts_historical_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    raw = pd.read_csv(io.BytesIO(file_bytes), header=None)
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

    if not records:
        raise ValueError("Historical baseline could not be parsed. Use the StockCharts-style historical export.")

    return pd.DataFrame(records).sort_values(["symbol", "date"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def parse_realtime_snapshot_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))

    if "Symbol" not in df.columns:
        raise ValueError("Realtime snapshot must contain a 'Symbol' column.")

    df["Symbol"] = df["Symbol"].astype(str).map(normalize_symbol)

    close_candidates = [
        "Close",
        "Last",
        "Price",
        "Current",
        "Value",
        "Daily Close",
        "Close Price",
    ]
    close_col = next((c for c in close_candidates if c in df.columns), None)
    if close_col is None:
        raise ValueError("Realtime snapshot must contain a close-like column.")

    df["Close"] = pd.to_numeric(df[close_col], errors="coerce")

    pct_candidates = [
        "Daily PctChange(1,Daily Close)",
        "% Change",
        "Pct Change",
        "Change %",
        "Daily Change %",
    ]
    pct_col = next((c for c in pct_candidates if c in df.columns), None)
    df["PctChange"] = pd.to_numeric(df[pct_col], errors="coerce") if pct_col else np.nan

    return df


# ============================================================
# FEATURE ENGINE
# ============================================================
@st.cache_data(show_spinner=False)
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
        g["ma20"] = g["close"].rolling(20).mean()
        g["ma50"] = g["close"].rolling(50).mean()
        g["ma200"] = g["close"].rolling(200).mean()
        g["daily_pct"] = g["close"].pct_change() * 100

        frames.append(g)

    return pd.concat(frames, ignore_index=True)


def get_latest_feature_snapshot(hist_feat: pd.DataFrame) -> Dict[str, float]:
    snapshot: Dict[str, float] = {}
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
    snapshot: Dict[str, float] = {}
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


# ============================================================
# REGIME / SCORE LOGIC
# ============================================================
def score_nyad(val: float, thresholds: Dict[str, float]) -> int:
    if pd.isna(val):
        return 0
    if val > thresholds["NYAD_THRUST"]:
        return 2
    if val > 300:
        return 1
    if val < -1500:
        return -2
    if val < -300:
        return -1
    return 0


def score_spxadp(val: float, thresholds: Dict[str, float]) -> int:
    if pd.isna(val):
        return 0
    if val > thresholds["SPXADP_THRUST"]:
        return 2
    if val > 20:
        return 1
    if val < -60:
        return -2
    if val < -20:
        return -1
    return 0


def derive_bpspx_direction(latest_close: float, prior_close: float) -> int:
    if pd.isna(latest_close) or pd.isna(prior_close):
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


def compute_nymo_proxy(snapshot: Dict[str, float], thresholds: Dict[str, float], prev_snapshot: Optional[Dict[str, float]]) -> float:
    nyad = snapshot.get("$NYAD", np.nan)
    spxadp = snapshot.get("$SPXADP", np.nan)
    bpspx = snapshot.get("$BPSPX", np.nan)
    prev_bpspx = (prev_snapshot or {}).get("$BPSPX", np.nan)
    bpspx_dir = derive_bpspx_direction(bpspx, prev_bpspx)
    return 0.5 * score_nyad(nyad, thresholds) + 0.3 * score_spxadp(spxadp, thresholds) + 0.2 * bpspx_dir


def qwen_repair_trigger(snapshot: Dict[str, float], thresholds: Dict[str, float]) -> bool:
    bpspx_bb = snapshot.get("$BPSPX_%B", np.nan)
    spxa50r = snapshot.get("$SPXA50R", np.nan)
    return bool(
        pd.notna(bpspx_bb)
        and pd.notna(spxa50r)
        and bpspx_bb > thresholds["BPSPX_BB_LIFT"]
        and spxa50r > thresholds["SPXA50R_REPAIR"]
    )


def breadth_stage(snapshot: Dict[str, float], thresholds: Dict[str, float]) -> Tuple[int, str]:
    bpspx_bb = snapshot.get("$BPSPX_%B", np.nan)
    spxa50r = snapshot.get("$SPXA50R", np.nan)
    bpspx = snapshot.get("$BPSPX", np.nan)
    nysi = snapshot.get("$NYSI", np.nan)

    if pd.isna(bpspx_bb):
        return 0, "Unknown"
    if bpspx_bb < thresholds["BPSPX_BB_LIFT"]:
        return 0, "Washout / lower-band pressure"
    if bpspx_bb >= thresholds["BPSPX_BB_LIFT"] and (pd.isna(spxa50r) or spxa50r <= thresholds["SPXA50R_REPAIR"]):
        return 1, "Bounce starting"
    if bpspx_bb > thresholds["BPSPX_BB_LIFT"] and pd.notna(spxa50r) and spxa50r > thresholds["SPXA50R_REPAIR"] and (pd.isna(bpspx) or bpspx <= thresholds["BPSPX_CONFIRM"]):
        return 2, "Breadth repair"
    if bpspx_bb > thresholds["BPSPX_BB_LIFT"] and pd.notna(spxa50r) and spxa50r > thresholds["SPXA50R_REPAIR"] and pd.notna(bpspx) and bpspx > thresholds["BPSPX_CONFIRM"] and (pd.isna(nysi) or nysi <= thresholds["NYSI_POSITIVE"]):
        return 3, "Participation confirmation"
    if bpspx_bb > thresholds["BPSPX_BB_LIFT"] and pd.notna(spxa50r) and spxa50r > thresholds["SPXA50R_REPAIR"] and pd.notna(bpspx) and bpspx > thresholds["BPSPX_CONFIRM"] and pd.notna(nysi) and nysi > thresholds["NYSI_POSITIVE"]:
        return 4, "Trend durability / healthier regime"
    return 0, "Unknown"


def bounce_quality(snapshot: Dict[str, float], prev_snapshot: Optional[Dict[str, float]], thresholds: Dict[str, float]) -> Tuple[int, List[str]]:
    score = 0
    notes: List[str] = []

    bpspx = snapshot.get("$BPSPX", np.nan)
    bpspx_bb = snapshot.get("$BPSPX_%B", np.nan)
    spxa50r = snapshot.get("$SPXA50R", np.nan)
    nyad = snapshot.get("$NYAD", np.nan)
    spxadp = snapshot.get("$SPXADP", np.nan)
    cpce = snapshot.get("$CPCE", np.nan)
    vxx = snapshot.get("VXX", np.nan)
    rsp_spy = snapshot.get("RSP:SPY", np.nan)

    if pd.notna(bpspx_bb):
        if bpspx_bb > thresholds["BPSPX_BB_LIFT"]:
            score += 1
            notes.append("BPSPX %B lifted")
        else:
            notes.append("BPSPX %B pinned low")

    if pd.notna(spxa50r):
        if spxa50r > thresholds["SPXA50R_REPAIR"]:
            score += 2
            notes.append("SPXA50R > 30")
        elif spxa50r >= 25:
            score += 1
            notes.append("SPXA50R improving")

    if pd.notna(bpspx):
        if bpspx > thresholds["BPSPX_CONFIRM"]:
            score += 2
            notes.append("BPSPX > 45")
        elif bpspx >= 40:
            score += 1
            notes.append("BPSPX improving")

    if pd.notna(nyad) and pd.notna(spxadp):
        if nyad > thresholds["NYAD_THRUST"] and spxadp > thresholds["SPXADP_THRUST"]:
            score += 2
            notes.append("Breadth thrust")
        elif nyad > 500 or spxadp > 20:
            score += 1
            notes.append("Moderate thrust")

    if pd.notna(cpce) and cpce >= thresholds["CPCE_FEAR"]:
        score += 1
        notes.append("CPCE fear support")

    if prev_snapshot:
        prev_vxx = prev_snapshot.get("VXX", np.nan)
        if pd.notna(vxx) and pd.notna(prev_vxx) and vxx < prev_vxx:
            score += 1
            notes.append("VXX easing")

        prev_rsp_spy = prev_snapshot.get("RSP:SPY", np.nan)
        if pd.notna(rsp_spy) and pd.notna(prev_rsp_spy) and rsp_spy >= prev_rsp_spy:
            score += 1
            notes.append("Equal-weight not lagging")

    return min(score, 9), notes


def breadth_confluence_score(snapshot: Dict[str, float], thresholds: Dict[str, float]) -> float:
    score = 0.0

    bpspx_bb = snapshot.get("$BPSPX_%B", np.nan)
    spxa50r = snapshot.get("$SPXA50R", np.nan)
    bpspx = snapshot.get("$BPSPX", np.nan)
    nymo = snapshot.get("$NYMO", np.nan)
    nysi = snapshot.get("$NYSI", np.nan)
    nyad = snapshot.get("$NYAD", np.nan)
    spxadp = snapshot.get("$SPXADP", np.nan)
    cpce = snapshot.get("$CPCE", np.nan)
    nyhl = snapshot.get("$NYHL", np.nan)

    if pd.notna(bpspx_bb):
        score += 1.0 if bpspx_bb > thresholds["BPSPX_BB_LIFT"] else -1.0
    if pd.notna(spxa50r):
        score += 1.5 if spxa50r > thresholds["SPXA50R_REPAIR"] else (-0.5 if spxa50r >= 25 else -1.5)
    if pd.notna(bpspx):
        score += 1.5 if bpspx > thresholds["BPSPX_CONFIRM"] else (-0.5 if bpspx >= 40 else -1.5)
    if pd.notna(nymo):
        score += 1.0 if nymo > thresholds["NYMO_HEALTHY"] else (-1.0 if nymo < -50 else 0.0)
    if pd.notna(nysi):
        score += 1.5 if nysi > thresholds["NYSI_POSITIVE"] else (-1.5 if nysi < thresholds["NYSI_LEVERAGE_OK"] else -0.5)
    if pd.notna(nyad) and pd.notna(spxadp):
        score += 1.5 if nyad > thresholds["NYAD_THRUST"] and spxadp > thresholds["SPXADP_THRUST"] else 0.0
    if pd.notna(cpce):
        score += 0.5 if cpce >= thresholds["CPCE_FEAR"] else 0.0
    if pd.notna(nyhl):
        score += 1.0 if nyhl > thresholds["NYHL_POSITIVE"] else -0.5

    return max(-10.0, min(10.0, round(score, 2)))


def scenario_scores(snapshot: Dict[str, float], thresholds: Dict[str, float]) -> Dict[str, int]:
    bearish_cont = 0
    oversold_bounce = 0
    true_repair = 0

    bpspx = snapshot.get("$BPSPX", np.nan)
    bpspx_bb = snapshot.get("$BPSPX_%B", np.nan)
    spxa50r = snapshot.get("$SPXA50R", np.nan)
    nymo = snapshot.get("$NYMO", np.nan)
    nysi = snapshot.get("$NYSI", np.nan)
    nyad = snapshot.get("$NYAD", np.nan)
    spxadp = snapshot.get("$SPXADP", np.nan)
    cpce = snapshot.get("$CPCE", np.nan)

    if pd.notna(nysi) and nysi < thresholds["NYSI_LEVERAGE_OK"]:
        bearish_cont += 3
    if pd.notna(nymo) and nymo < thresholds["NYMO_HEALTHY"]:
        bearish_cont += 2
    if pd.notna(bpspx) and bpspx < thresholds["BPSPX_CONFIRM"]:
        bearish_cont += 2
    if pd.notna(spxa50r) and spxa50r < thresholds["SPXA50R_REPAIR"]:
        bearish_cont += 2

    if pd.notna(cpce) and cpce >= thresholds["CPCE_FEAR"]:
        oversold_bounce += 2
    if pd.notna(bpspx_bb) and bpspx_bb < thresholds["BPSPX_BB_LIFT"]:
        oversold_bounce += 2
    if pd.notna(nymo) and nymo < 0:
        oversold_bounce += 1
    if pd.notna(spxa50r) and spxa50r >= 25:
        oversold_bounce += 1

    if qwen_repair_trigger(snapshot, thresholds):
        true_repair += 3
    if pd.notna(bpspx) and bpspx > thresholds["BPSPX_CONFIRM"]:
        true_repair += 2
    if pd.notna(nymo) and nymo > thresholds["NYMO_HEALTHY"]:
        true_repair += 1
    if pd.notna(nysi) and nysi > thresholds["NYSI_POSITIVE"]:
        true_repair += 2
    if pd.notna(nyad) and pd.notna(spxadp) and nyad > thresholds["NYAD_THRUST"] and spxadp > thresholds["SPXADP_THRUST"]:
        true_repair += 2

    return {
        "Bearish continuation": bearish_cont,
        "Oversold bounce": oversold_bounce,
        "True repair": true_repair,
    }


def failed_bounce_risk(snapshot: Dict[str, float], prev_snapshot: Optional[Dict[str, float]], thresholds: Dict[str, float]) -> Tuple[str, bool, List[str]]:
    reasons = []
    risk_points = 0

    stage_num, _ = breadth_stage(snapshot, thresholds)
    bpspx_bb = snapshot.get("$BPSPX_%B", np.nan)
    spxa50r = snapshot.get("$SPXA50R", np.nan)
    nymo = snapshot.get("$NYMO", np.nan)
    nysi = snapshot.get("$NYSI", np.nan)
    rsp_spy = snapshot.get("RSP:SPY", np.nan)
    vxx = snapshot.get("VXX", np.nan)

    prev_bpspx_bb = (prev_snapshot or {}).get("$BPSPX_%B", np.nan)
    prev_spxa50r = (prev_snapshot or {}).get("$SPXA50R", np.nan)
    prev_rsp_spy = (prev_snapshot or {}).get("RSP:SPY", np.nan)
    prev_vxx = (prev_snapshot or {}).get("VXX", np.nan)

    if stage_num <= 1:
        risk_points += 1
        reasons.append("Bounce still early-stage")
    if pd.notna(bpspx_bb) and bpspx_bb < thresholds["BPSPX_BB_LIFT"]:
        risk_points += 2
        reasons.append("BPSPX %B below repair lift")
    if pd.notna(prev_bpspx_bb) and pd.notna(bpspx_bb) and bpspx_bb < prev_bpspx_bb:
        risk_points += 1
        reasons.append("BPSPX %B deteriorating")
    if pd.notna(spxa50r) and spxa50r < thresholds["SPXA50R_REPAIR"]:
        risk_points += 1
        reasons.append("SPXA50R below repair threshold")
    if pd.notna(prev_spxa50r) and pd.notna(spxa50r) and spxa50r < prev_spxa50r:
        risk_points += 1
        reasons.append("SPXA50R weakening")
    if pd.notna(nymo) and nymo < thresholds["NYMO_HEALTHY"]:
        risk_points += 1
        reasons.append("NYMO still negative")
    if pd.notna(nysi) and nysi < thresholds["NYSI_LEVERAGE_OK"]:
        risk_points += 2
        reasons.append("NYSI still structurally damaged")
    if pd.notna(prev_rsp_spy) and pd.notna(rsp_spy) and rsp_spy < prev_rsp_spy:
        risk_points += 1
        reasons.append("RSP:SPY weakening")
    if pd.notna(prev_vxx) and pd.notna(vxx) and vxx > prev_vxx:
        risk_points += 1
        reasons.append("VXX rising")

    if risk_points >= 7:
        return "HIGH", True, reasons
    if risk_points >= 4:
        return "MED", True, reasons
    return "LOW", False, reasons


def build_narrative(snapshot: Dict[str, float], prev_snapshot: Optional[Dict[str, float]], thresholds: Dict[str, float]) -> str:
    stage_num, stage_name = breadth_stage(snapshot, thresholds)
    repair = qwen_repair_trigger(snapshot, thresholds)
    confluence = breadth_confluence_score(snapshot, thresholds)
    bounce_score, notes = bounce_quality(snapshot, prev_snapshot, thresholds)
    failed_risk, short_probe_ok, failed_reasons = failed_bounce_risk(snapshot, prev_snapshot, thresholds)

    parts = [
        f"Stage {stage_num}: {stage_name}.",
        f"Breadth Confluence Score: {confluence}/10.",
        f"Bounce Quality Score: {bounce_score}/9.",
        f"Failed-bounce risk: {failed_risk}.",
    ]

    bpspx_bb = snapshot.get("$BPSPX_%B", np.nan)
    if pd.notna(bpspx_bb):
        parts.append(f"BPSPX Bollinger %B is {bpspx_bb:.2f}.")

    parts.append("Qwen repair trigger is active." if repair else "Qwen repair trigger is not active.")

    if short_probe_ok and failed_risk in {"MED", "HIGH"}:
        parts.append("Short probe is only attractive on failed-bounce conditions, not fresh panic lows.")

    bpspx = snapshot.get("$BPSPX", np.nan)
    spxa50r = snapshot.get("$SPXA50R", np.nan)
    nymo = snapshot.get("$NYMO", np.nan)
    nysi = snapshot.get("$NYSI", np.nan)

    if pd.notna(bpspx) and pd.notna(spxa50r):
        parts.append(f"BPSPX {bpspx:.2f} and SPXA50R {spxa50r:.2f} define participation and depth.")
    if pd.notna(nymo):
        parts.append(f"NYMO is {nymo:.2f}.")
    if pd.notna(nysi):
        parts.append(f"NYSI is {nysi:.2f}.")
    if notes:
        parts.append("Key supports / drags: " + "; ".join(notes) + ".")
    if failed_reasons:
        parts.append("Failed-bounce drivers: " + "; ".join(failed_reasons) + ".")

    return " ".join(parts)


# ============================================================
# WEEKLY
# ============================================================
def weekly_analysis(hist_feat: pd.DataFrame) -> pd.DataFrame:
    rows = []
    focus = ["$BPSPX", "$SPXA50R", "$NYMO", "$NYSI", "RSP"]

    for sym in focus:
        g = hist_feat[hist_feat["symbol"] == sym].sort_values("date").copy()
        if g.empty:
            continue

        weekly = (
            g.set_index("date")[["close"]]
            .resample("W-FRI")
            .last()
            .dropna()
            .rename(columns={"close": "weekly_close"})
        )
        if len(weekly) < 3:
            continue

        weekly["w1"] = weekly["weekly_close"].diff(1)
        weekly["w4"] = weekly["weekly_close"].diff(4)
        latest = weekly.iloc[-1]

        if pd.isna(latest["w4"]):
            trend = "insufficient"
        elif latest["w4"] > 0:
            trend = "improving"
        elif latest["w4"] < 0:
            trend = "deteriorating"
        else:
            trend = "flat"

        rows.append(
            {
                "Symbol": sym,
                "Weekly Close": round(float(latest["weekly_close"]), 2),
                "1W Change": None if pd.isna(latest["w1"]) else round(float(latest["w1"]), 2),
                "4W Change": None if pd.isna(latest["w4"]) else round(float(latest["w4"]), 2),
                "Weekly Trend": trend,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# UPLOAD HISTORY
# ============================================================
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


def latest_upload_rows(n: int = 5) -> pd.DataFrame:
    hist = load_upload_history()
    if hist.empty:
        return hist
    if "upload_ts" in hist.columns:
        hist = hist.sort_values("upload_ts")
    return hist.tail(n).reset_index(drop=True)


def build_upload_summary(snapshot: Dict[str, float], thresholds: Dict[str, float], tag: str, prev_snapshot: Optional[Dict[str, float]]) -> Dict:
    stage_num, stage_name = breadth_stage(snapshot, thresholds)
    bounce_score, _ = bounce_quality(snapshot, prev_snapshot, thresholds)
    confluence = breadth_confluence_score(snapshot, thresholds)
    scenarios = scenario_scores(snapshot, thresholds)
    failed_risk, short_probe_ok, _ = failed_bounce_risk(snapshot, prev_snapshot, thresholds)

    return {
        "upload_ts": datetime.now().isoformat(timespec="seconds"),
        "tag": tag,
        "BPSPX": snapshot.get("$BPSPX", np.nan),
        "BPSPX_%B": snapshot.get("$BPSPX_%B", np.nan),
        "SPXA50R": snapshot.get("$SPXA50R", np.nan),
        "NYMO": snapshot.get("$NYMO", np.nan),
        "NYSI": snapshot.get("$NYSI", np.nan),
        "NYAD": snapshot.get("$NYAD", np.nan),
        "SPXADP": snapshot.get("$SPXADP", np.nan),
        "CPCE": snapshot.get("$CPCE", np.nan),
        "NYHL": snapshot.get("$NYHL", np.nan),
        "VXX": snapshot.get("VXX", np.nan),
        "RSP:SPY": snapshot.get("RSP:SPY", np.nan),
        "NYMO_PROXY": snapshot.get("NYMO_PROXY", np.nan),
        "ConfluenceScore": confluence,
        "BounceScore": bounce_score,
        "Stage": stage_num,
        "StageName": stage_name,
        "RepairTrigger": qwen_repair_trigger(snapshot, thresholds),
        "FailedBounceRisk": failed_risk,
        "ShortProbeEligible": short_probe_ok,
        "BearishContinuation": scenarios["Bearish continuation"],
        "OversoldBounce": scenarios["Oversold bounce"],
        "TrueRepair": scenarios["True repair"],
    }


def intraday_multi_upload_analysis(history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df.empty or len(history_df) < 2:
        return pd.DataFrame()

    cols = ["BPSPX", "BPSPX_%B", "SPXA50R", "NYMO", "NYSI", "BounceScore", "ConfluenceScore"]
    rows = []

    last = history_df.iloc[-1]
    prev = history_df.iloc[-2]

    for col in cols:
        if col not in history_df.columns:
            continue
        cur = safe_float(last[col])
        old = safe_float(prev[col])
        delta = cur - old if pd.notna(cur) and pd.notna(old) else np.nan

        if pd.isna(delta):
            trend = "n/a"
        elif delta > 0:
            trend = "improving"
        elif delta < 0:
            trend = "worsening"
        else:
            trend = "flat"

        rows.append(
            {
                "Metric": col,
                "Current": None if pd.isna(cur) else round(float(cur), 2),
                "Prior Upload": None if pd.isna(old) else round(float(old), 2),
                "Delta": None if pd.isna(delta) else round(float(delta), 2),
                "Direction": trend,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MOMENTUM CONTEXT
# ============================================================
def target_for_symbol(sym: str, thresholds: Dict[str, float]) -> str:
    targets = {
        "$BPSPX_%B": f">{thresholds['BPSPX_BB_LIFT']:.2f}",
        "$SPXA50R": f">{thresholds['SPXA50R_REPAIR']:.0f}",
        "$BPSPX": f">{thresholds['BPSPX_CONFIRM']:.0f}",
        "$NYMO": f">{thresholds['NYMO_HEALTHY']:.0f}",
        "$NYSI": f">{thresholds['NYSI_POSITIVE']:.0f}",
        "$NYAD": f">{thresholds['NYAD_THRUST']:.0f}",
        "$SPXADP": f">{thresholds['SPXADP_THRUST']:.0f}",
        "$CPCE": f">={thresholds['CPCE_FEAR']:.2f}",
        "$NYHL": f">{thresholds['NYHL_POSITIVE']:.0f}",
        "RSP:SPY": "rising",
        "VXX": "falling",
        "SMH:SPY": "rising",
        "XLF:SPY": "rising",
        "IWM:SPY": "rising",
    }
    return targets.get(sym, "improving")


def verbose_state(sym: str, cur: float, prev: float, thresholds: Dict[str, float]) -> str:
    delta = cur - prev if pd.notna(cur) and pd.notna(prev) else np.nan

    if pd.isna(cur):
        return "missing"

    improving = pd.notna(delta) and delta > 0
    worsening = pd.notna(delta) and delta < 0

    if sym == "$BPSPX_%B":
        if cur < thresholds["BPSPX_BB_LIFT"]:
            return "pinned low but creeping up" if improving else "pinned low and not yet repairing"
        if cur < 0.50:
            return "lifted off the lows and repairing"
        return "well off the lows and broadening"

    if sym == "$SPXA50R":
        if cur < 25:
            return "weak breadth depth and not yet repaired" if not improving else "weak breadth depth but improving"
        if cur <= thresholds["SPXA50R_REPAIR"]:
            return "near repair threshold and grinding higher" if improving else "near repair threshold but stalling"
        return "repair passed and improving" if improving else "repair passed but momentum is flattening"

    if sym == "$BPSPX":
        if cur < 40:
            return "washed out participation; bounce still fragile" if not improving else "washed out but repairing"
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
            return "structurally damaged trend and not leverage-ready" if not improving else "structurally damaged but improving"
        if cur < thresholds["NYSI_POSITIVE"]:
            return "still negative, but repair is building" if improving else "still negative and fragile"
        return "trend backdrop positive and more durable"

    if sym == "$NYAD":
        if cur > thresholds["NYAD_THRUST"]:
            return "strong breadth thrust day"
        if cur > 300:
            return "breadth positive but not a thrust"
        if cur < -300:
            return "negative breadth pressure"
        return "flat to mixed breadth"

    if sym == "$SPXADP":
        if cur > thresholds["SPXADP_THRUST"]:
            return "strong advancing-volume thrust"
        if cur > 20:
            return "positive breadth volume but not a thrust"
        if cur < -20:
            return "negative breadth volume pressure"
        return "mixed breadth volume"

    if sym == "$CPCE":
        if cur >= thresholds["CPCE_FEAR"]:
            return "fear elevated, which can support a bounce"
        return "fear support fading"

    if sym == "$NYHL":
        if cur > thresholds["NYHL_POSITIVE"]:
            return "new highs are outpacing lows"
        return "new lows still dominating"

    if sym == "VXX":
        if worsening:
            return "volatility rising, which pressures the bounce"
        if improving:
            return "volatility easing, which helps repair"
        return "volatility mixed"

    if sym in {"RSP:SPY", "SMH:SPY", "XLF:SPY", "IWM:SPY"}:
        if improving:
            return "leadership ratio improving"
        if worsening:
            return "leadership ratio weakening"
        return "leadership ratio flat"

    return "improving" if improving else "worsening" if worsening else "flat"


def build_momentum_context_table(
    symbols: List[str],
    snapshot: Dict[str, float],
    prev_snapshot: Dict[str, float],
    thresholds: Dict[str, float],
) -> pd.DataFrame:
    rows = []

    for sym in symbols:
        key = sym
        if sym in {"$BPSPX"}:
            extra_key = "$BPSPX_%B"
            if extra_key not in symbols and extra_key not in rows:
                pass

        cur = snapshot.get(sym, np.nan)
        prev = prev_snapshot.get(sym, np.nan)
        delta = cur - prev if pd.notna(cur) and pd.notna(prev) else np.nan

        rows.append(
            {
                "Symbol": sym,
                "Current": None if pd.isna(cur) else round(float(cur), 2),
                "Prior": None if pd.isna(prev) else round(float(prev), 2),
                "Delta": None if pd.isna(delta) else round(float(delta), 2),
                "Target": target_for_symbol(sym, thresholds),
                "State": verbose_state(sym, cur, prev, thresholds),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# CHARTS
# ============================================================
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


def make_symbol_chart(hist_feat: pd.DataFrame, sym: str, bb_lift: float):
    g = hist_feat[hist_feat["symbol"] == sym].sort_values("date").copy()
    if g.empty:
        return None, None, None

    g["candle_pattern"] = g.apply(detect_candle_pattern, axis=1)
    last = g.iloc[-1]

    price_fig = go.Figure()
    price_fig.add_trace(
        go.Candlestick(
            x=g["date"],
            open=g["open"],
            high=g["high"],
            low=g["low"],
            close=g["close"],
            name=sym,
        )
    )
    if g["ma20"].notna().any():
        price_fig.add_trace(go.Scatter(x=g["date"], y=g["ma20"], name="MA20"))
    if g["ma50"].notna().any():
        price_fig.add_trace(go.Scatter(x=g["date"], y=g["ma50"], name="MA50"))
    if g["ma200"].notna().any():
        price_fig.add_trace(go.Scatter(x=g["date"], y=g["ma200"], name="MA200"))
    price_fig.update_layout(height=300, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=15, b=10))

    osc_fig = go.Figure()
    osc_fig.add_trace(go.Scatter(x=g["date"], y=g["pct_b20"], name="Bollinger %B(20,2)"))
    osc_fig.add_trace(go.Scatter(x=g["date"], y=g["rsi14"], name="RSI14", visible="legendonly"))
    osc_fig.add_trace(go.Scatter(x=g["date"], y=g["cci20"], name="CCI20", visible="legendonly"))
    osc_fig.add_trace(go.Scatter(x=g["date"], y=g["tsi_fast"], name="TSI(4,2,4)", visible="legendonly"))
    osc_fig.add_hline(y=1.0, line_dash="dash")
    osc_fig.add_hline(y=0.5, line_dash="dot")
    osc_fig.add_hline(y=bb_lift, line_dash="dot", annotation_text=f"%B {bb_lift:.2f}")
    osc_fig.add_hline(y=0.0, line_dash="dash")
    osc_fig.update_layout(height=240, margin=dict(l=10, r=10, t=15, b=10))

    diag = {
        "Date": str(last["date"].date()),
        "Close": None if pd.isna(last["close"]) else round(float(last["close"]), 2),
        "Bollinger %B(20,2)": None if pd.isna(last["pct_b20"]) else round(float(last["pct_b20"]), 2),
        "RSI14": None if pd.isna(last["rsi14"]) else round(float(last["rsi14"]), 2),
        "CCI20": None if pd.isna(last["cci20"]) else round(float(last["cci20"]), 2),
        "TSI": None if pd.isna(last["tsi_fast"]) else round(float(last["tsi_fast"]), 2),
        "ROC3": None if pd.isna(last["roc3"]) else round(float(last["roc3"]), 2),
        "ROC5": None if pd.isna(last["roc5"]) else round(float(last["roc5"]), 2),
        "Candle": last["candle_pattern"],
    }

    return price_fig, osc_fig, diag


# ============================================================
# LLM
# ============================================================
def build_llm_prompt(
    snapshot: Dict[str, float],
    prev_snapshot: Optional[Dict[str, float]],
    thresholds: Dict[str, float],
    upload_history_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    core_momentum_df: pd.DataFrame,
    extended_momentum_df: pd.DataFrame,
) -> str:
    stage_num, stage_name = breadth_stage(snapshot, thresholds)
    repair = qwen_repair_trigger(snapshot, thresholds)
    confluence = breadth_confluence_score(snapshot, thresholds)
    bounce_score, notes = bounce_quality(snapshot, prev_snapshot, thresholds)
    scenarios = scenario_scores(snapshot, thresholds)
    failed_risk, short_probe_ok, failed_reasons = failed_bounce_risk(snapshot, prev_snapshot, thresholds)

    recent_uploads = upload_history_df.tail(5).to_dict(orient="records") if not upload_history_df.empty else []
    weekly_rows = weekly_df.to_dict(orient="records") if not weekly_df.empty else []

    prompt = f"""
You are a market breadth regime analyst.

Interpret this structured breadth dashboard.

Rules:
- distinguish oversold bounce vs true repair vs failed bounce risk
- discuss immediate delta vs prior upload
- discuss short trend from recent uploads
- discuss weekly trend
- give separate guidance for:
  1. existing long holder
  2. new long entry
  3. leverage long
  4. failed-bounce short probe
- use the momentum context tables heavily
- do not invent probabilities; use weighted scenario language
- be concise but specific

Current state:
- Stage: {stage_num} ({stage_name})
- Repair trigger: {repair}
- Breadth Confluence Score: {confluence}/10
- Bounce Quality Score: {bounce_score}/9
- Failed-bounce risk: {failed_risk}
- Short probe eligible: {short_probe_ok}
- Bounce notes: {notes}
- Failed-bounce reasons: {failed_reasons}
- Scenario scores: {scenarios}

Core momentum context:
{core_momentum_df.to_json(orient="records", indent=2)}

Extended momentum context:
{extended_momentum_df.to_json(orient="records", indent=2)}

Recent upload history:
{json.dumps(recent_uploads, indent=2)}

Weekly table:
{json.dumps(weekly_rows, indent=2)}

Return sections:
- Executive Summary
- Immediate Delta
- Short Trend
- Weekly Context
- Action Guidance
- What Confirms Repair
- What Invalidates Bounce
""".strip()

    return prompt


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Baseline / Uploads")
    baseline_file = st.file_uploader("Upload historical baseline CSV", type=["csv"])
    realtime_file = st.file_uploader("Upload new realtime / EOD snapshot CSV", type=["csv"])
    upload_tag = st.text_input("Upload tag", value="manual upload")

    st.markdown("---")
    st.subheader("Thresholds")
    thresholds = DEFAULT_THRESHOLDS.copy()
    thresholds["BPSPX_BB_LIFT"] = st.number_input("BPSPX %B repair lift", value=float(thresholds["BPSPX_BB_LIFT"]), step=0.01, format="%.2f")
    thresholds["SPXA50R_REPAIR"] = st.number_input("SPXA50R repair threshold", value=float(thresholds["SPXA50R_REPAIR"]), step=1.0, format="%.1f")
    thresholds["BPSPX_CONFIRM"] = st.number_input("BPSPX confirm threshold", value=float(thresholds["BPSPX_CONFIRM"]), step=1.0, format="%.1f")
    thresholds["NYMO_HEALTHY"] = st.number_input("NYMO healthy threshold", value=float(thresholds["NYMO_HEALTHY"]), step=5.0, format="%.1f")
    thresholds["NYSI_LEVERAGE_OK"] = st.number_input("NYSI leverage threshold", value=float(thresholds["NYSI_LEVERAGE_OK"]), step=5.0, format="%.1f")

    st.markdown("---")
    st.subheader("Display")
    show_charts = st.multiselect("Minimal charts", ALL_SYMBOLS + ["$BPSPX"], default=DEFAULT_MINIMAL_CHARTS)

    st.markdown("---")
    st.subheader("OpenAI / ChatGPT")
    st.text_input("Manual API key (optional)", type="password", key="manual_api_key")
    run_llm = st.checkbox("Run ChatGPT analysis if key exists", value=False)
    show_prompt = st.checkbox("Show prompt", value=False)

    st.markdown("---")
    if st.button("Clear stored baseline"):
        if BASELINE_PATH.exists():
            BASELINE_PATH.unlink()
        if BASELINE_META_PATH.exists():
            BASELINE_META_PATH.unlink()
        st.success("Stored baseline cleared.")

    if st.button("Clear upload history"):
        if UPLOAD_HISTORY_PATH.exists():
            UPLOAD_HISTORY_PATH.unlink()
        st.success("Upload history cleared.")

    if st.button("Clear LLM history"):
        if LLM_HISTORY_PATH.exists():
            LLM_HISTORY_PATH.unlink()
        st.success("LLM history cleared.")


# ============================================================
# BASELINE MEMORY
# ============================================================
baseline_meta = load_json(BASELINE_META_PATH, default={})

if baseline_file is not None:
    try:
        parsed_baseline = parse_stockcharts_historical_from_bytes(baseline_file.getvalue())
        parsed_baseline.to_parquet(BASELINE_PATH, index=False)
        baseline_meta = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "source_filename": baseline_file.name,
            "rows": int(len(parsed_baseline)),
            "symbols": int(parsed_baseline["symbol"].nunique()),
        }
        save_json(BASELINE_META_PATH, baseline_meta)
        st.success("Historical baseline saved.")
    except Exception as e:
        st.error(f"Could not save baseline: {e}")

if not BASELINE_PATH.exists():
    st.info("Upload and save the historical baseline first.")
    st.stop()

try:
    hist_long = pd.read_parquet(BASELINE_PATH)
    hist_feat = add_indicator_features(hist_long)
except Exception as e:
    st.error(f"Could not load stored baseline: {e}")
    st.stop()

baseline_snapshot = get_latest_feature_snapshot(hist_feat)
baseline_prior_snapshot = get_prior_feature_snapshot(hist_feat)
weekly_df = weekly_analysis(hist_feat)
upload_history_df = load_upload_history()


# ============================================================
# CURRENT SNAPSHOT
# ============================================================
snapshot = baseline_snapshot.copy()
prev_snapshot = baseline_prior_snapshot.copy()

realtime_df = pd.DataFrame()

if realtime_file is not None:
    try:
        realtime_df = parse_realtime_snapshot_from_bytes(realtime_file.getvalue())

        for _, row in realtime_df.iterrows():
            sym = row["Symbol"]
            close_val = safe_float(row["Close"])
            if pd.notna(close_val):
                snapshot[sym] = close_val

        snapshot["NYMO_PROXY"] = compute_nymo_proxy(snapshot, thresholds, prev_snapshot)

        summary_row = build_upload_summary(snapshot, thresholds, upload_tag, prev_snapshot)
        append_upload_history(summary_row)
        upload_history_df = load_upload_history()

        st.success("Realtime snapshot processed and appended to upload history.")
    except Exception as e:
        st.error(f"Could not parse realtime snapshot: {e}")
        st.stop()
else:
    snapshot["NYMO_PROXY"] = compute_nymo_proxy(snapshot, thresholds, prev_snapshot)


# ============================================================
# CALCULATIONS
# ============================================================
stage_num, stage_name = breadth_stage(snapshot, thresholds)
repair_trigger = qwen_repair_trigger(snapshot, thresholds)
bounce_score, bounce_notes = bounce_quality(snapshot, prev_snapshot, thresholds)
confluence_score = breadth_confluence_score(snapshot, thresholds)
scenarios = scenario_scores(snapshot, thresholds)
failed_risk, short_probe_ok, failed_reasons = failed_bounce_risk(snapshot, prev_snapshot, thresholds)

recent_uploads_df = latest_upload_rows(5)
intraday_df = intraday_multi_upload_analysis(recent_uploads_df)

actions_df = pd.DataFrame(
    [
        {
            "Use Case": "Existing long hold",
            "Status": "YES" if stage_num >= 1 else "CAUTION",
            "Guidance": "Hold only if bounce is improving and confluence is not deteriorating.",
        },
        {
            "Use Case": "New long entry",
            "Status": "YES" if repair_trigger and snapshot.get("$BPSPX", np.nan) > thresholds["BPSPX_CONFIRM"] else "NO / EARLY",
            "Guidance": "Prefer repair trigger + participation confirmation.",
        },
        {
            "Use Case": "Leverage long",
            "Status": "YES" if repair_trigger and snapshot.get("$BPSPX", np.nan) > thresholds["BPSPX_CONFIRM"] and snapshot.get("$NYSI", np.nan) > thresholds["NYSI_LEVERAGE_OK"] else "NO",
            "Guidance": "Reserve leverage for broader confirmation.",
        },
        {
            "Use Case": "Short probe",
            "Status": "ONLY FAILED BOUNCE" if short_probe_ok else "LESS ATTRACTIVE",
            "Guidance": "Prefer failed bounce / rejection, not fresh panic lows.",
        },
    ]
)

core_momentum_symbols = ["$BPSPX_%B"] + CORE_SYMBOLS
core_momentum_df = build_momentum_context_table(core_momentum_symbols, snapshot, prev_snapshot, thresholds)
extended_momentum_df = build_momentum_context_table(EXTENDED_SYMBOLS, snapshot, prev_snapshot, thresholds)


# ============================================================
# SUMMARY
# ============================================================
st.subheader("Regime Summary")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("BPSPX", fmt_num(snapshot.get("$BPSPX", np.nan)))
c2.metric("BPSPX %B", fmt_num(snapshot.get("$BPSPX_%B", np.nan)))
c3.metric("SPXA50R", fmt_num(snapshot.get("$SPXA50R", np.nan)))
c4.metric("NYMO", fmt_num(snapshot.get("$NYMO", np.nan)))
c5.metric("NYSI", fmt_num(snapshot.get("$NYSI", np.nan)))
c6.metric("NYMO Proxy", fmt_num(snapshot.get("NYMO_PROXY", np.nan)))

d1, d2, d3, d4, d5 = st.columns(5)
d1.metric("Breadth Confluence", f"{confluence_score}/10")
d2.metric("Bounce Quality", f"{bounce_score}/9")
d3.metric("Regime Stage", f"{stage_num}")
d4.metric("Repair Trigger", "ON" if repair_trigger else "OFF")
d5.metric("Failed Bounce Risk", failed_risk)

st.write(build_narrative(snapshot, prev_snapshot, thresholds))


# ============================================================
# SCENARIOS
# ============================================================
st.subheader("Scenario Scores")
s1, s2, s3 = st.columns(3)
s1.metric("Bearish continuation", scenarios["Bearish continuation"])
s2.metric("Oversold bounce", scenarios["Oversold bounce"])
s3.metric("True repair", scenarios["True repair"])
st.caption("These are weighted scenario scores, not statistical probabilities.")


# ============================================================
# ACTIONS
# ============================================================
st.subheader("Action Dashboard")
st.dataframe(actions_df, use_container_width=True, hide_index=True)


# ============================================================
# MOMENTUM CONTEXT
# ============================================================
st.subheader("Momentum Context — Core")
st.dataframe(core_momentum_df, use_container_width=True, hide_index=True)

st.subheader("Momentum Context — Extended")
st.dataframe(extended_momentum_df, use_container_width=True, hide_index=True)


# ============================================================
# FAILED BOUNCE
# ============================================================
st.subheader("Failed-Bounce Short Framework")
f1, f2 = st.columns(2)
f1.metric("Short Probe Eligible", "YES" if short_probe_ok else "NO")
f2.metric("Failed-Bounce Risk", failed_risk)
if failed_reasons:
    st.write("Drivers: " + "; ".join(failed_reasons))


# ============================================================
# MULTI-UPLOAD
# ============================================================
st.subheader("Intraday / Multi-Upload Analysis")
if intraday_df.empty:
    st.info("Need at least two saved uploads to compare current vs prior upload.")
else:
    st.dataframe(intraday_df, use_container_width=True, hide_index=True)

if not recent_uploads_df.empty:
    st.subheader("Recent Upload History")
    st.dataframe(recent_uploads_df, use_container_width=True, hide_index=True)


# ============================================================
# WEEKLY
# ============================================================
st.subheader("Weekly Analysis")
if weekly_df.empty:
    st.info("Not enough historical data for weekly analysis.")
else:
    st.dataframe(weekly_df, use_container_width=True, hide_index=True)


# ============================================================
# REALTIME SNAPSHOT
# ============================================================
st.subheader("Current Realtime Snapshot")
if realtime_df.empty:
    st.info("No new realtime snapshot uploaded this run. Using stored historical baseline.")
else:
    show_cols = [c for c in ["Symbol", "Close", "PctChange"] if c in realtime_df.columns]
    st.dataframe(realtime_df[show_cols], use_container_width=True, hide_index=True)


# ============================================================
# CHARTS
# ============================================================
st.subheader("Minimal Chart Panel")
for sym in show_charts:
    price_fig, osc_fig, diag = make_symbol_chart(hist_feat, sym, thresholds["BPSPX_BB_LIFT"])
    if price_fig is None:
        continue

    with st.expander(sym, expanded=sym in ["$BPSPX", "$SPXA50R"]):
        left, right = st.columns([1.7, 1.0])
        with left:
            st.plotly_chart(price_fig, use_container_width=True)
            st.plotly_chart(osc_fig, use_container_width=True)
        with right:
            st.json(diag)


# ============================================================
# CHATGPT
# ============================================================
st.subheader("ChatGPT Interpretation")

prompt = build_llm_prompt(
    snapshot,
    prev_snapshot,
    thresholds,
    upload_history_df,
    weekly_df,
    core_momentum_df,
    extended_momentum_df,
)

if show_prompt:
    st.text_area("Prompt", prompt, height=300)

client, api_key, model_name = get_openai_client_and_model()
use_live_llm = run_llm and client is not None

if use_live_llm:
    try:
        with st.spinner(f"Running ChatGPT analysis with {model_name}..."):
            llm_text = call_openai_analysis(client, model_name, prompt)
        st.markdown(llm_text)

        append_jsonl(
            LLM_HISTORY_PATH,
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model": model_name,
                "prompt": prompt,
                "response": llm_text,
            },
        )
    except Exception as e:
        st.error(f"OpenAI API call failed: {e}")
        st.markdown("### Fallback Rule-Based Interpretation")
        st.write(build_narrative(snapshot, prev_snapshot, thresholds))
else:
    st.markdown("### Rule-Based Interpretation")
    st.write(build_narrative(snapshot, prev_snapshot, thresholds))
    st.caption("Live ChatGPT analysis is optional. Add an API key in secrets, env var, or the sidebar field.")


# ============================================================
# BASELINE STATUS
# ============================================================
st.subheader("Stored Baseline Status")
st.json(baseline_meta if baseline_meta else {"status": "No metadata found"})


# ============================================================
# DOWNLOADS
# ============================================================
st.subheader("Downloads")
parsed_csv = hist_feat.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download parsed historical features",
    data=parsed_csv,
    file_name="parsed_breadth_features.csv",
    mime="text/csv",
)

upload_hist_df = load_upload_history()
if not upload_hist_df.empty:
    st.download_button(
        "Download upload history",
        data=upload_hist_df.to_csv(index=False).encode("utf-8"),
        file_name="breadth_upload_history.csv",
        mime="text/csv",
    )

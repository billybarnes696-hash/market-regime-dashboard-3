import io
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE
# ============================================================
st.set_page_config(page_title="Breadth Engine Dashboard", layout="wide")
st.title("Breadth Engine Dashboard")
st.caption("Breadth regime engine: historical + realtime overlay + action map")


# ============================================================
# THRESHOLDS
# ============================================================
THRESHOLDS = {
    "BPSPX_BB_LIFT": 0.20,
    "SPXA50R_REPAIR": 30.0,
    "BPSPX_CONFIRM": 45.0,
    "NYMO_HEALTHY": -20.0,
    "NYMO_WASHOUT": -80.0,
    "NYSI_TREND_OK": 0.0,
    "NYSI_LEVERAGE_OK": -50.0,
    "NYAD_THRUST": 1500.0,
    "SPXADP_THRUST": 60.0,
    "BPSPX_DIR_STRONG": 3.0,
    "CPCE_FEAR": 0.80,
    "NYHL_POSITIVE": 0.0,
}


# ============================================================
# CONSTANTS
# ============================================================
OHLC_PATTERN = re.compile(
    r"(?P<day>Mon|Tue|Wed|Thu|Fri)\s+"
    r"(?P<date>\d{2}-\d{2}-\d{4})\s+"
    r"(?P<open>-?\d+(?:\.\d+)?)\s+"
    r"(?P<high>-?\d+(?:\.\d+)?)\s+"
    r"(?P<low>-?\d+(?:\.\d+)?)\s+"
    r"(?P<close>-?\d+(?:\.\d+)?)\s+"
    r"(?P<volume>-?\d+(?:\.\d+)?)"
)

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

DEFAULT_CHARTS = ["$BPSPX", "$SPXA50R", "$NYMO", "$NYSI", "$NYAD", "$NYHL", "RSP", "VXX"]


# ============================================================
# UTILS
# ============================================================
def normalize_symbol(sym: str) -> str:
    return str(sym).strip()


def fmt_num(val, digits: int = 2) -> str:
    if pd.isna(val):
        return "n/a"
    return f"{float(val):.{digits}f}"


def safe_float(val) -> float:
    try:
        return float(val)
    except Exception:
        return np.nan


def bool_py(x) -> bool:
    if pd.isna(x):
        return False
    return bool(x)


def get_snapshot_val(snapshot: Dict[str, float], key: str) -> float:
    return snapshot.get(key, np.nan)


# ============================================================
# FILE PARSERS
# ============================================================
@st.cache_data(show_spinner=False)
def parse_historical_raw_from_bytes(file_bytes: bytes) -> pd.DataFrame:
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
        raise ValueError(
            "Historical CSV could not be parsed. Make sure it is the StockCharts-style 1-column export."
        )

    df = pd.DataFrame(records)
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_realtime_table_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))

    if "Symbol" not in df.columns:
        raise ValueError("Realtime CSV must contain a 'Symbol' column.")

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
        raise ValueError(
            "Realtime CSV must contain one of these columns: "
            + ", ".join(close_candidates)
        )

    df["Close"] = pd.to_numeric(df[close_col], errors="coerce")

    change_candidates = [
        "Daily PctChange(1,Daily Close)",
        "Pct Change",
        "% Change",
        "Change %",
        "Daily Change %",
    ]
    found_change = next((c for c in change_candidates if c in df.columns), None)
    if found_change is not None:
        df["PctChange"] = pd.to_numeric(df[found_change], errors="coerce")
    else:
        df["PctChange"] = np.nan

    return df


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


def true_strength_index(
    series: pd.Series,
    long: int = 25,
    short: int = 13,
    signal: int = 7,
) -> Tuple[pd.Series, pd.Series]:
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
        g["tsi_slow"], g["tsi_slow_sig"] = true_strength_index(g["close"], 25, 13, 7)

        g["ma20"] = g["close"].rolling(20).mean()
        g["ma50"] = g["close"].rolling(50).mean()
        g["ma200"] = g["close"].rolling(200).mean()

        g["daily_change"] = g["close"].diff()
        g["daily_pct"] = g["close"].pct_change() * 100

        frames.append(g)

    return pd.concat(frames, ignore_index=True)


def wide_from_long(hist: pd.DataFrame) -> pd.DataFrame:
    return hist.pivot(index="date", columns="symbol", values="close").sort_index()


# ============================================================
# PATTERN DETECTION
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


def detect_bear_kiss(g: pd.DataFrame, ma_col: str = "ma50") -> pd.Series:
    ma = g[ma_col]
    return ((g["high"] >= ma * 0.995) & (g["close"] < ma)).fillna(False)


def detect_bull_kiss(g: pd.DataFrame, ma_col: str = "ma50") -> pd.Series:
    ma = g[ma_col]
    return ((g["low"] <= ma * 1.005) & (g["close"] > ma)).fillna(False)


# ============================================================
# CORE BREADTH LOGIC
# ============================================================
def score_nyad(val: float) -> int:
    if pd.isna(val):
        return 0
    if val > THRESHOLDS["NYAD_THRUST"]:
        return 2
    if val > 300:
        return 1
    if val < -1500:
        return -2
    if val < -300:
        return -1
    return 0


def score_spxadp(val: float) -> int:
    if pd.isna(val):
        return 0
    if val > THRESHOLDS["SPXADP_THRUST"]:
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
    if diff >= THRESHOLDS["BPSPX_DIR_STRONG"]:
        return 2
    if diff > 0:
        return 1
    if diff <= -THRESHOLDS["BPSPX_DIR_STRONG"]:
        return -2
    if diff < 0:
        return -1
    return 0


def compute_nymo_proxy(nyad: float, spxadp: float, bpspx_dir: int = 0) -> float:
    return 0.5 * score_nyad(nyad) + 0.3 * score_spxadp(spxadp) + 0.2 * bpspx_dir


def qwen_repair_trigger(snapshot: Dict[str, float]) -> bool:
    bpspx_bb = get_snapshot_val(snapshot, "$BPSPX_%B")
    spxa50r = get_snapshot_val(snapshot, "$SPXA50R")
    return bool(
        pd.notna(bpspx_bb)
        and pd.notna(spxa50r)
        and bpspx_bb > THRESHOLDS["BPSPX_BB_LIFT"]
        and spxa50r > THRESHOLDS["SPXA50R_REPAIR"]
    )


def breadth_stage(snapshot: Dict[str, float]) -> Tuple[int, str]:
    bpspx_bb = get_snapshot_val(snapshot, "$BPSPX_%B")
    spxa50r = get_snapshot_val(snapshot, "$SPXA50R")
    bpspx = get_snapshot_val(snapshot, "$BPSPX")
    nysi = get_snapshot_val(snapshot, "$NYSI")

    if pd.isna(bpspx_bb):
        return 0, "Unknown"

    if bpspx_bb < THRESHOLDS["BPSPX_BB_LIFT"]:
        return 0, "Washout / lower-band pressure"
    if bpspx_bb >= THRESHOLDS["BPSPX_BB_LIFT"] and (pd.isna(spxa50r) or spxa50r <= THRESHOLDS["SPXA50R_REPAIR"]):
        return 1, "Bounce starting"
    if (
        bpspx_bb > THRESHOLDS["BPSPX_BB_LIFT"]
        and pd.notna(spxa50r)
        and spxa50r > THRESHOLDS["SPXA50R_REPAIR"]
        and (pd.isna(bpspx) or bpspx <= THRESHOLDS["BPSPX_CONFIRM"])
    ):
        return 2, "Breadth repair"
    if (
        bpspx_bb > THRESHOLDS["BPSPX_BB_LIFT"]
        and pd.notna(spxa50r)
        and spxa50r > THRESHOLDS["SPXA50R_REPAIR"]
        and pd.notna(bpspx)
        and bpspx > THRESHOLDS["BPSPX_CONFIRM"]
        and (pd.isna(nysi) or nysi <= THRESHOLDS["NYSI_TREND_OK"])
    ):
        return 3, "Participation confirmation"
    if (
        bpspx_bb > THRESHOLDS["BPSPX_BB_LIFT"]
        and pd.notna(spxa50r)
        and spxa50r > THRESHOLDS["SPXA50R_REPAIR"]
        and pd.notna(bpspx)
        and bpspx > THRESHOLDS["BPSPX_CONFIRM"]
        and pd.notna(nysi)
        and nysi > THRESHOLDS["NYSI_TREND_OK"]
    ):
        return 4, "Trend durability / healthier regime"

    return 0, "Unknown"


def classify_market_state(snapshot: Dict[str, float]) -> str:
    bpspx = get_snapshot_val(snapshot, "$BPSPX")
    spxa50r = get_snapshot_val(snapshot, "$SPXA50R")
    nymo = get_snapshot_val(snapshot, "$NYMO")
    proxy = get_snapshot_val(snapshot, "NYMO_PROXY")
    bpspx_bb = get_snapshot_val(snapshot, "$BPSPX_%B")

    if pd.notna(nymo) and nymo < THRESHOLDS["NYMO_WASHOUT"]:
        return "capitulation washout"
    if pd.notna(bpspx_bb) and bpspx_bb < THRESHOLDS["BPSPX_BB_LIFT"] and pd.notna(bpspx) and bpspx < THRESHOLDS["BPSPX_CONFIRM"]:
        return "oversold / lower-band pressure"
    if qwen_repair_trigger(snapshot) and pd.notna(bpspx) and bpspx < THRESHOLDS["BPSPX_CONFIRM"]:
        return "early breadth repair"
    if pd.notna(proxy) and proxy >= 1.5 and pd.notna(bpspx) and bpspx < THRESHOLDS["BPSPX_CONFIRM"]:
        return "breadth thrust / early repair"
    if (
        pd.notna(bpspx)
        and pd.notna(spxa50r)
        and bpspx >= THRESHOLDS["BPSPX_CONFIRM"]
        and spxa50r >= THRESHOLDS["SPXA50R_REPAIR"]
        and (pd.isna(nymo) or nymo > THRESHOLDS["NYMO_HEALTHY"])
    ):
        return "confirmed trend repair"
    if (
        pd.notna(bpspx)
        and pd.notna(spxa50r)
        and bpspx < THRESHOLDS["BPSPX_CONFIRM"]
        and spxa50r < THRESHOLDS["SPXA50R_REPAIR"]
    ):
        return "bearish bounce / incomplete repair"
    return "repair phase"


def bounce_quality(snapshot: Dict[str, float], prev_snapshot: Optional[Dict[str, float]] = None):
    notes = []
    score = 0

    bpspx = get_snapshot_val(snapshot, "$BPSPX")
    spxa50r = get_snapshot_val(snapshot, "$SPXA50R")
    nyad = get_snapshot_val(snapshot, "$NYAD")
    spxadp = get_snapshot_val(snapshot, "$SPXADP")
    vxx = get_snapshot_val(snapshot, "VXX")
    rsp_spy = get_snapshot_val(snapshot, "RSP:SPY")
    bpspx_bb = get_snapshot_val(snapshot, "$BPSPX_%B")
    cpce = get_snapshot_val(snapshot, "$CPCE")

    if pd.notna(bpspx_bb):
        if bpspx_bb > THRESHOLDS["BPSPX_BB_LIFT"]:
            score += 1
            notes.append("BPSPX Bollinger %B > 0.20")
        else:
            notes.append("BPSPX Bollinger %B still pinned low")

    if pd.notna(bpspx):
        if bpspx >= THRESHOLDS["BPSPX_CONFIRM"]:
            score += 2
            notes.append("Participation >45")
        elif bpspx >= 40:
            score += 1
            notes.append("Participation improving")

    if pd.notna(spxa50r):
        if spxa50r > THRESHOLDS["SPXA50R_REPAIR"]:
            score += 2
            notes.append("SPXA50R >30")
        elif spxa50r >= 25:
            score += 1
            notes.append("Depth improving")

    if pd.notna(nyad) and pd.notna(spxadp):
        if nyad > THRESHOLDS["NYAD_THRUST"] and spxadp > THRESHOLDS["SPXADP_THRUST"]:
            score += 2
            notes.append("Strong breadth thrust")
        elif nyad > 500 or spxadp > 20:
            score += 1
            notes.append("Moderate thrust")

    if pd.notna(cpce) and cpce >= THRESHOLDS["CPCE_FEAR"]:
        score += 1
        notes.append("CPCE fear support")

    if prev_snapshot and pd.notna(vxx):
        prev_vxx = prev_snapshot.get("VXX", np.nan)
        if pd.notna(prev_vxx) and vxx < prev_vxx:
            score += 1
            notes.append("Volatility easing")

    if prev_snapshot and pd.notna(rsp_spy):
        prev_ratio = prev_snapshot.get("RSP:SPY", np.nan)
        if pd.notna(prev_ratio) and rsp_spy >= prev_ratio:
            score += 1
            notes.append("Equal-weight not lagging")

    return min(score, 9), notes


def scenario_scores(snapshot: Dict[str, float]) -> Dict[str, int]:
    bpspx = get_snapshot_val(snapshot, "$BPSPX")
    bpspx_bb = get_snapshot_val(snapshot, "$BPSPX_%B")
    spxa50r = get_snapshot_val(snapshot, "$SPXA50R")
    nymo = get_snapshot_val(snapshot, "$NYMO")
    nysi = get_snapshot_val(snapshot, "$NYSI")
    nyad = get_snapshot_val(snapshot, "$NYAD")
    spxadp = get_snapshot_val(snapshot, "$SPXADP")
    cpce = get_snapshot_val(snapshot, "$CPCE")

    bearish_cont = 0
    oversold_bounce = 0
    true_repair = 0

    if pd.notna(nysi) and nysi < THRESHOLDS["NYSI_LEVERAGE_OK"]:
        bearish_cont += 3
    if pd.notna(nymo) and nymo < THRESHOLDS["NYMO_HEALTHY"]:
        bearish_cont += 2
    if pd.notna(bpspx) and bpspx < THRESHOLDS["BPSPX_CONFIRM"]:
        bearish_cont += 2
    if pd.notna(spxa50r) and spxa50r < THRESHOLDS["SPXA50R_REPAIR"]:
        bearish_cont += 2

    if pd.notna(cpce) and cpce >= THRESHOLDS["CPCE_FEAR"]:
        oversold_bounce += 2
    if pd.notna(bpspx_bb) and bpspx_bb < THRESHOLDS["BPSPX_BB_LIFT"]:
        oversold_bounce += 2
    if pd.notna(nymo) and nymo < 0:
        oversold_bounce += 1
    if pd.notna(spxa50r) and spxa50r >= 25:
        oversold_bounce += 1

    if qwen_repair_trigger(snapshot):
        true_repair += 3
    if pd.notna(bpspx) and bpspx > THRESHOLDS["BPSPX_CONFIRM"]:
        true_repair += 2
    if pd.notna(nymo) and nymo > THRESHOLDS["NYMO_HEALTHY"]:
        true_repair += 1
    if pd.notna(nysi) and nysi > THRESHOLDS["NYSI_TREND_OK"]:
        true_repair += 2
    if pd.notna(nyad) and pd.notna(spxadp) and nyad > THRESHOLDS["NYAD_THRUST"] and spxadp > THRESHOLDS["SPXADP_THRUST"]:
        true_repair += 2

    return {
        "Bearish continuation score": bearish_cont,
        "Oversold bounce score": oversold_bounce,
        "True repair score": true_repair,
    }


def what_changed_today(snapshot: Dict[str, float], prev_snapshot: Optional[Dict[str, float]]) -> pd.DataFrame:
    if not prev_snapshot:
        return pd.DataFrame(columns=["Indicator", "Current", "Prior", "Delta", "Interpretation"])

    rows = []
    watch = [
        "$BPSPX",
        "$BPSPX_%B",
        "$SPXA50R",
        "$NYMO",
        "$NYSI",
        "$NYAD",
        "$SPXADP",
        "$CPCE",
        "RSP:SPY",
        "VXX",
    ]

    for key in watch:
        cur = get_snapshot_val(snapshot, key)
        prev = get_snapshot_val(prev_snapshot, key)
        if pd.isna(cur) and pd.isna(prev):
            continue

        delta = np.nan
        if pd.notna(cur) and pd.notna(prev):
            delta = cur - prev

        interp = ""
        if key == "$BPSPX_%B":
            if pd.notna(cur):
                if cur > THRESHOLDS["BPSPX_BB_LIFT"]:
                    interp = "Lifted off lower band"
                else:
                    interp = "Still lower-band pressure"
        elif key == "$SPXA50R":
            if pd.notna(cur):
                if cur > THRESHOLDS["SPXA50R_REPAIR"]:
                    interp = "Repair threshold passed"
                elif cur >= 25:
                    interp = "Improving but incomplete"
                else:
                    interp = "Weak breadth depth"
        elif key == "$NYMO":
            if pd.notna(cur):
                if cur > THRESHOLDS["NYMO_HEALTHY"]:
                    interp = "Near healthier momentum"
                elif cur < THRESHOLDS["NYMO_WASHOUT"]:
                    interp = "Washout momentum"
                else:
                    interp = "Still negative momentum"
        elif key == "$NYSI":
            if pd.notna(cur):
                interp = "Trend durability improving" if cur > THRESHOLDS["NYSI_TREND_OK"] else "Trend still damaged"
        elif key == "$CPCE":
            if pd.notna(cur):
                interp = "Fear support present" if cur >= THRESHOLDS["CPCE_FEAR"] else "Fear support fading"
        elif key == "VXX":
            if pd.notna(delta):
                interp = "Vol easing" if delta < 0 else "Vol rising"
        elif key == "RSP:SPY":
            if pd.notna(delta):
                interp = "EW improving" if delta >= 0 else "EW lagging"
        elif key == "$NYAD":
            if pd.notna(cur):
                interp = "Potential thrust" if cur > THRESHOLDS["NYAD_THRUST"] else "No thrust"
        elif key == "$SPXADP":
            if pd.notna(cur):
                interp = "Potential thrust" if cur > THRESHOLDS["SPXADP_THRUST"] else "No thrust"

        rows.append(
            {
                "Indicator": key,
                "Current": None if pd.isna(cur) else round(float(cur), 2),
                "Prior": None if pd.isna(prev) else round(float(prev), 2),
                "Delta": None if pd.isna(delta) else round(float(delta), 2),
                "Interpretation": interp,
            }
        )

    return pd.DataFrame(rows)


def build_momentum_context(snapshot: Dict[str, float], prev_snapshot: Optional[Dict[str, float]]) -> pd.DataFrame:
    rows = []

    def classify(indicator: str, value: float, delta: float) -> str:
        if pd.isna(value):
            return "Missing"
        if indicator == "$NYMO":
            if value < THRESHOLDS["NYMO_WASHOUT"]:
                return "Washout"
            if value < THRESHOLDS["NYMO_HEALTHY"]:
                return "Negative"
            return "Healthy"
        if indicator == "$NYSI":
            if value < THRESHOLDS["NYSI_LEVERAGE_OK"]:
                return "Damaged"
            if value < THRESHOLDS["NYSI_TREND_OK"]:
                return "Improving but negative"
            return "Positive"
        if indicator == "$BPSPX_%B":
            if value < THRESHOLDS["BPSPX_BB_LIFT"]:
                return "Pinned low"
            return "Lifted"
        if indicator == "$SPXA50R":
            if value < 25:
                return "Weak"
            if value <= THRESHOLDS["SPXA50R_REPAIR"]:
                return "Near repair"
            return "Repair passed"
        return "Neutral"

    for key in ["$BPSPX_%B", "$SPXA50R", "$NYMO", "$NYSI", "$NYAD", "$SPXADP"]:
        cur = get_snapshot_val(snapshot, key)
        prev = get_snapshot_val(prev_snapshot or {}, key)
        delta = cur - prev if pd.notna(cur) and pd.notna(prev) else np.nan
        rows.append(
            {
                "Indicator": key,
                "Current": None if pd.isna(cur) else round(float(cur), 2),
                "Delta": None if pd.isna(delta) else round(float(delta), 2),
                "State": classify(key, cur, delta),
            }
        )

    return pd.DataFrame(rows)


def action_dashboard(snapshot: Dict[str, float]) -> pd.DataFrame:
    stage_num, stage_name = breadth_stage(snapshot)
    repair_on = qwen_repair_trigger(snapshot)
    bpspx = get_snapshot_val(snapshot, "$BPSPX")
    nysi = get_snapshot_val(snapshot, "$NYSI")
    nymo = get_snapshot_val(snapshot, "$NYMO")

    rows = []

    rows.append(
        {
            "Use Case": "New long entry",
            "Status": "YES" if repair_on and pd.notna(bpspx) and bpspx > THRESHOLDS["BPSPX_CONFIRM"] else "NO / EARLY",
            "Guidance": "Prefer only after repair trigger + participation confirmation.",
        }
    )

    rows.append(
        {
            "Use Case": "Existing long hold",
            "Status": "YES" if stage_num >= 1 else "CAUTION",
            "Guidance": "Hold only if bounce is improving; weak if lower-band pressure persists.",
        }
    )

    rows.append(
        {
            "Use Case": "Leverage long",
            "Status": "YES" if repair_on and pd.notna(bpspx) and bpspx > THRESHOLDS["BPSPX_CONFIRM"] and pd.notna(nysi) and nysi > THRESHOLDS["NYSI_LEVERAGE_OK"] else "NO",
            "Guidance": "Reserve leverage for broader confirmation and improving summation.",
        }
    )

    rows.append(
        {
            "Use Case": "New short entry",
            "Status": "WAIT" if stage_num <= 1 else "LESS ATTRACTIVE",
            "Guidance": "Prefer failed bounce / rejection setups rather than fresh panic lows.",
        }
    )

    rows.append(
        {
            "Use Case": "Probe long",
            "Status": "YES" if stage_num == 1 else "NO",
            "Guidance": "Small size only when bounce starts but full repair not yet confirmed.",
        }
    )

    rows.append(
        {
            "Use Case": "Add risk",
            "Status": "YES" if stage_num >= 3 and pd.notna(nymo) and nymo > THRESHOLDS["NYMO_HEALTHY"] else "NO",
            "Guidance": f"Current stage: {stage_num} ({stage_name}).",
        }
    )

    return pd.DataFrame(rows)


def build_narrative(snapshot: Dict[str, float], prev_snapshot: Optional[Dict[str, float]] = None) -> str:
    state = classify_market_state(snapshot)
    stage_num, stage_name = breadth_stage(snapshot)
    bq, notes = bounce_quality(snapshot, prev_snapshot)

    bpspx = get_snapshot_val(snapshot, "$BPSPX")
    bpspx_bb = get_snapshot_val(snapshot, "$BPSPX_%B")
    spxa50r = get_snapshot_val(snapshot, "$SPXA50R")
    nymo = get_snapshot_val(snapshot, "$NYMO")
    nysi = get_snapshot_val(snapshot, "$NYSI")
    nyad = get_snapshot_val(snapshot, "$NYAD")
    spxadp = get_snapshot_val(snapshot, "$SPXADP")

    parts = [f"State: {state}.", f"Stage {stage_num}: {stage_name}."]

    if pd.notna(bpspx_bb):
        parts.append(f"BPSPX Bollinger %B is {bpspx_bb:.2f}.")
        if bpspx_bb < THRESHOLDS["BPSPX_BB_LIFT"]:
            parts.append("Breadth is still pinned near the lower Bollinger zone.")
        else:
            parts.append("Breadth has lifted off the lower Bollinger zone.")

    if pd.notna(bpspx) and pd.notna(spxa50r):
        parts.append(f"BPSPX {bpspx:.2f} and SPXA50R {spxa50r:.2f} define participation and depth.")

    if pd.notna(nymo):
        parts.append(f"Official NYMO is {nymo:.2f}.")
    if pd.notna(nysi):
        parts.append(f"NYSI is {nysi:.2f}.")

    if qwen_repair_trigger(snapshot):
        parts.append("Qwen repair trigger is active: BPSPX %B > 0.20 and SPXA50R > 30.")
    else:
        parts.append("Qwen repair trigger is not active yet.")

    if pd.notna(nyad) and pd.notna(spxadp) and nyad > THRESHOLDS["NYAD_THRUST"] and spxadp > THRESHOLDS["SPXADP_THRUST"]:
        parts.append("Breadth thrust conditions are present.")

    parts.append(f"Bounce Quality Score: {bq}/9 ({'; '.join(notes) if notes else 'limited confirmation'}).")

    if pd.notna(bpspx) and pd.notna(spxa50r):
        if bpspx > THRESHOLDS["BPSPX_CONFIRM"] and spxa50r > THRESHOLDS["SPXA50R_REPAIR"]:
            parts.append("Broad repair is approaching stronger confirmation territory.")
        else:
            parts.append("Repair may be underway, but full confirmation is still missing.")

    return " ".join(parts)


def rank_signals(snapshot: Dict[str, float]) -> pd.DataFrame:
    rows = []

    def add(signal: str, status: str, detail: str, score: int):
        rows.append({"Signal": signal, "Status": status, "Detail": detail, "Score": score})

    bpspx = get_snapshot_val(snapshot, "$BPSPX")
    bpspx_bb = get_snapshot_val(snapshot, "$BPSPX_%B")
    spxa50r = get_snapshot_val(snapshot, "$SPXA50R")
    nymo = get_snapshot_val(snapshot, "$NYMO")
    nysi = get_snapshot_val(snapshot, "$NYSI")
    nyad = get_snapshot_val(snapshot, "$NYAD")
    spxadp = get_snapshot_val(snapshot, "$SPXADP")
    nyhl = get_snapshot_val(snapshot, "$NYHL")
    ratio = get_snapshot_val(snapshot, "RSP:SPY")

    add(
        "Bollinger lift",
        "PASS" if pd.notna(bpspx_bb) and bpspx_bb > THRESHOLDS["BPSPX_BB_LIFT"] else "WAIT",
        f"BPSPX %B {fmt_num(bpspx_bb)}",
        1 if pd.notna(bpspx_bb) and bpspx_bb > THRESHOLDS["BPSPX_BB_LIFT"] else 0,
    )
    add(
        "Depth repair",
        "PASS" if pd.notna(spxa50r) and spxa50r > THRESHOLDS["SPXA50R_REPAIR"] else "WAIT",
        f"SPXA50R {fmt_num(spxa50r)}",
        2 if pd.notna(spxa50r) and spxa50r > THRESHOLDS["SPXA50R_REPAIR"] else 0,
    )
    add(
        "Qwen repair trigger",
        "PASS" if qwen_repair_trigger(snapshot) else "WAIT",
        f"BPSPX %B {fmt_num(bpspx_bb)} and SPXA50R {fmt_num(spxa50r)}",
        2 if qwen_repair_trigger(snapshot) else 0,
    )
    add(
        "Participation",
        "PASS" if pd.notna(bpspx) and bpspx > THRESHOLDS["BPSPX_CONFIRM"] else "WAIT",
        f"BPSPX {fmt_num(bpspx)}",
        2 if pd.notna(bpspx) and bpspx > THRESHOLDS["BPSPX_CONFIRM"] else 0,
    )
    add(
        "Momentum thrust",
        "PASS" if pd.notna(nyad) and pd.notna(spxadp) and nyad > THRESHOLDS["NYAD_THRUST"] and spxadp > THRESHOLDS["SPXADP_THRUST"] else "PARTIAL",
        f"NYAD {fmt_num(nyad, 0)}, SPXADP {fmt_num(spxadp, 1)}",
        2 if pd.notna(nyad) and pd.notna(spxadp) and nyad > THRESHOLDS["NYAD_THRUST"] and spxadp > THRESHOLDS["SPXADP_THRUST"] else 1,
    )
    add(
        "Official NYMO",
        "PASS" if pd.notna(nymo) and nymo > THRESHOLDS["NYMO_HEALTHY"] else "PARTIAL",
        f"NYMO {fmt_num(nymo)}",
        2 if pd.notna(nymo) and nymo > THRESHOLDS["NYMO_HEALTHY"] else 1 if pd.notna(nymo) and nymo > -50 else 0,
    )
    add(
        "Durability",
        "PASS" if pd.notna(nysi) and nysi > THRESHOLDS["NYSI_TREND_OK"] else "WAIT",
        f"NYSI {fmt_num(nysi)}",
        2 if pd.notna(nysi) and nysi > THRESHOLDS["NYSI_TREND_OK"] else 0,
    )
    add(
        "Leadership",
        "PASS" if pd.notna(nyhl) and nyhl > THRESHOLDS["NYHL_POSITIVE"] else "WAIT",
        f"NYHL {fmt_num(nyhl)}, RSP:SPY {fmt_num(ratio, 3)}",
        1 if pd.notna(nyhl) and nyhl > THRESHOLDS["NYHL_POSITIVE"] else 0,
    )

    return pd.DataFrame(rows).sort_values(["Score", "Signal"], ascending=[False, True]).reset_index(drop=True)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Inputs")
    hist_file = st.file_uploader("Historical CSV", type=["csv"])
    rt_file = st.file_uploader("Realtime CSV", type=["csv"])
    show_symbols = st.multiselect("Charts", CORE_SYMBOLS, default=DEFAULT_CHARTS)

    st.markdown("---")
    st.subheader("Thresholds")
    THRESHOLDS["BPSPX_BB_LIFT"] = st.number_input("BPSPX %B repair lift", value=float(THRESHOLDS["BPSPX_BB_LIFT"]), step=0.01, format="%.2f")
    THRESHOLDS["SPXA50R_REPAIR"] = st.number_input("SPXA50R repair threshold", value=float(THRESHOLDS["SPXA50R_REPAIR"]), step=1.0, format="%.1f")
    THRESHOLDS["BPSPX_CONFIRM"] = st.number_input("BPSPX participation confirm", value=float(THRESHOLDS["BPSPX_CONFIRM"]), step=1.0, format="%.1f")
    THRESHOLDS["NYMO_HEALTHY"] = st.number_input("NYMO healthy threshold", value=float(THRESHOLDS["NYMO_HEALTHY"]), step=5.0, format="%.1f")
    THRESHOLDS["NYSI_LEVERAGE_OK"] = st.number_input("NYSI leverage threshold", value=float(THRESHOLDS["NYSI_LEVERAGE_OK"]), step=5.0, format="%.1f")

if hist_file is None or rt_file is None:
    st.info("Please upload both the historical CSV and the realtime CSV.")
    st.stop()


# ============================================================
# LOAD
# ============================================================
try:
    with st.spinner("Parsing files and building indicators..."):
        hist_long = parse_historical_raw_from_bytes(hist_file.getvalue())
        rt = load_realtime_table_from_bytes(rt_file.getvalue())
        hist_feat = add_indicator_features(hist_long)
        wide = wide_from_long(hist_long)
except Exception as e:
    st.error(f"Load failed: {e}")
    st.stop()

if hist_long.empty:
    st.error("Historical data is empty after parsing.")
    st.stop()

if wide.empty:
    st.error("Wide history could not be built.")
    st.stop()

latest_hist_date = wide.index.max()
prev_hist_date = wide.index[-2] if len(wide.index) > 1 else latest_hist_date

latest_hist = wide.loc[latest_hist_date].to_dict()
prev_hist = wide.loc[prev_hist_date].to_dict() if prev_hist_date is not None else {}

snapshot: Dict[str, float] = latest_hist.copy()
prev_snapshot: Dict[str, float] = prev_hist.copy()

# add realtime closes into current snapshot
for _, row in rt.iterrows():
    sym = row["Symbol"]
    close_val = pd.to_numeric(row["Close"], errors="coerce")
    if pd.notna(close_val):
        snapshot[sym] = float(close_val)

# add latest feature values into snapshot
latest_feat_rows = hist_feat.sort_values("date").groupby("symbol", as_index=False).tail(1)
for _, row in latest_feat_rows.iterrows():
    sym = row["symbol"]
    snapshot[f"{sym}_%B"] = row.get("pct_b20", np.nan)
    snapshot[f"{sym}_RSI14"] = row.get("rsi14", np.nan)
    snapshot[f"{sym}_CCI20"] = row.get("cci20", np.nan)
    snapshot[f"{sym}_TSI"] = row.get("tsi_fast", np.nan)

    prev_rows = hist_feat[hist_feat["symbol"] == sym].sort_values("date")
    if len(prev_rows) >= 2:
        prev_row = prev_rows.iloc[-2]
        prev_snapshot[f"{sym}_%B"] = prev_row.get("pct_b20", np.nan)
        prev_snapshot[f"{sym}_RSI14"] = prev_row.get("rsi14", np.nan)
        prev_snapshot[f"{sym}_CCI20"] = prev_row.get("cci20", np.nan)
        prev_snapshot[f"{sym}_TSI"] = prev_row.get("tsi_fast", np.nan)

bpspx_dir = derive_bpspx_direction(get_snapshot_val(snapshot, "$BPSPX"), get_snapshot_val(prev_snapshot, "$BPSPX"))
snapshot["NYMO_PROXY"] = compute_nymo_proxy(
    get_snapshot_val(snapshot, "$NYAD"),
    get_snapshot_val(snapshot, "$SPXADP"),
    bpspx_dir,
)
prev_snapshot["NYMO_PROXY"] = compute_nymo_proxy(
    get_snapshot_val(prev_snapshot, "$NYAD"),
    get_snapshot_val(prev_snapshot, "$SPXADP"),
    derive_bpspx_direction(get_snapshot_val(prev_snapshot, "$BPSPX"), np.nan),
)

stage_num, stage_name = breadth_stage(snapshot)
repair_trigger_on = qwen_repair_trigger(snapshot)
scenarios = scenario_scores(snapshot)
bq, bq_notes = bounce_quality(snapshot, prev_snapshot)

changed_df = what_changed_today(snapshot, prev_snapshot)
momentum_df = build_momentum_context(snapshot, prev_snapshot)
actions_df = action_dashboard(snapshot)
ranking_df = rank_signals(snapshot)


# ============================================================
# TOP METRICS
# ============================================================
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("BPSPX", fmt_num(get_snapshot_val(snapshot, "$BPSPX")))
c2.metric("BPSPX %B", fmt_num(get_snapshot_val(snapshot, "$BPSPX_%B")))
c3.metric("SPXA50R", fmt_num(get_snapshot_val(snapshot, "$SPXA50R")))
c4.metric("Official NYMO", fmt_num(get_snapshot_val(snapshot, "$NYMO")))
c5.metric("NYSI", fmt_num(get_snapshot_val(snapshot, "$NYSI")))
c6.metric("NYMO Proxy", fmt_num(get_snapshot_val(snapshot, "NYMO_PROXY")))

left, right = st.columns([1.3, 0.7])

with left:
    st.subheader("Market Narrative")
    st.write(build_narrative(snapshot, prev_snapshot))

    st.subheader("Signal Ranking")
    st.dataframe(ranking_df, use_container_width=True, hide_index=True)

with right:
    st.subheader("Regime Status")
    if repair_trigger_on:
        st.success("Qwen Repair Trigger: ACTIVE")
    else:
        st.warning("Qwen Repair Trigger: Not Active")

    st.info(f"Stage {stage_num}: {stage_name}")
    st.metric("Bounce Quality", f"{bq}/9")
    st.write("; ".join(bq_notes) if bq_notes else "Limited confirmation")


# ============================================================
# SCENARIO SCORES
# ============================================================
st.subheader("Scenario Scores")
score_cols = st.columns(3)
score_cols[0].metric("Bearish continuation", scenarios["Bearish continuation score"])
score_cols[1].metric("Oversold bounce", scenarios["Oversold bounce score"])
score_cols[2].metric("True repair", scenarios["True repair score"])
st.caption("These are weighted scenario scores, not statistical probabilities.")


# ============================================================
# ACTION DASHBOARD
# ============================================================
st.subheader("Action Dashboard")
st.dataframe(actions_df, use_container_width=True, hide_index=True)


# ============================================================
# MOMENTUM CONTEXT + CHANGES
# ============================================================
m1, m2 = st.columns(2)

with m1:
    st.subheader("Momentum Context")
    st.dataframe(momentum_df, use_container_width=True, hide_index=True)

with m2:
    st.subheader("What Changed vs Prior Day")
    st.dataframe(changed_df, use_container_width=True, hide_index=True)


# ============================================================
# REALTIME SNAPSHOT
# ============================================================
st.subheader("Realtime Snapshot")
cols_to_show = [c for c in ["Symbol", "Close", "PctChange"] if c in rt.columns]
if not cols_to_show:
    cols_to_show = ["Symbol", "Close"]
st.dataframe(rt[cols_to_show], use_container_width=True, hide_index=True)


# ============================================================
# READINESS
# ============================================================
st.subheader("Readiness")
r1, r2, r3 = st.columns(3)

rsp_ready = (
    repair_trigger_on
    and pd.notna(get_snapshot_val(snapshot, "$BPSPX"))
    and get_snapshot_val(snapshot, "$BPSPX") > THRESHOLDS["BPSPX_CONFIRM"]
)

ursp_ready = (
    repair_trigger_on
    and pd.notna(get_snapshot_val(snapshot, "$BPSPX"))
    and pd.notna(get_snapshot_val(snapshot, "$NYSI"))
    and get_snapshot_val(snapshot, "$BPSPX") > THRESHOLDS["BPSPX_CONFIRM"]
    and get_snapshot_val(snapshot, "$NYSI") > THRESHOLDS["NYSI_LEVERAGE_OK"]
)

short_probe_ok = (
    stage_num <= 1
    and pd.notna(get_snapshot_val(snapshot, "$NYSI"))
    and get_snapshot_val(snapshot, "$NYSI") < THRESHOLDS["NYSI_LEVERAGE_OK"]
)

with r1:
    if rsp_ready:
        st.success("RSP Swing Ready")
    else:
        st.warning("RSP Swing Not Fully Confirmed")

with r2:
    if ursp_ready:
        st.success("URSP / Leverage Ready")
    else:
        st.warning("URSP Not Ready")

with r3:
    if short_probe_ok:
        st.info("Short Probe Setup: only on failed bounce")
    else:
        st.info("Short Probe Setup: not favored now")


# ============================================================
# CHARTS
# ============================================================
st.subheader("Historical Charts + Oscillators")

for sym in show_symbols:
    g = hist_feat[hist_feat["symbol"] == sym].sort_values("date").copy()
    if g.empty:
        continue

    g["candle_pattern"] = g.apply(detect_candle_pattern, axis=1)
    g["bear_kiss_50"] = detect_bear_kiss(g, "ma50")
    g["bull_kiss_50"] = detect_bull_kiss(g, "ma50")

    last = g.iloc[-1]

    with st.expander(sym, expanded=sym in ["$BPSPX", "$SPXA50R", "RSP"]):
        col_a, col_b = st.columns([1.7, 1])

        with col_a:
            fig = go.Figure()
            fig.add_trace(
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
                fig.add_trace(go.Scatter(x=g["date"], y=g["ma20"], name="MA20"))
            if g["ma50"].notna().any():
                fig.add_trace(go.Scatter(x=g["date"], y=g["ma50"], name="MA50"))
            if g["ma200"].notna().any():
                fig.add_trace(go.Scatter(x=g["date"], y=g["ma200"], name="MA200"))

            fig.update_layout(
                height=360,
                xaxis_rangeslider_visible=False,
                margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

            osc = go.Figure()
            osc.add_trace(go.Scatter(x=g["date"], y=g["tsi_fast"], name="TSI(4,2,4)"))
            osc.add_trace(go.Scatter(x=g["date"], y=g["tsi_fast_sig"], name="TSI Signal"))
            osc.add_trace(go.Scatter(x=g["date"], y=g["cci20"], name="CCI20", visible="legendonly"))
            osc.add_trace(go.Scatter(x=g["date"], y=g["rsi14"], name="RSI14", visible="legendonly"))
            osc.add_trace(go.Scatter(x=g["date"], y=g["pct_b20"], name="Bollinger %B(20,2)"))
            osc.add_hline(y=1.0, line_dash="dash")
            osc.add_hline(y=0.5, line_dash="dot")
            osc.add_hline(y=THRESHOLDS["BPSPX_BB_LIFT"], line_dash="dot", annotation_text=f"%B {THRESHOLDS['BPSPX_BB_LIFT']:.2f}")
            osc.add_hline(y=0.0, line_dash="dash")
            osc.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(osc, use_container_width=True)

        with col_b:
            tsi_cross_up = False
            tsi_cross_down = False

            if len(g) > 1:
                prev = g.iloc[-2]
                if (
                    pd.notna(last["tsi_fast"])
                    and pd.notna(last["tsi_fast_sig"])
                    and pd.notna(prev["tsi_fast"])
                    and pd.notna(prev["tsi_fast_sig"])
                ):
                    tsi_cross_up = (last["tsi_fast"] > last["tsi_fast_sig"]) and (prev["tsi_fast"] <= prev["tsi_fast_sig"])
                    tsi_cross_down = (last["tsi_fast"] < last["tsi_fast_sig"]) and (prev["tsi_fast"] >= prev["tsi_fast_sig"])

            cci_creep = bool(g["cci20"].tail(3).is_monotonic_increasing) if len(g) >= 3 else False
            rsi_creep = bool(g["rsi14"].tail(3).is_monotonic_increasing) if len(g) >= 3 else False

            diag = {
                "Date": str(last["date"].date()),
                "Close": None if pd.isna(last["close"]) else round(float(last["close"]), 2),
                "Bollinger %B(20,2)": None if pd.isna(last["pct_b20"]) else round(float(last["pct_b20"]), 2),
                "ROC3": None if pd.isna(last["roc3"]) else round(float(last["roc3"]), 2),
                "ROC5": None if pd.isna(last["roc5"]) else round(float(last["roc5"]), 2),
                "RSI14": None if pd.isna(last["rsi14"]) else round(float(last["rsi14"]), 2),
                "CCI20": None if pd.isna(last["cci20"]) else round(float(last["cci20"]), 2),
                "TSI": None if pd.isna(last["tsi_fast"]) else round(float(last["tsi_fast"]), 2),
                "TSI signal": None if pd.isna(last["tsi_fast_sig"]) else round(float(last["tsi_fast_sig"]), 2),
                "Candle": last["candle_pattern"],
                "Bear kiss 50MA": bool_py(last["bear_kiss_50"]),
                "Bull kiss 50MA": bool_py(last["bull_kiss_50"]),
                "TSI cross up": bool(tsi_cross_up),
                "TSI cross down": bool(tsi_cross_down),
                "CCI creeping up": bool(cci_creep),
                "RSI creeping up": bool(rsi_creep),
            }

            st.markdown("**Latest diagnostics**")
            st.json(diag)


# ============================================================
# QA / CHAT
# ============================================================
st.subheader("Narrative / Chat")
q = st.text_input(
    "Ask about the breadth state",
    placeholder="Is the Qwen repair trigger on? Is this leverage-ready? What stage are we in?",
)

if q:
    ql = q.lower()
    answer = []

    if "repair trigger" in ql or "qwen" in ql:
        if repair_trigger_on:
            answer.append(
                f"Qwen repair trigger is active because BPSPX %B is {fmt_num(get_snapshot_val(snapshot, '$BPSPX_%B'))} and SPXA50R is {fmt_num(get_snapshot_val(snapshot, '$SPXA50R'))}."
            )
        else:
            answer.append(
                f"Qwen repair trigger is not active because BPSPX %B is {fmt_num(get_snapshot_val(snapshot, '$BPSPX_%B'))} and SPXA50R is {fmt_num(get_snapshot_val(snapshot, '$SPXA50R'))}."
            )

    if "stage" in ql:
        answer.append(f"Current regime stage is Stage {stage_num}: {stage_name}.")

    if "rsp" in ql:
        answer.append("RSP is cleaner than leverage when the repair trigger is on and participation is broadening.")

    if "ursp" in ql or "spxl" in ql or "leverage" in ql:
        answer.append("Leverage is best reserved for repair trigger on, BPSPX > 45, and NYSI improving enough to reduce structural fragility.")

    if "short" in ql:
        answer.append("Shorts are cleaner on failed bounce setups than on fresh panic lows in this framework.")

    if "thrust" in ql:
        answer.append(
            f"NYAD is {fmt_num(get_snapshot_val(snapshot, '$NYAD'), 0)} and SPXADP is {fmt_num(get_snapshot_val(snapshot, '$SPXADP'), 1)}."
        )

    if not answer:
        answer.append(build_narrative(snapshot, prev_snapshot))

    st.write(" ".join(answer))


# ============================================================
# DOWNLOAD
# ============================================================
st.subheader("Downloads")
parsed_csv = hist_feat.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download parsed historical CSV",
    data=parsed_csv,
    file_name="parsed_breadth_history.csv",
    mime="text/csv",
)

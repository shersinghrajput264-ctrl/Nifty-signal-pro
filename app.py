# ============================================================
# NIFTY + STOCK SIGNAL PRO V3
# Robust / safer / completed-candle / liquidity-aware
# ============================================================

from __future__ import annotations

from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import logging

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="NIFTY + Stock Signal Pro V4",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

IST = ZoneInfo("Asia/Kolkata")
NIFTY = "^NSEI"

DATA_TTL = 45
DAILY_TTL = 300
FUNDAMENTAL_TTL = 3600
MIN_INTRADAY_CANDLES = 100
MIN_DAILY_CANDLES = 220
MAX_WORKERS = 5
INTRADAY_INTERVAL = "5m"

DEFAULT_STOCKS = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS",
    "BHARTIARTL.NS", "INFY.NS", "TCS.NS", "ITC.NS", "LT.NS",
    "AXISBANK.NS", "KOTAKBANK.NS", "HINDUNILVR.NS", "M&M.NS",
    "SUNPHARMA.NS", "MARUTI.NS", "BAJFINANCE.NS", "TITAN.NS",
    "HCLTECH.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS",
    "TATASTEEL.NS", "JSWSTEEL.NS", "ADANIENT.NS", "ADANIPORTS.NS",
    "COALINDIA.NS", "WIPRO.NS", "TECHM.NS", "TATAMOTORS.NS",
    "ASIANPAINT.NS", "ULTRACEMCO.NS", "NESTLEIND.NS", "BAJAJFINSV.NS",
    "HINDALCO.NS", "GRASIM.NS", "CIPLA.NS", "DRREDDY.NS", "DIVISLAB.NS",
    "EICHERMOT.NS", "HEROMOTOCO.NS", "APOLLOHOSP.NS", "BRITANNIA.NS",
    "BPCL.NS", "IOC.NS", "GAIL.NS", "TATACONSUM.NS", "BEL.NS",
    "HAL.NS", "IRFC.NS", "RVNL.NS", "DLF.NS", "TRENT.NS", "VBL.NS",
    "ZOMATO.NS", "JIOFIN.NS",
]

if "stock_results" not in st.session_state:
    st.session_state.stock_results = None
if "last_stock_scan" not in st.session_state:
    st.session_state.last_stock_scan = None
if "nifty_last_refresh" not in st.session_state:
    st.session_state.nifty_last_refresh = None

logging.basicConfig(level=logging.WARNING)

# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=np.nan):
    try:
        if value is None:
            return default
        if isinstance(value, pd.DataFrame):
            if value.empty:
                return default
            value = value.iloc[0, 0]
        elif isinstance(value, pd.Series):
            if value.empty:
                return default
            value = value.iloc[0]
        value = float(value)
        return value if np.isfinite(value) else default
    except (TypeError, ValueError, IndexError):
        return default


def finite(value):
    x = safe_float(value)
    return bool(np.isfinite(x))


def fmt(value, digits=2):
    x = safe_float(value)
    return "-" if not np.isfinite(x) else f"{x:,.{digits}f}"


def pct(value, digits=1):
    x = safe_float(value)
    return "-" if not np.isfinite(x) else f"{x:.{digits}f}%"


def clamp(value, low, high):
    x = safe_float(value)
    if not np.isfinite(x):
        return low
    return max(low, min(high, x))


def market_status():
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return "🔴 NSE closed — Weekend"
    if dt_time(9, 15) <= now.time() <= dt_time(15, 30):
        return "🟢 NSE market hours"
    return "🟡 NSE market closed"

# ============================================================
# DATA CLEANING / DOWNLOAD
# ============================================================

def clean_ohlcv(data):
    if data is None or data.empty:
        return pd.DataFrame()

    df = data.copy()

    if isinstance(df.columns, pd.MultiIndex):
        cols = []
        wanted = {"Open", "High", "Low", "Close", "Volume"}
        for col in df.columns:
            match = next((str(x) for x in col if str(x) in wanted), None)
            cols.append(match if match else str(col[-1]))
        df.columns = cols

    required = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in required):
        return pd.DataFrame()

    df = df[required].copy()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df["Volume"] = df["Volume"].fillna(0).clip(lower=0)

    if isinstance(df.index, pd.DatetimeIndex):
        try:
            if df.index.tz is not None:
                df.index = df.index.tz_convert(IST)
            else:
                df.index = df.index.tz_localize(IST)
        except Exception:
            pass
        df = df[~df.index.duplicated(keep="last")].sort_index()

    return df


def extract_symbol(raw, symbol):
    if raw is None or raw.empty:
        return pd.DataFrame()
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            lv0 = list(raw.columns.get_level_values(0))
            lv1 = list(raw.columns.get_level_values(1))
            if symbol in lv0:
                return clean_ohlcv(raw[symbol])
            if symbol in lv1:
                return clean_ohlcv(raw.xs(symbol, axis=1, level=1))
        return clean_ohlcv(raw)
    except (KeyError, ValueError, IndexError):
        return pd.DataFrame()


def _download_batch(symbols, period, interval):
    symbols = tuple(dict.fromkeys(symbols))
    if not symbols:
        return {}
    try:
        raw = yf.download(
            tickers=list(symbols),
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=True,
            group_by="ticker",
        )
    except Exception as exc:
        logging.warning("Yahoo download failed: %s", exc)
        return {}

    result = {}
    for symbol in symbols:
        df = extract_symbol(raw, symbol)
        minimum = MIN_INTRADAY_CANDLES if interval != "1d" else MIN_DAILY_CANDLES
        if len(df) >= minimum:
            result[symbol] = df
    return result


@st.cache_data(ttl=DATA_TTL, show_spinner=False)
def download_intraday_batch(symbols):
    return _download_batch(tuple(symbols), "5d", INTRADAY_INTERVAL)


@st.cache_data(ttl=DAILY_TTL, show_spinner=False)
def download_daily_batch(symbols):
    return _download_batch(tuple(symbols), "2y", "1d")


@st.cache_data(ttl=DATA_TTL, show_spinner=False)
def get_nifty_intraday():
    return download_intraday_batch((NIFTY,)).get(NIFTY, pd.DataFrame())


@st.cache_data(ttl=DAILY_TTL, show_spinner=False)
def get_nifty_daily():
    return download_daily_batch((NIFTY,)).get(NIFTY, pd.DataFrame())

# ============================================================
# COMPLETED CANDLE LOGIC
# ============================================================

def is_market_open_now():
    now = datetime.now(IST)
    return now.weekday() < 5 and dt_time(9, 15) <= now.time() <= dt_time(15, 30)


def get_completed_intraday_df(df):
    """Remove only a candle that is actually still forming.

    Yahoo can return the latest completed candle after market close.
    Therefore we do NOT blindly drop the last row.
    """
    if df is None or len(df) < 3:
        return pd.DataFrame()

    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        return out.iloc[:-1].copy()

    now = datetime.now(IST)
    last_ts = out.index[-1]
    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=IST)
    else:
        last_ts = last_ts.astimezone(IST)

    # 5-minute candle ending at minute 15/20/... is complete when its
    # timestamp is older than the current 5-minute bucket.
    bucket_minute = (now.minute // 5) * 5
    current_bucket = now.replace(minute=bucket_minute, second=0, microsecond=0)
    if is_market_open_now() and last_ts >= current_bucket:
        return out.iloc[:-1].copy()
    return out

# ============================================================
# INDICATORS
# ============================================================

def rsi_wilder(close, period=14):
    change = close.diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi[(avg_loss == 0) & (avg_gain > 0)] = 100
    rsi[(avg_gain == 0) & (avg_loss > 0)] = 0
    # Flat periods are neutral rather than missing.
    rsi[(avg_gain == 0) & (avg_loss == 0)] = 50
    return rsi


def calculate_intraday_indicators(df):
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

    for span in (9, 20, 50, 200):
        df[f"EMA{span}"] = close.ewm(span=span, adjust=False, min_periods=span).mean()
    df["RSI"] = rsi_wilder(close, 14)

    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    typical = (high + low + close) / 3
    vol = volume.fillna(0)
    if isinstance(df.index, pd.DatetimeIndex):
        session = pd.Series(df.index.date, index=df.index)
        pv = typical * vol
        df["VWAP"] = pv.groupby(session).cumsum() / vol.groupby(session).cumsum().replace(0, np.nan)
    else:
        df["VWAP"] = (typical * vol).cumsum() / vol.cumsum().replace(0, np.nan)

    df["VOL_AVG20"] = volume.shift(1).rolling(20, min_periods=10).mean()
    df["VOL_RATIO"] = np.where(df["VOL_AVG20"] > 0, volume / df["VOL_AVG20"], np.nan)
    df["HIGH10"] = high.shift(1).rolling(10, min_periods=10).max()
    df["LOW10"] = low.shift(1).rolling(10, min_periods=10).min()
    df["HIGH20"] = high.shift(1).rolling(20, min_periods=20).max()
    df["LOW20"] = low.shift(1).rolling(20, min_periods=20).min()
    df["ROC5"] = close.pct_change(5) * 100
    df["ROC10"] = close.pct_change(10) * 100

    candle_range = (high - low).replace(0, np.nan)
    df["BODY_PCT"] = (close - df["Open"]).abs() / candle_range
    df["CLOSE_LOCATION"] = (close - low) / candle_range
    return df


def calculate_daily_indicators(df):
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    close = df["Close"]
    df["EMA50_DAILY"] = close.ewm(span=50, adjust=False, min_periods=50).mean()
    df["EMA200_DAILY"] = close.ewm(span=200, adjust=False, min_periods=200).mean()
    return df

# ============================================================
# SCORING
# ============================================================

BULL_WEIGHTS = {
    "Price above VWAP": 1.30,
    "EMA9 > EMA20": 0.90,
    "EMA20 > EMA50": 1.20,
    "EMA20 rising": 0.60,
    "EMA50 rising": 0.50,
    "Above daily EMA200": 1.50,
    "RSI 52-72": 1.00,
    "ROC5 positive": 0.70,
    "ROC10 positive": 0.70,
    "Volume >= 1.2x": 1.20,
    "10-candle breakout": 1.50,
    "Strong bullish candle": 0.80,
}

BEAR_WEIGHTS = {
    "Price below VWAP": 1.30,
    "EMA9 < EMA20": 0.90,
    "EMA20 < EMA50": 1.20,
    "EMA20 falling": 0.60,
    "EMA50 falling": 0.50,
    "Below daily EMA200": 1.50,
    "RSI 28-48": 1.00,
    "ROC5 negative": 0.70,
    "ROC10 negative": 0.70,
    "Volume >= 1.2x": 1.20,
    "10-candle breakdown": 1.50,
    "Strong bearish candle": 0.80,
}
MAX_SCORE = sum(BULL_WEIGHTS.values())


def classify_signal(bull_pct, bear_pct):
    diff = abs(bull_pct - bear_pct)
    if bull_pct >= 72 and diff >= 18:
        return "STRONG BUY"
    if bull_pct >= 62 and diff >= 12:
        return "BUY WATCH"
    if bear_pct >= 72 and diff >= 18:
        return "STRONG SELL"
    if bear_pct >= 62 and diff >= 12:
        return "SELL WATCH"
    if bull_pct >= 52 and bull_pct > bear_pct:
        return "WATCH BUY"
    if bear_pct >= 52 and bear_pct > bull_pct:
        return "WATCH SELL"
    return "NO EDGE"


def analyze_frame(intraday, daily, symbol=None):
    if intraday is None or daily is None or intraday.empty or daily.empty:
        return None

    intraday = get_completed_intraday_df(calculate_intraday_indicators(intraday))
    daily = calculate_daily_indicators(daily)
    if len(intraday) < MIN_INTRADAY_CANDLES or len(daily) < MIN_DAILY_CANDLES:
        return None

    last, prev = intraday.iloc[-1], intraday.iloc[-2]
    dlast = daily.iloc[-1]

    price = safe_float(last["Close"])
    if not finite(price):
        return None

    vals = {k: safe_float(last[k]) for k in [
        "VWAP", "EMA9", "EMA20", "EMA50", "EMA200", "RSI", "ATR",
        "ROC5", "ROC10", "VOL_RATIO", "HIGH10", "LOW10", "BODY_PCT", "CLOSE_LOCATION"
    ]}
    daily_ema200 = safe_float(dlast["EMA200_DAILY"])
    prev_ema20, prev_ema50 = safe_float(prev["EMA20"]), safe_float(prev["EMA50"])

    bullish = {
        "Price above VWAP": finite(vals["VWAP"]) and price > vals["VWAP"],
        "EMA9 > EMA20": finite(vals["EMA9"]) and finite(vals["EMA20"]) and vals["EMA9"] > vals["EMA20"],
        "EMA20 > EMA50": finite(vals["EMA20"]) and finite(vals["EMA50"]) and vals["EMA20"] > vals["EMA50"],
        "EMA20 rising": finite(vals["EMA20"]) and finite(prev_ema20) and vals["EMA20"] > prev_ema20,
        "EMA50 rising": finite(vals["EMA50"]) and finite(prev_ema50) and vals["EMA50"] > prev_ema50,
        "Above daily EMA200": finite(daily_ema200) and price > daily_ema200,
        "RSI 52-72": finite(vals["RSI"]) and 52 <= vals["RSI"] <= 72,
        "ROC5 positive": finite(vals["ROC5"]) and vals["ROC5"] > 0,
        "ROC10 positive": finite(vals["ROC10"]) and vals["ROC10"] > 0,
        "Volume >= 1.2x": finite(vals["VOL_RATIO"]) and vals["VOL_RATIO"] >= 1.20,
        "10-candle breakout": finite(vals["HIGH10"]) and price > vals["HIGH10"],
        "Strong bullish candle": (
            price > safe_float(last["Open"]) and finite(vals["BODY_PCT"]) and
            finite(vals["CLOSE_LOCATION"]) and vals["BODY_PCT"] >= 0.45 and vals["CLOSE_LOCATION"] >= 0.65
        ),
    }
    bearish = {
        "Price below VWAP": finite(vals["VWAP"]) and price < vals["VWAP"],
        "EMA9 < EMA20": finite(vals["EMA9"]) and finite(vals["EMA20"]) and vals["EMA9"] < vals["EMA20"],
        "EMA20 < EMA50": finite(vals["EMA20"]) and finite(vals["EMA50"]) and vals["EMA20"] < vals["EMA50"],
        "EMA20 falling": finite(vals["EMA20"]) and finite(prev_ema20) and vals["EMA20"] < prev_ema20,
        "EMA50 falling": finite(vals["EMA50"]) and finite(prev_ema50) and vals["EMA50"] < prev_ema50,
        "Below daily EMA200": finite(daily_ema200) and price < daily_ema200,
        "RSI 28-48": finite(vals["RSI"]) and 28 <= vals["RSI"] <= 48,
        "ROC5 negative": finite(vals["ROC5"]) and vals["ROC5"] < 0,
        "ROC10 negative": finite(vals["ROC10"]) and vals["ROC10"] < 0,
        "Volume >= 1.2x": finite(vals["VOL_RATIO"]) and vals["VOL_RATIO"] >= 1.20,
        "10-candle breakdown": finite(vals["LOW10"]) and price < vals["LOW10"],
        "Strong bearish candle": (
            price < safe_float(last["Open"]) and finite(vals["BODY_PCT"]) and
            finite(vals["CLOSE_LOCATION"]) and vals["BODY_PCT"] >= 0.45 and vals["CLOSE_LOCATION"] <= 0.35
        ),
    }

    bull_score = sum(BULL_WEIGHTS[k] for k, v in bullish.items() if v)
    bear_score = sum(BEAR_WEIGHTS[k] for k, v in bearish.items() if v)
    bull_pct = bull_score / MAX_SCORE * 100
    bear_pct = bear_score / MAX_SCORE * 100

    # Contradiction penalty: don't call a setup strong when the opposite side
    # has substantial confirmation too.
    raw_diff = bull_pct - bear_pct
    signal = classify_signal(bull_pct, bear_pct)
    if abs(raw_diff) < 12 and signal in {"STRONG BUY", "STRONG SELL"}:
        signal = "NO EDGE"

    return {
        "symbol": symbol.replace(".NS", "") if symbol else "NIFTY",
        "ticker": symbol or NIFTY,
        "price": price,
        "bull_score": bull_pct,
        "bear_score": bear_pct,
        "score": raw_diff,
        "signal": signal,
        "rsi": vals["RSI"],
        "vwap": vals["VWAP"],
        "ema9": vals["EMA9"],
        "ema20": vals["EMA20"],
        "ema50": vals["EMA50"],
        "ema200_intraday": vals["EMA200"],
        "daily_ema200": daily_ema200,
        "atr": vals["ATR"],
        "roc5": vals["ROC5"],
        "roc10": vals["ROC10"],
        "volume_ratio": vals["VOL_RATIO"],
        "bullish": bullish,
        "bearish": bearish,
        "reasons": [k for k, v in bullish.items() if v],
        "warnings": [k for k, v in bearish.items() if v],
        "candle_time": intraday.index[-1],
        "chart_df": intraday,
    }

# ============================================================
# NIFTY / OPTION CHAIN
# ============================================================

def get_atm(price):
    p = safe_float(price)
    return np.nan if not finite(p) else int(round(p / 50) * 50)


def clean_options(df):
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    numeric = ["strike", "lastPrice", "bid", "ask", "change", "percentChange", "volume", "openInterest", "impliedVolatility"]
    for col in numeric:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ["volume", "openInterest"]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = out[col].fillna(0).clip(lower=0)
    return out


@st.cache_data(ttl=DATA_TTL, show_spinner=False)
def get_nifty_options():
    try:
        ticker = yf.Ticker(NIFTY)
        expirations = sorted(set(ticker.options or []))
        today = datetime.now(IST).date()
        valid = [x for x in expirations if pd.Timestamp(x).date() >= today]
        if not valid:
            return None, pd.DataFrame(), pd.DataFrame(), []
        expiry = valid[0]
        chain = ticker.option_chain(expiry)
        return expiry, clean_options(chain.calls), clean_options(chain.puts), valid
    except Exception as exc:
        logging.warning("Option chain failed: %s", exc)
        return None, pd.DataFrame(), pd.DataFrame(), []


def select_option(df, atm, direction):
    if df is None or df.empty or not finite(atm):
        return None
    work = df[df["strike"].notna()].copy()
    if work.empty:
        return None
    work["distance"] = (work["strike"] - atm).abs()
    work = work[work["distance"] <= 300].copy()
    if direction == "CE":
        work = work[(work["strike"] >= atm - 50) & (work["strike"] <= atm + 100)].copy()
    elif direction == "PE":
        work = work[(work["strike"] >= atm - 100) & (work["strike"] <= atm + 50)].copy()
    else:
        return None
    if work.empty:
        return None

    valid = (work["bid"] > 0) & (work["ask"] > 0) & (work["ask"] >= work["bid"])
    work["mid"] = np.where(valid, (work["bid"] + work["ask"]) / 2, work["lastPrice"])
    work["spread_pct"] = np.where((work["mid"] > 0) & valid, (work["ask"] - work["bid"]) / work["mid"] * 100, 999)
    work = work[(work["mid"] > 0) & (work["spread_pct"] <= 12)].copy()
    if work.empty:
        return None

    work["liquidity"] = np.log1p(work["volume"]) * 2 + np.log1p(work["openInterest"])
    work["score"] = work["liquidity"] - work["distance"] / 50 - work["spread_pct"] * 2
    return work.sort_values("score", ascending=False).iloc[0]


def calculate_pcr(calls, puts):
    if calls.empty or puts.empty:
        return np.nan, np.nan
    call_oi = safe_float(calls["openInterest"].sum(), 0)
    put_oi = safe_float(puts["openInterest"].sum(), 0)
    call_vol = safe_float(calls["volume"].sum(), 0)
    put_vol = safe_float(puts["volume"].sum(), 0)
    return (
        put_oi / call_oi if call_oi > 0 else np.nan,
        put_vol / call_vol if call_vol > 0 else np.nan,
    )


def calculate_max_pain(calls, puts):
    if calls.empty or puts.empty:
        return np.nan
    c = calls[calls["strike"].notna()].copy()
    p = puts[puts["strike"].notna()].copy()
    if c.empty or p.empty:
        return np.nan
    strikes = sorted(set(c["strike"]).union(set(p["strike"])))
    best, best_loss = np.nan, np.inf
    for settlement in strikes:
        call_loss = (np.maximum(settlement - c["strike"], 0) * c["openInterest"]).sum()
        put_loss = (np.maximum(p["strike"] - settlement, 0) * p["openInterest"]).sum()
        loss = call_loss + put_loss
        if loss < best_loss:
            best_loss, best = loss, settlement
    return best


def get_oi_levels(calls, puts, price):
    p = safe_float(price)
    if calls.empty or puts.empty or not finite(p):
        return np.nan, np.nan
    c = calls[(calls["strike"] >= p) & (calls["strike"] <= p + 500)].copy()
    q = puts[(puts["strike"] <= p) & (puts["strike"] >= p - 500)].copy()
    resistance = np.nan
    support = np.nan
    if not c.empty:
        c["level_score"] = c["openInterest"] / (1 + (c["strike"] - p) / 50)
        resistance = safe_float(c.loc[c["level_score"].idxmax(), "strike"])
    if not q.empty:
        q["level_score"] = q["openInterest"] / (1 + (p - q["strike"]) / 50)
        support = safe_float(q.loc[q["level_score"].idxmax(), "strike"])
    return resistance, support


def build_trade_plan(signal, option_row, nifty_price, atr, allow_watch=False):
    """Build an actionable NIFTY/option plan without inventing premium data.

    If a live option contract is available, premium entry/SL/targets are shown.
    If Yahoo option-chain data is unavailable, the app still shows an underlying
    NIFTY trigger and the nearest ATM CE/PE candidate; premium levels are marked
    unavailable instead of fabricated.
    """
    p, a = safe_float(nifty_price), safe_float(atr)
    if not finite(p) or not finite(a) or a <= 0:
        return None
    if signal not in {"BUY CE", "BUY PE", "WATCH CE", "WATCH PE"}:
        return None

    is_ce = signal in {"BUY CE", "WATCH CE"}
    direction = 1 if is_ce else -1
    # Previous completed-candle breakout proxy. ATR buffer avoids reacting to a
    # one-tick touch.
    trigger = p + direction * max(a * 0.15, p * 0.00035)
    nifty_stop = p - direction * a * 0.80

    plan = {
        "entry": np.nan,
        "stop_loss": np.nan,
        "target1": np.nan,
        "target2": np.nan,
        "target3": np.nan,
        "nifty_stop": nifty_stop,
        "nifty_trigger": trigger,
        "premium_available": False,
        "strike": safe_float(option_row.get("strike")) if option_row is not None else np.nan,
        "option_ltp": safe_float(option_row.get("lastPrice")) if option_row is not None else np.nan,
    }

    if option_row is None:
        return plan if allow_watch else None

    bid = safe_float(option_row.get("bid"))
    ask = safe_float(option_row.get("ask"))
    last = safe_float(option_row.get("lastPrice"))
    if finite(bid) and finite(ask) and bid > 0 and ask >= bid:
        entry = (bid + ask) / 2
    elif finite(last) and last > 0:
        entry = last
    else:
        return plan if allow_watch else None

    # For a LONG CE or LONG PE, the option premium rises when the
    # underlying moves in the expected direction. Therefore premium targets
    # are always ABOVE the premium entry; `direction` only controls the NIFTY
    # trigger/SL.
    stop = entry * 0.82
    risk = entry - stop
    if risk <= 0:
        return plan if allow_watch else None
    plan.update({
        "entry": entry,
        "stop_loss": stop,
        "target1": entry + 1.5 * risk,
        "target2": entry + 2.5 * risk,
        "target3": entry + 4.0 * risk,
        "premium_available": True,
        "risk_pct": (risk / entry) * 100,
    })
    return plan

# ============================================================
# STOCK FUNDAMENTALS
# ============================================================

@st.cache_data(ttl=FUNDAMENTAL_TTL, show_spinner=False)
def get_fundamentals(symbol):
    empty = {"market_cap": np.nan, "roe": np.nan, "debt_equity": np.nan, "profit_margin": np.nan, "revenue_growth": np.nan, "earnings_growth": np.nan}
    try:
        info = yf.Ticker(symbol).info
        vals = {
            "market_cap": safe_float(info.get("marketCap")),
            "roe": safe_float(info.get("returnOnEquity")),
            "debt_equity": safe_float(info.get("debtToEquity")),
            "profit_margin": safe_float(info.get("profitMargins")),
            "revenue_growth": safe_float(info.get("revenueGrowth")),
            "earnings_growth": safe_float(info.get("earningsGrowth")),
        }
        for k in ["roe", "profit_margin", "revenue_growth", "earnings_growth"]:
            if finite(vals[k]):
                vals[k] *= 100
        return vals
    except Exception:
        return empty


def calculate_day_change(daily_df, current_price):
    p = safe_float(current_price)
    if daily_df is None or daily_df.empty or not finite(p):
        return np.nan
    df = daily_df.copy()
    today = datetime.now(IST).date()
    previous_close = np.nan
    if isinstance(df.index, pd.DatetimeIndex):
        dates = [x.date() for x in df.index]
        if today in dates:
            i = dates.index(today)
            if i > 0:
                previous_close = safe_float(df.iloc[i - 1]["Close"])
    if not finite(previous_close):
        previous_close = safe_float(df.iloc[-1]["Close"])
    return ((p / previous_close) - 1) * 100 if finite(previous_close) and previous_close > 0 else np.nan


def build_stock_plan(row):
    """Actionable intraday stock plan from completed-candle data.

    Long plans are used for bullish signals; bearish signals get a SHORT plan
    so the dashboard can still explain entry/SL/targets instead of hiding the setup.
    Levels are decision-support estimates, not guaranteed prices/profits.
    """
    p = safe_float(row.get("price"))
    a = safe_float(row.get("atr"))
    h10 = safe_float(row.get("HIGH10"))
    l10 = safe_float(row.get("LOW10"))
    if not finite(p) or not finite(a) or a <= 0:
        return {}

    bullish = row.get("signal") in {"STRONG BUY", "BUY WATCH", "WATCH BUY"}
    bearish = row.get("signal") in {"STRONG SELL", "SELL WATCH", "WATCH SELL"}
    if not bullish and not bearish:
        return {"status": "WAIT", "direction": "WAIT"}

    direction = 1 if bullish else -1
    buffer = max(a * 0.15, p * 0.0005)

    # Prefer a real previous-range breakout/breakdown when available.
    if bullish and finite(h10):
        trigger = h10 + max(p * 0.0005, a * 0.05)
    elif bearish and finite(l10):
        trigger = l10 - max(p * 0.0005, a * 0.05)
    else:
        trigger = p + direction * buffer

    risk = max(a, p * 0.004)
    stop = trigger - direction * risk
    t1 = trigger + direction * 1.5 * risk
    t2 = trigger + direction * 2.5 * risk
    t3 = trigger + direction * 4.0 * risk

    # A breakout entry is considered active only after the completed candle
    # has actually crossed the trigger; otherwise WAIT is explicit.
    crossed = p >= trigger if direction == 1 else p <= trigger
    status = "ENTRY NOW" if crossed else "WAIT FOR TRIGGER"

    return {
        "entry": trigger, "sl": stop, "t1": t1, "t2": t2, "t3": t3,
        "t1_pct": (1.5 * risk / abs(trigger)) * 100,
        "t2_pct": (2.5 * risk / abs(trigger)) * 100,
        "t3_pct": (4.0 * risk / abs(trigger)) * 100,
        "status": status,
        "direction": "LONG" if direction == 1 else "SHORT",
        "exit_rule": "SL hit = EXIT; T1 = partial booking + trail SL; T2/T3 = further booking; 15:20 IST EOD exit.",
    }

def analyze_stock(symbol, intraday, daily):
    result = analyze_frame(intraday, daily, symbol)
    if result is None:
        return None
    result["day_change"] = calculate_day_change(daily, result["price"])
    plan = build_stock_plan(result)
    result.update({
        "entry": plan.get("entry", np.nan), "sl": plan.get("sl", np.nan),
        "t1": plan.get("t1", np.nan), "t2": plan.get("t2", np.nan), "t3": plan.get("t3", np.nan),
        "t1_pct": plan.get("t1_pct", np.nan), "t2_pct": plan.get("t2_pct", np.nan), "t3_pct": plan.get("t3_pct", np.nan),
        "entry_status": plan.get("status", "WAIT"),
    })
    return result

# ============================================================
# UI
# ============================================================

st.title("📈 NIFTY + Stock Signal Pro V4")
st.caption("Actionable Entry • SL • T1/T2/T3 • CE/PE strike • NIFTY trigger • Exit rules • Stock trade plans")

tab_nifty, tab_stocks = st.tabs(["📊 NIFTY + OPTIONS", "🚀 STOCK SCANNER"])

# ============================================================
# NIFTY RENDER
# ============================================================

def render_nifty():
    with st.spinner("NIFTY data load ho raha hai..."):
        nifty_intraday = get_nifty_intraday()
        nifty_daily = get_nifty_daily()

    if nifty_intraday.empty or nifty_daily.empty:
        st.error("❌ NIFTY data available nahi hai.")
        st.info("Yahoo Finance temporary rate-limit/data issue de sakta hai. Kuch seconds baad refresh karein.")
        return

    market = analyze_frame(nifty_intraday, nifty_daily, NIFTY)
    if market is None:
        st.warning("⚠️ Sufficient completed candles / technical data nahi mila.")
        return

    price = market["price"]
    atm = get_atm(price)
    expiry, calls, puts, expirations = get_nifty_options()
    ce_row = select_option(calls, atm, "CE")
    pe_row = select_option(puts, atm, "PE")
    oi_pcr, volume_pcr = calculate_pcr(calls, puts)
    max_pain = calculate_max_pain(calls, puts)
    resistance, support = get_oi_levels(calls, puts, price)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("NIFTY", fmt(price))
    m2.metric("ATM", str(atm))
    m3.metric("Bull Score", f"{market['bull_score']:.0f}%")
    m4.metric("Bear Score", f"{market['bear_score']:.0f}%")
    m5.metric("Signal", market["signal"])
    st.caption(f"{market_status()} • Signal uses last completed 5-minute candle: {market['candle_time']}")

    st.subheader("🚨 FINAL SIGNAL")
    # Strong trend = immediate candidate; moderate bullish trend = CE WATCH.
    # This keeps the app useful even when the option chain is temporarily down.
    if market["signal"] == "STRONG BUY":
        action = "BUY CE"
    elif market["signal"] in {"BUY WATCH", "WATCH BUY"}:
        action = "WATCH CE"
    elif market["signal"] == "STRONG SELL":
        action = "BUY PE"
    elif market["signal"] in {"SELL WATCH", "WATCH SELL"}:
        action = "WATCH PE"
    else:
        action = "WAIT"

    if action == "BUY CE":
        st.success("🟢 BUY CE CANDIDATE — bullish confirmation strong hai")
    elif action == "WATCH CE":
        st.info("🟡 CE WATCH — breakout confirmation ka wait karo")
    elif action == "BUY PE":
        st.error("🔴 BUY PE CANDIDATE — bearish confirmation strong hai")
    elif action == "WATCH PE":
        st.warning("🟠 PE WATCH — breakdown confirmation ka wait karo")
    else:
        st.warning("⚪ WAIT — clear directional edge nahi hai")

    st.subheader("📐 Technical Dashboard")
    a = st.columns(7)
    a[0].metric("EMA20", fmt(market["ema20"]))
    a[1].metric("EMA50", fmt(market["ema50"]))
    a[2].metric("Intraday EMA200", fmt(market["ema200_intraday"]))
    a[3].metric("Daily EMA200", fmt(market["daily_ema200"]))
    a[4].metric("VWAP", fmt(market["vwap"]))
    a[5].metric("RSI", fmt(market["rsi"]))
    a[6].metric("ATR", fmt(market["atr"]))

    st.subheader("⛓️ Option Chain Intelligence")
    if calls.empty and puts.empty:
        st.warning("⚠️ Yahoo Finance option chain unavailable — premium price invent nahi ki jayegi.")
    else:
        o = st.columns(6)
        o[0].metric("Expiry", str(expiry) if expiry else "-")
        o[1].metric("OI PCR", fmt(oi_pcr))
        o[2].metric("Volume PCR", fmt(volume_pcr))
        o[3].metric("Max Pain", fmt(max_pain, 0))
        o[4].metric("OI Resistance", fmt(resistance, 0))
        o[5].metric("OI Support", fmt(support, 0))

        rows = []
        for side, row in (("CE", ce_row), ("PE", pe_row)):
            if row is not None:
                rows.append({
                    "Side": side,
                    "Strike": safe_float(row.get("strike")),
                    "Last": safe_float(row.get("lastPrice")),
                    "Bid": safe_float(row.get("bid")),
                    "Ask": safe_float(row.get("ask")),
                    "Volume": safe_float(row.get("volume"), 0),
                    "OI": safe_float(row.get("openInterest"), 0),
                    "IV %": safe_float(row.get("impliedVolatility")) * 100,
                })
        if rows:
            st.dataframe(pd.DataFrame(rows).round(2), use_container_width=True, hide_index=True)
        else:
            st.info("ATM ke paas liquid CE/PE contract nahi mila.")

    st.subheader("🎯 CE / PE TRADE PLAN — ENTRY / SL / TARGET / EXIT")

    # Always show BOTH sides. The signal decides which side is preferred;
    # the other side is explicitly marked WAIT so the user never has to guess.
    ce_plan = build_trade_plan("BUY CE", ce_row, price, market["atr"], allow_watch=True)
    pe_plan = build_trade_plan("BUY PE", pe_row, price, market["atr"], allow_watch=True)

    def render_option_plan(side, plan, preferred):
        is_ce = side == "CE"
        expected = "UP" if is_ce else "DOWN"
        if plan is None:
            st.warning(f"{side}: plan unavailable — NIFTY data/ATR insufficient.")
            return

        if preferred:
            st.success(f"🟢 {side} — PREFERRED SIDE ({expected})")
        else:
            st.info(f"⚪ {side} — WAIT / NOT PREFERRED")

        strike_text = fmt(plan["strike"], 0) if finite(plan["strike"]) else str(atm)
        c = st.columns(5)
        c[0].metric(f"{side} Strike", strike_text)
        c[1].metric("Option LTP", fmt(plan["option_ltp"]) if finite(plan["option_ltp"]) else "Unavailable")
        c[2].metric("NIFTY Trigger", fmt(plan["nifty_trigger"]))
        c[3].metric("NIFTY SL", fmt(plan["nifty_stop"]))
        c[4].metric("Status", "BUY" if preferred else "WAIT")

        if plan["premium_available"]:
            q = st.columns(5)
            q[0].metric("Premium ENTRY", fmt(plan["entry"]))
            q[1].metric("Premium SL", fmt(plan["stop_loss"]))
            q[2].metric("T1", fmt(plan["target1"]))
            q[3].metric("T2", fmt(plan["target2"]))
            q[4].metric("T3", fmt(plan["target3"]))
            st.caption(
                f"Approx premium move: T1 +{((plan['target1']/plan['entry'])-1)*100:.1f}% • "
                f"T2 +{((plan['target2']/plan['entry'])-1)*100:.1f}% • "
                f"T3 +{((plan['target3']/plan['entry'])-1)*100:.1f}%"
            )
        else:
            st.warning(
                f"Live {side} premium unavailable. ₹{strike_text} {side} candidate hai. "
                f"Entry NIFTY {expected.lower()} trigger ₹{fmt(plan['nifty_trigger'])} par confirmation ke baad; "
                "broker ka live premium dekhkar order place karein."
            )

        st.write("**Entry:** " + (
            f"NIFTY {expected.lower()} ₹{fmt(plan['nifty_trigger'])} ke paar 5-min candle close/sustain kare."
        ))
        st.write(f"**SL:** NIFTY ₹{fmt(plan['nifty_stop'])} ke opposite side 5-min close ho to {side} EXIT.")
        if plan["premium_available"]:
            st.write(
                f"**EXIT:** Premium ₹{fmt(plan['target1'])} → 50% book; ₹{fmt(plan['target2'])} → next booking; "
                f"₹{fmt(plan['target3'])} → remaining exit/trailing. Premium SL ₹{fmt(plan['stop_loss'])}."
            )
        else:
            st.write("**EXIT:** NIFTY SL hit → exit. T1/T2/T3 premium levels live option price aane par calculate honge.")

    preferred_ce = action in {"BUY CE", "WATCH CE"}
    preferred_pe = action in {"BUY PE", "WATCH PE"}
    if action == "WAIT":
        st.warning("⚪ Abhi CE aur PE dono WAIT — clear directional edge nahi hai.")

    render_option_plan("CE", ce_plan, preferred_ce)
    st.divider()
    render_option_plan("PE", pe_plan, preferred_pe)

    st.caption(
        "⏰ Fresh entry generally 14:45 ke baad avoid. Open intraday option position ko "
        "15:20 tak square-off karna safer rule hai. Levels decision-support hain, guaranteed profit nahi."
    )

    st.subheader("✅ Confirmation Checklist")
    left, right = st.columns(2)
    with left:
        st.markdown("### 🟢 Bullish")
        for name, value in market["bullish"].items():
            st.write(("✅ " if value else "❌ ") + name)
    with right:
        st.markdown("### 🔴 Bearish")
        for name, value in market["bearish"].items():
            st.write(("✅ " if value else "❌ ") + name)

    st.subheader("📈 NIFTY 5-Minute Chart")
    chart_df = market["chart_df"][["Close", "EMA20", "EMA50", "VWAP"]].tail(150)
    st.line_chart(chart_df, height=450)

# ============================================================
# NIFTY TAB
# ============================================================

with tab_nifty:
    st.header("📊 NIFTY Signal + Option Intelligence")
    st.caption(market_status())
    c1, c2 = st.columns(2)
    with c1:
        refresh_nifty = st.button("🔄 Refresh NIFTY", use_container_width=True)
    with c2:
        auto_refresh = st.checkbox("⏱️ Auto Refresh (45s)", value=False)

    if refresh_nifty:
        download_intraday_batch.clear()
        download_daily_batch.clear()
        get_nifty_intraday.clear()
        get_nifty_daily.clear()
        get_nifty_options.clear()
        st.session_state.nifty_last_refresh = datetime.now(IST)
        st.rerun()

    if auto_refresh and hasattr(st, "fragment"):
        @st.fragment(run_every="45s")
        def _live_nifty():
            render_nifty()
        _live_nifty()
    elif auto_refresh:
        st.info("Current Streamlit version native fragment auto-refresh support nahi karta. Manual refresh use karein.")
        render_nifty()
    else:
        render_nifty()

# ============================================================
# STOCK SCANNER
# ============================================================

with tab_stocks:
    st.header("🚀 NSE Stock Scanner")
    st.caption("Completed candle + Daily EMA200 + VWAP + EMA trend + RSI + momentum + volume + breakout")

    s1, s2, s3 = st.columns(3)
    with s1:
        scan_count = st.selectbox("Stocks to scan", [25, 50], index=1)
    with s2:
        min_buy_score = st.slider("Minimum bullish score %", 40, 90, 62)
    with s3:
        run_scan = st.button("🚀 SCAN STOCKS", use_container_width=True)

    symbols = DEFAULT_STOCKS[:min(scan_count, len(DEFAULT_STOCKS))]

    if run_scan:
        with st.spinner(f"{len(symbols)} stocks ka market data download ho raha hai..."):
            intraday_data = download_intraday_batch(tuple(symbols))
            daily_data = download_daily_batch(tuple(symbols))

        results = []
        progress = st.progress(0.0)
        status = st.empty()

        def worker(sym):
            return analyze_stock(sym, intraday_data.get(sym, pd.DataFrame()), daily_data.get(sym, pd.DataFrame()))

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(worker, sym): sym for sym in symbols}
            total = len(futures)
            for done, future in enumerate(as_completed(futures), start=1):
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as exc:
                    logging.warning("Stock analysis failed for %s: %s", futures[future], exc)
                progress.progress(done / total)
                status.write(f"Analysing {done}/{total}...")

        progress.empty(); status.empty()
        if not results:
            st.error("❌ Valid stock data nahi mila. Yahoo Finance data/rate-limit check karein.")
            st.session_state.stock_results = None
        else:
            df = pd.DataFrame(results)
            df = df.sort_values(["score", "bull_score", "volume_ratio"], ascending=False, na_position="last")
            st.session_state.stock_results = df
            st.session_state.last_stock_scan = datetime.now(IST)

    result_df = st.session_state.stock_results
    if result_df is None:
        st.info("🚀 Scan Stocks button dabakar scanner start karein.")
    else:
        if st.session_state.last_stock_scan:
            st.caption("Last scan: " + st.session_state.last_stock_scan.strftime("%d-%m-%Y %H:%M:%S IST"))

        st.subheader("🏆 Top Current Setups")
        # Never hide the best stocks just because the strict filter is empty.
        # Mark the entry status so the user can see whether to WAIT or ENTER.
        filtered = result_df[(result_df["bull_score"] >= min_buy_score) & (result_df["score"] >= 12)]
        top = (filtered if not filtered.empty else result_df).head(10)
        if filtered.empty:
            st.warning("Strict bullish filter me clean setup nahi mila — neeche best available setups diye gaye hain. ENTRY status ko follow karein.")
        cols = ["symbol", "price", "signal", "bull_score", "score", "entry_status", "entry", "sl", "t1", "t2", "t3", "t1_pct", "t2_pct", "t3_pct"]
        st.dataframe(top[cols].rename(columns={"entry_status":"Entry Status","entry":"Entry","sl":"SL","t1":"T1","t2":"T2","t3":"T3","t1_pct":"T1 Profit %","t2_pct":"T2 Profit %","t3_pct":"T3 Profit %"}).round(2), use_container_width=True, hide_index=True)

        st.subheader("🔎 Top 5 Detailed Analysis")
        for rank, (_, row) in enumerate(top.head(5).iterrows(), start=1):
            with st.expander(f"#{rank} {row['symbol']} — {row['signal']} — Bull {row['bull_score']:.0f}%"):
                d = st.columns(6)
                d[0].metric("Price", fmt(row["price"]))
                d[1].metric("Bull", f"{row['bull_score']:.0f}%")
                d[2].metric("Edge", f"{row['score']:.0f}%")
                d[3].metric("Entry", fmt(row["entry"]))
                d[4].metric("SL", fmt(row["sl"]))
                d[5].metric("T1", fmt(row["t1"]))
                st.caption(f"T2 ₹{fmt(row['t2'])} • T3 ₹{fmt(row['t3'])} • Approx T1 {pct(row['t1_pct'])} / T2 {pct(row['t2_pct'])} / T3 {pct(row['t3_pct'])} • {row['entry_status']}")
                st.info("🚪 EXIT: SL hit = exit. T1 hit = partial profit book + SL trail. T2/T3 = further booking. 15:20 IST tak open intraday position square-off.")

                st.write("### 🟢 Bullish Confirmation")
                for reason in row["reasons"]:
                    st.success("✓ " + reason)
                if row["warnings"]:
                    st.write("### ⚠️ Bearish / Risk Factors")
                    for warning in row["warnings"]:
                        st.warning(warning)

                st.write("### 🏢 Fundamental Snapshot")
                f = get_fundamentals(row["ticker"])
                cap = f["market_cap"]
                cap_label = "-"
                if finite(cap):
                    cap_label = "Large Cap" if cap >= 2e12 else "Mid Cap" if cap >= 5e11 else "Small Cap"
                ff = st.columns(4)
                ff[0].metric("Market Cap", f"₹{cap / 1e7:,.0f} Cr" if finite(cap) else "-")
                ff[1].metric("ROE", pct(f["roe"]))
                ff[2].metric("Debt/Equity", fmt(f["debt_equity"]))
                ff[3].metric("Category", cap_label)
                ff2 = st.columns(3)
                ff2[0].metric("Revenue Growth", pct(f["revenue_growth"]))
                ff2[1].metric("Earnings Growth", pct(f["earnings_growth"]))
                ff2[2].metric("Profit Margin", pct(f["profit_margin"]))

        st.subheader("📋 Complete Scan Results")
        cols = ["symbol", "price", "signal", "bull_score", "score", "entry_status", "entry", "sl", "t1", "t2", "t1_pct", "t2_pct"]
        st.dataframe(result_df[cols].rename(columns={"entry_status":"Entry Status","entry":"Entry","sl":"SL","t1":"T1","t2":"T2","t1_pct":"T1 Profit %","t2_pct":"T2 Profit %"}).round(2), use_container_width=True, hide_index=True)

        st.subheader("📊 Strongest Bullish Setups")
        chart_df = result_df.sort_values("bull_score", ascending=False).head(10).set_index("symbol")[["bull_score"]]
        st.bar_chart(chart_df)

# ============================================================
# DISCLAIMER
# ============================================================

st.divider()
st.warning("""
⚠️ IMPORTANT

Ye app probability/confirmation based analysis hai. Guaranteed profit nahi hai.

V3 improvements:
• Last candle blindly remove nahi hota; market-open state ke hisaab se forming candle detect hota hai.
• Intraday EMA200 aur Daily EMA200 separate hain.
• Bullish/bearish scoring weights cleaned and correlated indicators ka over-weighting reduced hai.
• Stock ranking bull score ke saath actual bull-vs-bear edge par bhi based hai.
• NIFTY options action sirf STRONG BUY / STRONG SELL confirmation par activate hota hai.
• Option selection spread, liquidity, OI aur distance ko combine karta hai.
• Max Pain all available strikes par calculate hota hai.
• Yahoo download failures graceful handling ke saath hain.
• Duplicate refresh logic remove kiya gaya hai.
• Fundamental calls cached hain.
• Missing data se fake signal generate nahi hota.

Actual trade se pehle broker ka live price, bid/ask, liquidity, slippage, position size aur risk verify karein.
""")
st.caption("Last app refresh: " + datetime.now(IST).strftime("%d-%m-%Y %H:%M:%S IST"))

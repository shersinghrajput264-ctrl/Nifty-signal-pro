# ============================================================
# NIFTY + STOCK SIGNAL PRO V4
# Robust • Completed Candle • Liquidity Aware • Error Safe
# ============================================================

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo
import logging
import math

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# ============================================================
# APP CONFIG
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
OPTION_TTL = 45
FUNDAMENTAL_TTL = 3600

INTRADAY_INTERVAL = "5m"

MIN_INTRADAY_CANDLES = 100
MIN_DAILY_CANDLES = 220

MAX_WORKERS = 5

logging.basicConfig(level=logging.WARNING)


# ============================================================
# STOCK UNIVERSE
# ============================================================

DEFAULT_STOCKS = [
    "RELIANCE.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "INFY.NS",
    "TCS.NS",
    "ITC.NS",
    "LT.NS",
    "AXISBANK.NS",
    "KOTAKBANK.NS",
    "HINDUNILVR.NS",
    "M&M.NS",
    "SUNPHARMA.NS",
    "MARUTI.NS",
    "BAJFINANCE.NS",
    "TITAN.NS",
    "HCLTECH.NS",
    "NTPC.NS",
    "ONGC.NS",
    "POWERGRID.NS",
    "TATASTEEL.NS",
    "JSWSTEEL.NS",
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "COALINDIA.NS",
    "WIPRO.NS",
    "TECHM.NS",
    "TATAMOTORS.NS",
    "ASIANPAINT.NS",
    "ULTRACEMCO.NS",
    "NESTLEIND.NS",
    "BAJAJFINSV.NS",
    "HINDALCO.NS",
    "GRASIM.NS",
    "CIPLA.NS",
    "DRREDDY.NS",
    "DIVISLAB.NS",
    "EICHERMOT.NS",
    "HEROMOTOCO.NS",
    "APOLLOHOSP.NS",
    "BRITANNIA.NS",
    "BPCL.NS",
    "IOC.NS",
    "GAIL.NS",
    "TATACONSUM.NS",
    "BEL.NS",
    "HAL.NS",
    "IRFC.NS",
    "RVNL.NS",
    "DLF.NS",
    "TRENT.NS",
    "VBL.NS",
    "ZOMATO.NS",
    "JIOFIN.NS",
]


# ============================================================
# SESSION STATE
# ============================================================

if "stock_results" not in st.session_state:
    st.session_state.stock_results = None

if "last_stock_scan" not in st.session_state:
    st.session_state.last_stock_scan = None

if "nifty_last_refresh" not in st.session_state:
    st.session_state.nifty_last_refresh = None


# ============================================================
# GENERAL HELPERS
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

        if np.isfinite(value):
            return value

        return default

    except (TypeError, ValueError, IndexError):
        return default


def finite(value):
    return bool(np.isfinite(safe_float(value)))


def fmt(value, digits=2):
    value = safe_float(value)

    if not finite(value):
        return "-"

    return f"{value:,.{digits}f}"


def pct(value, digits=1):
    value = safe_float(value)

    if not finite(value):
        return "-"

    return f"{value:.{digits}f}%"


def clean_symbol(symbol):
    if not symbol:
        return "NIFTY"

    return str(symbol).replace(".NS", "")


def market_status():
    now = datetime.now(IST)

    if now.weekday() >= 5:
        return "🔴 NSE closed — Weekend"

    if dt_time(9, 15) <= now.time() <= dt_time(15, 30):
        return "🟢 NSE market hours"

    return "🟡 NSE market closed"


def is_market_open_now():
    now = datetime.now(IST)

    return (
        now.weekday() < 5
        and dt_time(9, 15) <= now.time() <= dt_time(15, 30)
    )


# ============================================================
# OHLCV CLEANING
# ============================================================

def clean_ohlcv(data):
    if data is None or data.empty:
        return pd.DataFrame()

    df = data.copy()

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    # --------------------------------------------------------
    # Robust MultiIndex handling
    # --------------------------------------------------------

    if isinstance(df.columns, pd.MultiIndex):
        flattened = []

        for col in df.columns:
            values = [str(x) for x in col]

            found = None

            for item in values:
                if item in required:
                    found = item
                    break

            flattened.append(
                found if found is not None else values[-1]
            )

        df.columns = flattened

    # Remove duplicated columns safely.
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="last")]

    if any(col not in df.columns for col in required):
        return pd.DataFrame()

    df = df[required].copy()

    for col in required:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
        ]
    )

    df["Volume"] = (
        df["Volume"]
        .fillna(0)
        .clip(lower=0)
    )

    # Remove impossible OHLC rows.
    df = df[
        (df["High"] >= df["Low"])
        & (df["High"] >= df["Open"])
        & (df["High"] >= df["Close"])
        & (df["Low"] <= df["Open"])
        & (df["Low"] <= df["Close"])
    ]

    if isinstance(df.index, pd.DatetimeIndex):

        try:
            if df.index.tz is not None:
                df.index = df.index.tz_convert(IST)
            else:
                df.index = df.index.tz_localize(IST)
        except Exception:
            pass

        df = (
            df[
                ~df.index.duplicated(
                    keep="last"
                )
            ]
            .sort_index()
        )

    return df


# ============================================================
# BATCH SYMBOL EXTRACTION
# ============================================================

def extract_symbol(raw, symbol):

    if raw is None or raw.empty:
        return pd.DataFrame()

    try:

        if isinstance(raw.columns, pd.MultiIndex):

            level0 = list(
                raw.columns.get_level_values(0)
            )

            level1 = list(
                raw.columns.get_level_values(1)
            )

            # Case: ticker is first level.
            if symbol in level0:
                selected = raw[symbol]
                return clean_ohlcv(selected)

            # Case: ticker is second level.
            if symbol in level1:
                selected = raw.xs(
                    symbol,
                    axis=1,
                    level=1,
                )
                return clean_ohlcv(selected)

        return clean_ohlcv(raw)

    except Exception as exc:
        logging.warning(
            "Symbol extraction failed for %s: %s",
            symbol,
            exc,
        )

        return pd.DataFrame()


# ============================================================
# YAHOO DOWNLOAD ENGINE
# ============================================================

def _download_batch(
    symbols,
    period,
    interval,
):

    symbols = tuple(
        dict.fromkeys(
            str(x) for x in symbols if x
        )
    )

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

        logging.warning(
            "Yahoo download failed: %s",
            exc,
        )

        return {}

    result = {}

    minimum = (
        MIN_INTRADAY_CANDLES
        if interval != "1d"
        else MIN_DAILY_CANDLES
    )

    for symbol in symbols:

        df = extract_symbol(
            raw,
            symbol,
        )

        if len(df) >= minimum:
            result[symbol] = df

    return result


@st.cache_data(
    ttl=DATA_TTL,
    show_spinner=False,
)
def download_intraday_batch(symbols):

    return _download_batch(
        tuple(symbols),
        "5d",
        INTRADAY_INTERVAL,
    )


@st.cache_data(
    ttl=DAILY_TTL,
    show_spinner=False,
)
def download_daily_batch(symbols):

    return _download_batch(
        tuple(symbols),
        "2y",
        "1d",
    )


@st.cache_data(
    ttl=DATA_TTL,
    show_spinner=False,
)
def get_nifty_intraday():

    data = download_intraday_batch(
        (NIFTY,)
    )

    return data.get(
        NIFTY,
        pd.DataFrame(),
    )


@st.cache_data(
    ttl=DAILY_TTL,
    show_spinner=False,
)
def get_nifty_daily():

    data = download_daily_batch(
        (NIFTY,)
    )

    return data.get(
        NIFTY,
        pd.DataFrame(),
    )


# ============================================================
# COMPLETED CANDLE ENGINE
# ============================================================

def get_completed_intraday_df(df):

    if df is None or len(df) < 3:
        return pd.DataFrame()

    out = df.copy()

    if not isinstance(
        out.index,
        pd.DatetimeIndex,
    ):
        return out.iloc[:-1].copy()

    now = datetime.now(IST)

    try:

        last_ts = out.index[-1]

        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(
                tzinfo=IST
            )
        else:
            last_ts = last_ts.astimezone(
                IST
            )

    except Exception:

        return out.iloc[:-1].copy()

    # Outside market hours, Yahoo normally
    # gives the last completed candle.
    if not is_market_open_now():
        return out

    bucket_minute = (
        now.minute // 5
    ) * 5

    current_bucket = now.replace(
        minute=bucket_minute,
        second=0,
        microsecond=0,
    )

    # If the last timestamp is in the current
    # forming 5-minute bucket, remove it.
    if last_ts >= current_bucket:
        return out.iloc[:-1].copy()

    return out


# ============================================================
# RSI
# ============================================================

def rsi_wilder(
    close,
    period=14,
):

    change = close.diff()

    gain = change.clip(
        lower=0
    )

    loss = -change.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = (
        avg_gain
        / avg_loss.replace(
            0,
            np.nan,
        )
    )

    rsi = (
        100
        - (
            100
            / (1 + rs)
        )
    )

    rsi.loc[
        (avg_loss == 0)
        & (avg_gain > 0)
    ] = 100

    rsi.loc[
        (avg_gain == 0)
        & (avg_loss > 0)
    ] = 0

    rsi.loc[
        (avg_gain == 0)
        & (avg_loss == 0)
    ] = 50

    return rsi


# ============================================================
# INTRADAY INDICATORS
# ============================================================

def calculate_intraday_indicators(df):

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = (
        df["Volume"]
        .fillna(0)
        .clip(lower=0)
    )

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    for span in (
        9,
        20,
        50,
        200,
    ):

        df[f"EMA{span}"] = (
            close
            .ewm(
                span=span,
                adjust=False,
                min_periods=span,
            )
            .mean()
        )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    df["RSI"] = rsi_wilder(
        close,
        14,
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (
                high
                - previous_close
            ).abs(),
            (
                low
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["ATR"] = tr.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    # --------------------------------------------------------
    # SESSION VWAP
    # --------------------------------------------------------

    typical_price = (
        high
        + low
        + close
    ) / 3

    pv = (
        typical_price
        * volume
    )

    if isinstance(
        df.index,
        pd.DatetimeIndex,
    ):

        session = pd.Series(
            df.index.date,
            index=df.index,
        )

        cumulative_pv = (
            pv
            .groupby(session)
            .cumsum()
        )

        cumulative_volume = (
            volume
            .groupby(session)
            .cumsum()
        )

        df["VWAP"] = (
            cumulative_pv
            / cumulative_volume.replace(
                0,
                np.nan,
            )
        )

    else:

        df["VWAP"] = (
            pv.cumsum()
            / volume.cumsum().replace(
                0,
                np.nan,
            )
        )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    df["VOL_AVG20"] = (
        volume
        .shift(1)
        .rolling(
            20,
            min_periods=10,
        )
        .mean()
    )

    df["VOL_RATIO"] = np.where(
        df["VOL_AVG20"] > 0,
        volume
        / df["VOL_AVG20"],
        np.nan,
    )

    # --------------------------------------------------------
    # BREAKOUT
    # --------------------------------------------------------

    df["HIGH10"] = (
        high
        .shift(1)
        .rolling(
            10,
            min_periods=10,
        )
        .max()
    )

    df["LOW10"] = (
        low
        .shift(1)
        .rolling(
            10,
            min_periods=10,
        )
        .min()
    )

    df["HIGH20"] = (
        high
        .shift(1)
        .rolling(
            20,
            min_periods=20,
        )
        .max()
    )

    df["LOW20"] = (
        low
        .shift(1)
        .rolling(
            20,
            min_periods=20,
        )
        .min()
    )

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    df["ROC5"] = (
        close.pct_change(5)
        * 100
    )

    df["ROC10"] = (
        close.pct_change(10)
        * 100
    )

    # --------------------------------------------------------
    # CANDLE QUALITY
    # --------------------------------------------------------

    candle_range = (
        high - low
    ).replace(
        0,
        np.nan,
    )

    df["BODY_PCT"] = (
        (
            close
            - df["Open"]
        ).abs()
        / candle_range
    )

    df["CLOSE_LOCATION"] = (
        (
            close
            - low
        )
        / candle_range
    )

    return df


# ============================================================
# DAILY INDICATORS
# ============================================================

def calculate_daily_indicators(df):

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    close = df["Close"]

    df["EMA50_DAILY"] = (
        close
        .ewm(
            span=50,
            adjust=False,
            min_periods=50,
        )
        .mean()
    )

    df["EMA200_DAILY"] = (
        close
        .ewm(
            span=200,
            adjust=False,
            min_periods=200,
        )
        .mean()
    )

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

MAX_SCORE = sum(
    BULL_WEIGHTS.values()
)


def classify_signal(
    bull_pct,
    bear_pct,
):

    if not finite(bull_pct):
        return "NO EDGE"

    if not finite(bear_pct):
        return "NO EDGE"

    difference = abs(
        bull_pct
        - bear_pct
    )

    if (
        bull_pct >= 72
        and difference >= 18
    ):
        return "STRONG BUY"

    if (
        bull_pct >= 62
        and difference >= 12
    ):
        return "BUY WATCH"

    if (
        bear_pct >= 72
        and difference >= 18
    ):
        return "STRONG SELL"

    if (
        bear_pct >= 62
        and difference >= 12
    ):
        return "SELL WATCH"

    if (
        bull_pct >= 52
        and bull_pct > bear_pct
    ):
        return "WATCH BUY"

    if (
        bear_pct >= 52
        and bear_pct > bull_pct
    ):
        return "WATCH SELL"

    return "NO EDGE"


# ============================================================
# ANALYSIS ENGINE
# ============================================================

def analyze_frame(
    intraday,
    daily,
    symbol=None,
):

    if (
        intraday is None
        or daily is None
        or intraday.empty
        or daily.empty
    ):
        return None

    # Calculate indicators BEFORE
    # removing forming candle.
    intraday = calculate_intraday_indicators(
        intraday
    )

    daily = calculate_daily_indicators(
        daily
    )

    intraday = get_completed_intraday_df(
        intraday
    )

    if (
        len(intraday)
        < MIN_INTRADAY_CANDLES
    ):
        return None

    if (
        len(daily)
        < MIN_DAILY_CANDLES
    ):
        return None

    last = intraday.iloc[-1]
    previous = intraday.iloc[-2]

    daily_last = daily.iloc[-1]

    price = safe_float(
        last["Close"]
    )

    if not finite(price):
        return None

    indicator_names = [
        "VWAP",
        "EMA9",
        "EMA20",
        "EMA50",
        "EMA200",
        "RSI",
        "ATR",
        "ROC5",
        "ROC10",
        "VOL_RATIO",
        "HIGH10",
        "LOW10",
        "BODY_PCT",
        "CLOSE_LOCATION",
    ]

    values = {
        name: safe_float(
            last[name]
        )
        for name in indicator_names
    }

    daily_ema200 = safe_float(
        daily_last[
            "EMA200_DAILY"
        ]
    )

    previous_ema20 = safe_float(
        previous["EMA20"]
    )

    previous_ema50 = safe_float(
        previous["EMA50"]
    )

    open_price = safe_float(
        last["Open"]
    )

    # --------------------------------------------------------
    # BULLISH
    # --------------------------------------------------------

    bullish = {

        "Price above VWAP":
            finite(values["VWAP"])
            and price > values["VWAP"],

        "EMA9 > EMA20":
            finite(values["EMA9"])
            and finite(values["EMA20"])
            and values["EMA9"]
            > values["EMA20"],

        "EMA20 > EMA50":
            finite(values["EMA20"])
            and finite(values["EMA50"])
            and values["EMA20"]
            > values["EMA50"],

        "EMA20 rising":
            finite(values["EMA20"])
            and finite(previous_ema20)
            and values["EMA20"]
            > previous_ema20,

        "EMA50 rising":
            finite(values["EMA50"])
            and finite(previous_ema50)
            and values["EMA50"]
            > previous_ema50,

        "Above daily EMA200":
            finite(daily_ema200)
            and price > daily_ema200,

        "RSI 52-72":
            finite(values["RSI"])
            and 52
            <= values["RSI"]
            <= 72,

        "ROC5 positive":
            finite(values["ROC5"])
            and values["ROC5"] > 0,

        "ROC10 positive":
            finite(values["ROC10"])
            and values["ROC10"] > 0,

        "Volume >= 1.2x":
            finite(values["VOL_RATIO"])
            and values["VOL_RATIO"]
            >= 1.20,

        "10-candle breakout":
            finite(values["HIGH10"])
            and price > values["HIGH10"],

        "Strong bullish candle":
            (
                finite(open_price)
                and price > open_price
                and finite(values["BODY_PCT"])
                and finite(
                    values["CLOSE_LOCATION"]
                )
                and values["BODY_PCT"]
                >= 0.45
                and values["CLOSE_LOCATION"]
                >= 0.65
            ),
    }

    # --------------------------------------------------------
    # BEARISH
    # --------------------------------------------------------

    bearish = {

        "Price below VWAP":
            finite(values["VWAP"])
            and price < values["VWAP"],

        "EMA9 < EMA20":
            finite(values["EMA9"])
            and finite(values["EMA20"])
            and values["EMA9"]
            < values["EMA20"],

        "EMA20 < EMA50":
            finite(values["EMA20"])
            and finite(values["EMA50"])
            and values["EMA20"]
            < values["EMA50"],

        "EMA20 falling":
            finite(values["EMA20"])
            and finite(previous_ema20)
            and values["EMA20"]
            < previous_ema20,

        "EMA50 falling":
            finite(values["EMA50"])
            and finite(previous_ema50)
            and values["EMA50"]
            < previous_ema50,

        "Below daily EMA200":
            finite(daily_ema200)
            and price < daily_ema200,

        "RSI 28-48":
            finite(values["RSI"])
            and 28
            <= values["RSI"]
            <= 48,

        "ROC5 negative":
            finite(values["ROC5"])
            and values["ROC5"] < 0,

        "ROC10 negative":
            finite(values["ROC10"])
            and values["ROC10"] < 0,

        "Volume >= 1.2x":
            finite(values["VOL_RATIO"])
            and values["VOL_RATIO"]
            >= 1.20,

        "10-candle breakdown":
            finite(values["LOW10"])
            and price < values["LOW10"],

        "Strong bearish candle":
            (
                finite(open_price)
                and price < open_price
                and finite(values["BODY_PCT"])
                and finite(
                    values["CLOSE_LOCATION"]
                )
                and values["BODY_PCT"]
                >= 0.45
                and values["CLOSE_LOCATION"]
                <= 0.35
            ),
    }

    # --------------------------------------------------------
    # SCORES
    # --------------------------------------------------------

    bull_score = sum(
        BULL_WEIGHTS[key]
        for key, value in bullish.items()
        if value
    )

    bear_score = sum(
        BEAR_WEIGHTS[key]
        for key, value in bearish.items()
        if value
    )

    bull_pct = (
        bull_score
        / MAX_SCORE
    ) * 100

    bear_pct = (
        bear_score
        / MAX_SCORE
    ) * 100

    score_difference = (
        bull_pct
        - bear_pct
    )

    signal = classify_signal(
        bull_pct,
        bear_pct,
    )

    # Strong signal is invalid if opposite
    # side is also very strong.
    if (
        signal == "STRONG BUY"
        and bear_pct >= 55
    ):
        signal = "NO EDGE"

    if (
        signal == "STRONG SELL"
        and bull_pct >= 55
    ):
        signal = "NO EDGE"

    return {

        "symbol": clean_symbol(
            symbol
        ),

        "ticker": (
            symbol
            if symbol
            else NIFTY
        ),

        "price": price,

        "bull_score": bull_pct,

        "bear_score": bear_pct,

        "score": score_difference,

        "signal": signal,

        "rsi": values["RSI"],

        "vwap": values["VWAP"],

        "ema9": values["EMA9"],

        "ema20": values["EMA20"],

        "ema50": values["EMA50"],

        "ema200_intraday":
            values["EMA200"],

        "daily_ema200":
            daily_ema200,

        "atr":
            values["ATR"],

        "roc5":
            values["ROC5"],

        "roc10":
            values["ROC10"],

        "volume_ratio":
            values["VOL_RATIO"],

        "bullish":
            bullish,

        "bearish":
            bearish,

        "reasons": [
            key
            for key, value
            in bullish.items()
            if value
        ],

        "warnings": [
            key
            for key, value
            in bearish.items()
            if value
        ],

        "candle_time":
            intraday.index[-1],

        "chart_df":
            intraday,
    }


def analyze_stock(
    symbol,
    intraday,
    daily,
):

    try:

        result = analyze_frame(
            intraday,
            daily,
            symbol,
        )

        if result is None:
            return None

        result["day_change"] = (
            calculate_day_change(
                daily,
                result["price"],
            )
        )

        return result

    except Exception as exc:

        logging.warning(
            "Analysis failed for %s: %s",
            symbol,
            exc,
        )

        return None


# ============================================================
# DAY CHANGE
# ============================================================

def calculate_day_change(
    daily_df,
    current_price,
):

    price = safe_float(
        current_price
    )

    if (
        daily_df is None
        or daily_df.empty
        or not finite(price)
    ):
        return np.nan

    df = daily_df.copy()

    previous_close = np.nan

    if isinstance(
        df.index,
        pd.DatetimeIndex,
    ):

        dates = [
            item.date()
            for item in df.index
        ]

        today = datetime.now(
            IST
        ).date()

        if today in dates:

            index = dates.index(
                today
            )

            if index > 0:

                previous_close = safe_float(
                    df.iloc[
                        index - 1
                    ]["Close"]
                )

    if not finite(previous_close):

        previous_close = safe_float(
            df.iloc[-1]["Close"]
        )

    if (
        not finite(previous_close)
        or previous_close <= 0
    ):
        return np.nan

    return (
        (
            price
            / previous_close
        )
        - 1
    ) * 100


# ============================================================
# NIFTY ATM
# ============================================================

def get_atm(price):

    price = safe_float(price)

    if not finite(price):
        return np.nan

    return int(
        math.floor(
            price / 50
            + 0.5
        )
        * 50
    )


# ============================================================
# OPTION CHAIN
# ============================================================

def clean_options(df):

    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    numeric_columns = [
        "strike",
        "lastPrice",
        "bid",
        "ask",
        "change",
        "percentChange",
        "volume",
        "openInterest",
        "impliedVolatility",
    ]

    for col in numeric_columns:

        if col in out.columns:

            out[col] = pd.to_numeric(
                out[col],
                errors="coerce",
            )

    for col in [
        "volume",
        "openInterest",
    ]:

        if col not in out.columns:
            out[col] = 0.0

        out[col] = (
            out[col]
            .fillna(0)
            .clip(lower=0)
        )

    if "strike" in out.columns:

        out = out[
            out["strike"].notna()
        ].copy()

    return out


@st.cache_data(
    ttl=OPTION_TTL,
    show_spinner=False,
)
def get_nifty_options():

    try:

        ticker = yf.Ticker(
            NIFTY
        )

        expirations = list(
            ticker.options or []
        )

        if not expirations:

            return (
                None,
                pd.DataFrame(),
                pd.DataFrame(),
                [],
            )

        today = datetime.now(
            IST
        ).date()

        valid_expiries = []

        for expiry in expirations:

            try:

                expiry_date = (
                    pd.Timestamp(
                        expiry
                    ).date()
                )

                if expiry_date >= today:
                    valid_expiries.append(
                        expiry
                    )

            except Exception:
                continue

        if not valid_expiries:

            return (
                None,
                pd.DataFrame(),
                pd.DataFrame(),
                [],
            )

        valid_expiries = sorted(
            valid_expiries
        )

        expiry = valid_expiries[0]

        chain = ticker.option_chain(
            expiry
        )

        calls = clean_options(
            chain.calls
        )

        puts = clean_options(
            chain.puts
        )

        return (
            expiry,
            calls,
            puts,
            valid_expiries,
        )

    except Exception as exc:

        logging.warning(
            "Option chain failed: %s",
            exc,
        )

        return (
            None,
            pd.DataFrame(),
            pd.DataFrame(),
            [],
        )


# ============================================================
# OPTION SELECTION
# ============================================================

def select_option(
    df,
    atm,
    direction,
):

    if (
        df is None
        or df.empty
        or not finite(atm)
    ):
        return None

    required = [
        "strike",
        "lastPrice",
        "bid",
        "ask",
        "volume",
        "openInterest",
    ]

    for col in required:

        if col not in df.columns:
            return None

    work = df.copy()

    work = work[
        work["strike"].notna()
    ].copy()

    if work.empty:
        return None

    work["distance"] = (
        work["strike"]
        - atm
    ).abs()

    work = work[
        work["distance"] <= 300
    ].copy()

    if direction == "CE":

        work = work[
            (
                work["strike"]
                >= atm - 50
            )
            &
            (
                work["strike"]
                <= atm + 100
            )
        ].copy()

    elif direction == "PE":

        work = work[
            (
                work["strike"]
                >= atm - 100
            )
            &
            (
                work["strike"]
                <= atm + 50
            )
        ].copy()

    else:

        return None

    if work.empty:
        return None

    bid = pd.to_numeric(
        work["bid"],
        errors="coerce",
    ).fillna(0)

    ask = pd.to_numeric(
        work["ask"],
        errors="coerce",
    ).fillna(0)

    last_price = pd.to_numeric(
        work["lastPrice"],
        errors="coerce",
    ).fillna(0)

    valid_market = (
        (bid > 0)
        &
        (ask > 0)
        &
        (ask >= bid)
    )

    work["mid"] = np.where(
        valid_market,
        (bid + ask) / 2,
        last_price,
    )

    work["spread"] = np.where(
        valid_market,
        ask - bid,
        np.nan,
    )

    work["spread_pct"] = np.where(
        (
            work["mid"] > 0
        )
        &
        np.isfinite(
            work["spread"]
        ),
        (
            work["spread"]
            / work["mid"]
        ) * 100,
        999,
    )

    work = work[
        work["mid"] > 0
    ].copy()

    if work.empty:
        return None

    # Hard liquidity filter.
    work = work[
        work["spread_pct"] <= 12
    ].copy()

    if work.empty:
        return None

    volume = pd.to_numeric(
        work["volume"],
        errors="coerce",
    ).fillna(0)

    oi = pd.to_numeric(
        work["openInterest"],
        errors="coerce",
    ).fillna(0)

    work["liquidity_score"] = (
        np.log1p(volume) * 2
        + np.log1p(oi)
    )

    work["distance_penalty"] = (
        work["distance"]
        / 50
    )

    work["spread_penalty"] = (
        work["spread_pct"]
        .clip(0, 12)
        * 2
    )

    work["selection_score"] = (
        work["liquidity_score"]
        - work["distance_penalty"]
        - work["spread_penalty"]
    )

    work = work.sort_values(
        "selection_score",
        ascending=False,
    )

    if work.empty:
        return None

    return work.iloc[0]


# ============================================================
# PCR
# ============================================================

def calculate_pcr(
    calls,
    puts,
):

    if (
        calls is None
        or puts is None
        or calls.empty
        or puts.empty
    ):
        return (
            np.nan,
            np.nan,
        )

    call_oi = safe_float(
        calls["openInterest"].sum(),
        0,
    )

    put_oi = safe_float(
        puts["openInterest"].sum(),
        0,
    )

    call_volume = safe_float(
        calls["volume"].sum(),
        0,
    )

    put_volume = safe_float(
        puts["volume"].sum(),
        0,
    )

    oi_pcr = (
        put_oi / call_oi
        if call_oi > 0
        else np.nan
    )

    volume_pcr = (
        put_volume
        / call_volume
        if call_volume > 0
        else np.nan
    )

    return (
        oi_pcr,
        volume_pcr,
    )


# ============================================================
# MAX PAIN
# ============================================================

def calculate_max_pain(
    calls,
    puts,
):

    if (
        calls is None
        or puts is None
        or calls.empty
        or puts.empty
    ):
        return np.nan

    c = calls.copy()
    p = puts.copy()

    required = [
        "strike",
        "openInterest",
    ]

    if any(
        col not in c.columns
        for col in required
    ):
        return np.nan

    if any(
        col not in p.columns
        for col in required
    ):
        return np.nan

    c = c[
        c["strike"].notna()
    ].copy()

    p = p[
        p["strike"].notna()
    ].copy()

    if c.empty or p.empty:
        return np.nan

    c["openInterest"] = (
        pd.to_numeric(
            c["openInterest"],
            errors="coerce",
        )
        .fillna(0)
        .clip(lower=0)
    )

    p["openInterest"] = (
        pd.to_numeric(
            p["openInterest"],
            errors="coerce",
        )
        .fillna(0)
        .clip(lower=0)
    )

    c = (
        c.groupby(
            "strike",
            as_index=False,
        )["openInterest"]
        .sum()
    )

    p = (
        p.groupby(
            "strike",
            as_index=False,
        )["openInterest"]
        .sum()
    )

    call_strikes = c[
        "strike"
    ].to_numpy(dtype=float)

    call_oi = c[
        "openInterest"
    ].to_numpy(dtype=float)

    put_strikes = p[
        "strike"
    ].to_numpy(dtype=float)

    put_oi = p[
        "openInterest"
    ].to_numpy(dtype=float)

    strikes = sorted(
        set(call_strikes)
        .union(
            set(put_strikes)
        )
    )

    if not strikes:
        return np.nan

    best_strike = np.nan
    lowest_loss = np.inf

    for settlement in strikes:

        call_loss = np.sum(
            np.maximum(
                settlement
                - call_strikes,
                0,
            )
            * call_oi
        )

        put_loss = np.sum(
            np.maximum(
                put_strikes
                - settlement,
                0,
            )
            * put_oi
        )

        total_loss = (
            call_loss
            + put_loss
        )

        if total_loss < lowest_loss:

            lowest_loss = (
                total_loss
            )

            best_strike = (
                settlement
            )

    return best_strike


# ============================================================
# OI SUPPORT / RESISTANCE
# ============================================================

def get_oi_levels(
    calls,
    puts,
    price,
):

    if (
        calls is None
        or puts is None
        or calls.empty
        or puts.empty
    ):
        return (
            np.nan,
            np.nan,
        )

    price = safe_float(
        price
    )

    if not finite(price):
        return (
            np.nan,
            np.nan,
        )

    calls_near = calls[
        (
            calls["strike"]
            >= price
        )
        &
        (
            calls["strike"]
            <= price + 500
        )
    ].copy()

    puts_near = puts[
        (
            puts["strike"]
            <= price
        )
        &
        (
            puts["strike"]
            >= price - 500
        )
    ].copy()

    resistance = np.nan
    support = np.nan

    if not calls_near.empty:

        calls_near["level_score"] = (
            calls_near[
                "openInterest"
            ]
            /
            (
                1
                +
                (
                    calls_near[
                        "strike"
                    ]
                    - price
                ) / 50
            )
        )

        resistance = safe_float(
            calls_near.loc[
                calls_near[
                    "level_score"
                ].idxmax(),
                "strike",
            ]
        )

    if not puts_near.empty:

        puts_near["level_score"] = (
            puts_near[
                "openInterest"
            ]
            /
            (
                1
                +
                (
                    price
                    - puts_near[
                        "strike"
                    ]
                ) / 50
            )
        )

        support = safe_float(
            puts_near.loc[
                puts_near[
                    "level_score"
                ].idxmax(),
                "strike",
            ]
        )

    return (
        resistance,
        support,
    )


# ============================================================
# TRADE PLAN
# ============================================================

def build_trade_plan(
    signal,
    option_row,
    nifty_price,
    atr,
):

    if (
        option_row is None
        or signal
        not in {
            "BUY CE",
            "BUY PE",
        }
    ):
        return None

    nifty_price = safe_float(
        nifty_price
    )

    atr = safe_float(
        atr
    )

    if (
        not finite(nifty_price)
        or not finite(atr)
        or atr <= 0
    ):
        return None

    bid = safe_float(
        option_row.get("bid")
    )

    ask = safe_float(
        option_row.get("ask")
    )

    last_price = safe_float(
        option_row.get("lastPrice")
    )

    if (
        finite(bid)
        and finite(ask)
        and bid > 0
        and ask > 0
        and ask >= bid
    ):

        entry = (
            bid + ask
        ) / 2

    elif (
        finite(last_price)
        and last_price > 0
    ):

        entry = last_price

    else:

        return None

    if entry <= 0:
        return None

    # Conservative option premium SL.
    stop_loss = (
        entry * 0.82
    )

    risk = (
        entry
        - stop_loss
    )

    target1 = (
        entry
        + risk * 1.5
    )

    target2 = (
        entry
        + risk * 2.5
    )

    target3 = (
        entry
        + risk * 4.0
    )

    if signal == "BUY CE":

        nifty_stop = (
            nifty_price
            - 0.80 * atr
        )

    else:

        nifty_stop = (
            nifty_price
            + 0.80 * atr
        )

    return {

        "entry": entry,

        "stop_loss":
            stop_loss,

        "target1":
            target1,

        "target2":
            target2,

        "target3":
            target3,

        "nifty_stop":
            nifty_stop,
    }


# ============================================================
# FUNDAMENTALS
# ============================================================

@st.cache_data(
    ttl=FUNDAMENTAL_TTL,
    show_spinner=False,
)
def get_fundamentals(symbol):

    empty = {
        "market_cap": np.nan,
        "roe": np.nan,
        "debt_equity": np.nan,
        "profit_margin": np.nan,
        "revenue_growth": np.nan,
        "earnings_growth": np.nan,
    }

    try:

        info = yf.Ticker(
            symbol
        ).info

        if not isinstance(
            info,
            dict,
        ):
            return empty

        result = {
            "market_cap":
                safe_float(
                    info.get(
                        "marketCap"
                    )
                ),

            "roe":
                safe_float(
                    info.get(
                        "returnOnEquity"
                    )
                ),

            "debt_equity":
                safe_float(
                    info.get(
                        "debtToEquity"
                    )
                ),

            "profit_margin":
                safe_float(
                    info.get(
                        "profitMargins"
                    )
                ),

            "revenue_growth":
                safe_float(
                    info.get(
                        "revenueGrowth"
                    )
                ),

            "earnings_growth":
                safe_float(
                    info.get(
                        "earningsGrowth"
                    )
                ),
        }

        # Yahoo usually returns these as decimals.
        for key in [
            "roe",
            "profit_margin",
            "revenue_growth",
            "earnings_growth",
        ]:

            value = result[key]

            if finite(value):

                # If already supplied as a percentage,
                # do not multiply again.
                if abs(value) <= 1.5:
                    result[key] = (
                        value * 100
                    )

        return result

    except Exception as exc:

        logging.warning(
            "Fundamental data failed for %s: %s",
            symbol,
            exc,
        )

        return empty


def get_cap_label(
    market_cap
):

    market_cap = safe_float(
        market_cap
    )

    if not finite(market_cap):
        return "-"

    if market_cap >= 2e12:
        return "Large Cap"

    if market_cap >= 5e11:
        return "Mid Cap"

    return "Small Cap"


# ============================================================
# NIFTY RENDER
# ============================================================

def render_nifty():

    with st.spinner(
        "NIFTY data load ho raha hai..."
    ):

        nifty_intraday = (
            get_nifty_intraday()
        )

        nifty_daily = (
            get_nifty_daily()
        )

    if (
        nifty_intraday.empty
        or nifty_daily.empty
    ):

        st.error(
            "❌ NIFTY data available nahi hai."
        )

        st.info(
            "Yahoo Finance temporary "
            "rate-limit/data issue de sakta hai. "
            "Thodi der baad refresh karein."
        )

        return

    market = analyze_frame(
        nifty_intraday,
        nifty_daily,
        NIFTY,
    )

    if market is None:

        st.warning(
            "⚠️ Valid completed candles / "
            "technical data sufficient nahi hai."
        )

        return

    price = market["price"]

    atm = get_atm(
        price
    )

    (
        expiry,
        calls,
        puts,
        expirations,
    ) = get_nifty_options()

    ce_row = select_option(
        calls,
        atm,
        "CE",
    )

    pe_row = select_option(
        puts,
        atm,
        "PE",
    )

    oi_pcr, volume_pcr = (
        calculate_pcr(
            calls,
            puts,
        )
    )

    max_pain = (
        calculate_max_pain(
            calls,
            puts,
        )
    )

    resistance, support = (
        get_oi_levels(
            calls,
            puts,
            price,
        )
    )

    # ========================================================
    # TOP METRICS
    # ========================================================

    st.subheader(
        "📊 NIFTY Snapshot"
    )

    m1, m2, m3, m4, m5 = (
        st.columns(5)
    )

    m1.metric(
        "NIFTY",
        fmt(price),
    )

    m2.metric(
        "ATM",
        str(atm),
    )

    m3.metric(
        "Bull Score",
        f'{market["bull_score"]:.0f}%',
    )

    m4.metric(
        "Bear Score",
        f'{market["bear_score"]:.0f}%',
    )

    m5.metric(
        "Signal",
        market["signal"],
    )

    st.caption(
        f"{market_status()} • "
        f"Signal last completed 5-minute candle "
        f"par based hai: "
        f"{market['candle_time']}"
    )

    # ========================================================
    # FINAL SIGNAL
    # ========================================================

    st.subheader(
        "🚨 FINAL SIGNAL"
    )

    signal = market["signal"]

    if signal == "STRONG BUY":

        st.success(
            "🟢 STRONG BUY CE — "
            "Bullish confirmation strong"
        )

    elif signal == "STRONG SELL":

        st.error(
            "🔴 STRONG BUY PE — "
            "Bearish confirmation strong"
        )

    elif signal == "BUY WATCH":

        st.info(
            "🟢 BUY WATCH — "
            "Confirmation ka wait karein."
        )

    elif signal == "SELL WATCH":

        st.info(
            "🔴 SELL WATCH — "
            "Confirmation ka wait karein."
        )

    elif signal == "WATCH BUY":

        st.info(
            "🟢 WATCH BUY — "
            "Setup developing hai."
        )

    elif signal == "WATCH SELL":

        st.info(
            "🔴 WATCH SELL — "
            "Setup developing hai."
        )

    else:

        st.warning(
            "🟡 WAIT — "
            "Strong confirmation nahi mili."
        )

    if signal == "STRONG BUY":
        action = "BUY CE"

    elif signal == "STRONG SELL":
        action = "BUY PE"

    else:
        action = "WAIT"

    # ========================================================
    # TECHNICAL DASHBOARD
    # ========================================================

    st.subheader(
        "📐 Technical Dashboard"
    )

    a = st.columns(7)

    a[0].metric(
        "EMA20",
        fmt(
            market["ema20"]
        ),
    )

    a[1].metric(
        "EMA50",
        fmt(
            market["ema50"]
        ),
    )

    a[2].metric(
        "Intraday EMA200",
        fmt(
            market[
                "ema200_intraday"
            ]
        ),
    )

    a[3].metric(
        "Daily EMA200",
        fmt(
            market[
                "daily_ema200"
            ]
        ),
    )

    a[4].metric(
        "VWAP",
        fmt(
            market["vwap"]
        ),
    )

    a[5].metric(
        "RSI",
        fmt(
            market["rsi"]
        ),
    )

    a[6].metric(
        "ATR",
        fmt(
            market["atr"]
        ),
    )

    # ========================================================
    # MOMENTUM
    # ========================================================

    st.subheader(
        "⚡ Momentum / Volume"
    )

    q1, q2, q3 = (
        st.columns(3)
    )

    q1.metric(
        "ROC5",
        pct(
            market["roc5"]
        ),
    )

    q2.metric(
        "ROC10",
        pct(
            market["roc10"]
        ),
    )

    volume_ratio = market[
        "volume_ratio"
    ]

    q3.metric(
        "Volume Ratio",
        (
            f"{volume_ratio:.2f}x"
            if finite(
                volume_ratio
            )
            else "-"
        ),
    )

    # ========================================================
    # OPTION CHAIN
    # ========================================================

    st.subheader(
        "⛓️ Option Chain Intelligence"
    )

    if (
        calls.empty
        or puts.empty
    ):

        st.warning(
            "⚠️ Yahoo Finance se "
            "NIFTY option chain available nahi hui."
        )

    else:

        o = st.columns(6)

        o[0].metric(
            "Expiry",
            str(expiry),
        )

        o[1].metric(
            "OI PCR",
            fmt(oi_pcr),
        )

        o[2].metric(
            "Volume PCR",
            fmt(volume_pcr),
        )

        o[3].metric(
            "Max Pain",
            fmt(
                max_pain,
                0,
            ),
        )

        o[4].metric(
            "OI Resistance",
            fmt(
                resistance,
                0,
            ),
        )

        o[5].metric(
            "OI Support",
            fmt(
                support,
                0,
            ),
        )

        rows = []

        for side, row in (
            ("CE", ce_row),
            ("PE", pe_row),
        ):

            if row is None:
                continue

            iv = safe_float(
                row.get(
                    "impliedVolatility"
                )
            )

            rows.append(
                {
                    "Side": side,

                    "Strike":
                        safe_float(
                            row.get(
                                "strike"
                            )
                        ),

                    "Last":
                        safe_float(
                            row.get(
                                "lastPrice"
                            )
                        ),

                    "Bid":
                        safe_float(
                            row.get(
                                "bid"
                            )
                        ),

                    "Ask":
                        safe_float(
                            row.get(
                                "ask"
                            )
                        ),

                    "Volume":
                        safe_float(
                            row.get(
                                "volume"
                            ),
                            0,
                        ),

                    "OI":
                        safe_float(
                            row.get(
                                "openInterest"
                            ),
                            0,
                        ),

                    "IV %":
                        (
                            iv * 100
                            if finite(iv)
                            else np.nan
                        ),
                }
            )

        if rows:

            st.dataframe(
                pd.DataFrame(
                    rows
                ).round(2),
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.info(
                "Near-ATM liquid option "
                "contract identify nahi hua."
            )

    # ========================================================
    # TRADE PLAN
    # ========================================================

    st.subheader(
        "🎯 Trade Plan"
    )

    if action == "BUY CE":
        selected_option = ce_row

    elif action == "BUY PE":
        selected_option = pe_row

    else:
        selected_option = None

    plan = build_trade_plan(
        action,
        selected_option,
        price,
        market["atr"],
    )

    if plan is None:

        st.info(
            "Trade plan unavailable. "
            "Strong signal + liquid option "
            "confirmation required."
        )

    else:

        p1, p2, p3, p4 = (
            st.columns(4)
        )

        p1.metric(
            "Entry",
            fmt(
                plan["entry"]
            ),
        )

        p2.metric(
            "SL",
            fmt(
                plan["stop_loss"]
            ),
        )

        p3.metric(
            "T1",
            fmt(
                plan["target1"]
            ),
        )

        p4.metric(
            "T2",
            fmt(
                plan["target2"]
            ),
        )

        st.caption(
            f"T3: "
            f"{fmt(plan['target3'])} | "
            f"Underlying NIFTY risk level: "
            f"{fmt(plan['nifty_stop'])}"
        )

    # ========================================================
    # CHECKLIST
    # ========================================================

    st.subheader(
        "✅ Confirmation Checklist"
    )

    left, right = (
        st.columns(2)
    )

    with left:

        st.markdown(
            "### 🟢 Bullish"
        )

        for name, value in (
            market["bullish"].items()
        ):

            st.write(
                (
                    "✅ "
                    if value
                    else "❌ "
                )
                + name
            )

    with right:

        st.markdown(
            "### 🔴 Bearish"
        )

        for name, value in (
            market["bearish"].items()
        ):

            st.write(
                (
                    "✅ "
                    if value
                    else "❌ "
                )
                + name
            )

    # ========================================================
    # CHART
    # ========================================================

    st.subheader(
        "📈 NIFTY 5-Minute Chart"
    )

    chart_columns = [
        "Close",
        "EMA20",
        "EMA50",
        "VWAP",
    ]

    chart_df = (
        market["chart_df"]
        .copy()
    )

    available_columns = [
        col
        for col in chart_columns
        if col in chart_df.columns
    ]

    if available_columns:

        chart_df = (
            chart_df[
                available_columns
            ]
            .tail(150)
            .dropna(
                how="all"
            )
        )

        if not chart_df.empty:

            st.line_chart(
                chart_df,
                height=450,
            )


# ============================================================
# TABS
# ============================================================

tab_nifty, tab_stocks = st.tabs(
    [
        "📊 NIFTY + OPTIONS",
        "🚀 STOCK SCANNER",
    ]
)


# ============================================================
# NIFTY TAB
# ============================================================

with tab_nifty:

    st.header(
        "📊 NIFTY Signal + Option Intelligence"
    )

    st.caption(
        market_status()
    )

    c1, c2 = (
        st.columns(2)
    )

    with c1:

        refresh_nifty = st.button(
            "🔄 Refresh NIFTY",
            use_container_width=True,
        )

    with c2:

        auto_refresh = st.checkbox(
            "⏱️ Auto Refresh (45s)",
            value=False,
        )

    if refresh_nifty:

        try:
            download_intraday_batch.clear()
        except Exception:
            pass

        try:
            download_daily_batch.clear()
        except Exception:
            pass

        try:
            get_nifty_intraday.clear()
        except Exception:
            pass

        try:
            get_nifty_daily.clear()
        except Exception:
            pass

        try:
            get_nifty_options.clear()
        except Exception:
            pass

        st.session_state.nifty_last_refresh = (
            datetime.now(IST)
        )

        st.rerun()

    # Streamlit fragment is supported in newer versions.
    if (
        auto_refresh
        and hasattr(st, "fragment")
    ):

        @st.fragment(
            run_every="45s"
        )
        def live_nifty():

            render_nifty()

        live_nifty()

    else:

        if auto_refresh:

            st.info(
                "Auto-refresh ke liye recent "
                "Streamlit version use karein. "
                "Abhi manual refresh available hai."
            )

        render_nifty()


# ============================================================
# STOCK SCANNER
# ============================================================

with tab_stocks:

    st.header(
        "🚀 NSE Stock Scanner"
    )

    st.caption(
        "Completed candle + Daily EMA200 + "
        "VWAP + EMA trend + RSI + momentum + "
        "volume + breakout"
    )

    s1, s2, s3 = (
        st.columns(3)
    )

    with s1:

        scan_count = st.selectbox(
            "Stocks to scan",
            [25, 50],
            index=1,
        )

    with s2:

        min_buy_score = st.slider(
            "Minimum bullish score %",
            40,
            90,
            62,
        )

    with s3:

        run_scan = st.button(
            "🚀 SCAN STOCKS",
            use_container_width=True,
        )

    symbols = DEFAULT_STOCKS[
        :min(
            scan_count,
            len(DEFAULT_STOCKS),
        )
    ]

    if run_scan:

        with st.spinner(
            f"{len(symbols)} stocks ka "
            "market data download ho raha hai..."
        ):

            intraday_data = (
                download_intraday_batch(
                    tuple(symbols)
                )
            )

            daily_data = (
                download_daily_batch(
                    tuple(symbols)
                )
            )

        results = []

        progress = st.progress(
            0.0
        )

        status = st.empty()

        def worker(symbol):

            return analyze_stock(
                symbol,
                intraday_data.get(
                    symbol,
                    pd.DataFrame(),
                ),
                daily_data.get(
                    symbol,
                    pd.DataFrame(),
                ),
            )

        futures = {}

        with ThreadPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:

            for symbol in symbols:

                futures[
                    executor.submit(
                        worker,
                        symbol,
                    )
                ] = symbol

            total = len(
                futures
            )

            for done, future in enumerate(
                as_completed(
                    futures
                ),
                start=1,
            ):

                symbol = futures[
                    future
                ]

                try:

                    result = (
                        future.result()
                    )

                    if result is not None:

                        results.append(
                            result
                        )

                except Exception as exc:

                    logging.warning(
                        "Stock analysis failed "
                        "for %s: %s",
                        symbol,
                        exc,
                    )

                if total:

                    progress.progress(
                        done / total
                    )

                status.write(
                    f"Analysing "
                    f"{done}/{total}..."
                )

        progress.empty()
        status.empty()

        if not results:

            st.error(
                "❌ Valid stock data nahi mila. "
                "Yahoo Finance data/rate-limit "
                "check karein."
            )

            st.session_state.stock_results = None

        else:

            result_df = pd.DataFrame(
                results
            )

            sort_columns = [
                "score",
                "bull_score",
                "volume_ratio",
            ]

            result_df = (
                result_df
                .sort_values(
                    sort_columns,
                    ascending=False,
                    na_position="last",
                )
                .reset_index(
                    drop=True
                )
            )

            st.session_state.stock_results = (
                result_df
            )

            st.session_state.last_stock_scan = (
                datetime.now(IST)
            )

    # ========================================================
    # RESULTS
    # ========================================================

    result_df = (
        st.session_state.stock_results
    )

    if result_df is None:

        st.info(
            "🚀 Scan Stocks button dabakar "
            "scanner start karein."
        )

    else:

        if (
            st.session_state.last_stock_scan
        ):

            st.caption(
                "Last scan: "
                + st.session_state
                .last_stock_scan
                .strftime(
                    "%d-%m-%Y %H:%M:%S IST"
                )
            )

        # ----------------------------------------------------
        # TOP SETUPS
        # ----------------------------------------------------

        st.subheader(
            "🏆 Top Current Setups"
        )

        top = result_df[
            (
                result_df[
                    "bull_score"
                ]
                >= min_buy_score
            )
            &
            (
                result_df[
                    "score"
                ]
                >= 12
            )
        ].head(10)

        if top.empty:

            st.warning(
                "Current scan me selected "
                "bullish threshold + minimum edge "
                "cross karne wala setup nahi mila."
            )

        else:

            display_columns = [
                "symbol",
                "price",
                "signal",
                "bull_score",
                "bear_score",
                "score",
                "day_change",
                "roc5",
                "roc10",
                "rsi",
                "volume_ratio",
            ]

            available = [
                col
                for col in display_columns
                if col in top.columns
            ]

            st.dataframe(
                top[
                    available
                ].round(2),
                use_container_width=True,
                hide_index=True,
            )

        # ----------------------------------------------------
        # TOP 5 DETAILS
        # ----------------------------------------------------

        st.subheader(
            "🔎 Top 5 Detailed Analysis"
        )

        for rank, (_, row) in enumerate(
            top.head(5).iterrows(),
            start=1,
        ):

            with st.expander(
                f"#{rank} "
                f"{row['symbol']} — "
                f"{row['signal']} — "
                f"Bull "
                f"{row['bull_score']:.0f}%"
            ):

                d1, d2, d3, d4, d5 = (
                    st.columns(5)
                )

                d1.metric(
                    "Price",
                    fmt(
                        row["price"]
                    ),
                )

                d2.metric(
                    "Bull",
                    f'{row["bull_score"]:.0f}%',
                )

                d3.metric(
                    "Bear",
                    f'{row["bear_score"]:.0f}%',
                )

                d4.metric(
                    "Edge",
                    f'{row["score"]:.0f}%',
                )

                d5.metric(
                    "RSI",
                    fmt(
                        row["rsi"]
                    ),
                )

                st.write(
                    "### 🟢 Bullish Confirmation"
                )

                reasons = row.get(
                    "reasons",
                    [],
                )

                if reasons:

                    for reason in reasons:

                        st.success(
                            "✓ "
                            + str(reason)
                        )

                else:

                    st.info(
                        "Strong bullish "
                        "confirmation nahi."
                    )

                warnings = row.get(
                    "warnings",
                    [],
                )

                if warnings:

                    st.write(
                        "### ⚠️ Bearish / "
                        "Risk Factors"
                    )

                    for warning in warnings:

                        st.warning(
                            str(warning)
                        )

                # ------------------------------------------------
                # FUNDAMENTALS
                # ------------------------------------------------

                st.write(
                    "### 🏢 Fundamental Snapshot"
                )

                fundamentals = (
                    get_fundamentals(
                        row["ticker"]
                    )
                )

                market_cap = (
                    fundamentals[
                        "market_cap"
                    ]
                )

                cap_label = (
                    get_cap_label(
                        market_cap
                    )
                )

                f1, f2, f3, f4 = (
                    st.columns(4)
                )

                f1.metric(
                    "Market Cap",
                    (
                        f"₹"
                        f"{market_cap / 1e7:,.0f}"
                        f" Cr"
                        if finite(
                            market_cap
                        )
                        else "-"
                    ),
                )

                f2.metric(
                    "ROE",
                    pct(
                        fundamentals[
                            "roe"
                        ]
                    ),
                )

                f3.metric(
                    "Debt/Equity",
                    fmt(
                        fundamentals[
                            "debt_equity"
                        ]
                    ),
                )

                f4.metric(
                    "Category",
                    cap_label,
                )

                f5, f6, f7 = (
                    st.columns(3)
                )

                f5.metric(
                    "Revenue Growth",
                    pct(
                        fundamentals[
                            "revenue_growth"
                        ]
                    ),
                )

                f6.metric(
                    "Earnings Growth",
                    pct(
                        fundamentals[
                            "earnings_growth"
                        ]
                    ),
                )

                f7.metric(
                    "Profit Margin",
                    pct(
                        fundamentals[
                            "profit_margin"
                        ]
                    ),
                )

        # ----------------------------------------------------
        # COMPLETE RESULTS
        # ----------------------------------------------------

        st.subheader(
            "📋 Complete Scan Results"
        )

        complete_columns = [
            "symbol",
            "price",
            "signal",
            "bull_score",
            "bear_score",
            "score",
            "day_change",
            "roc5",
            "roc10",
            "rsi",
            "volume_ratio",
        ]

        available = [
            col
            for col in complete_columns
            if col in result_df.columns
        ]

        st.dataframe(
            result_df[
                available
            ].round(2),
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # BULLISH CHART
        # ----------------------------------------------------

        st.subheader(
            "📊 Strongest Bullish Setups"
        )

        chart_data = (
            result_df
            .sort_values(
                "bull_score",
                ascending=False,
            )
            .head(10)
            .set_index(
                "symbol"
            )[
                [
                    "bull_score"
                ]
            ]
        )

        if not chart_data.empty:

            st.bar_chart(
                chart_data
            )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.warning(
    """
⚠️ IMPORTANT

Ye app probability/confirmation based analysis hai.
Guaranteed profit nahi hai.

V4 improvements:

• Forming 5-minute candle ko market state ke according handle kiya gaya hai.
• Yahoo Finance MultiIndex data handling safer hai.
• Duplicate OHLCV columns handle kiye gaye hain.
• Invalid OHLC rows remove kiye jaate hain.
• Intraday EMA200 aur Daily EMA200 separate hain.
• Bullish aur bearish scores independently calculate hote hain.
• Opposite-side confirmation strong hone par STRONG signal downgrade hota hai.
• Session VWAP use hota hai.
• Volume current candle ke against previous 20 candles se compare hota hai.
• Breakout previous candles ke against calculate hota hai.
• NIFTY option selection spread + liquidity + OI + distance based hai.
• Wide-spread options reject kiye jaate hain.
• Max Pain duplicate strikes ko aggregate karke calculate hota hai.
• OI support/resistance nearby strikes par calculate hota hai.
• Option chain failure graceful handling ke saath hai.
• Fundamental data cached hai.
• Fundamental percentage values double-multiply hone se protected hain.
• Stock scanner individual stock failure par poora scan crash nahi karta.
• Empty/missing data se fake signal generate nahi hota.
• Streamlit refresh clear calls safely handle kiye gaye hain.
• Trade plan sirf STRONG BUY / STRONG SELL state mein generate hota hai.

Yahoo Finance broker-grade execution feed nahi hai.

Actual trade se pehle broker ka live price,
bid/ask, liquidity, slippage, position size,
market conditions aur risk verify karein.
"""
)

st.caption(
    "Last app refresh: "
    + datetime.now(
        IST
    ).strftime(
        "%d-%m-%Y %H:%M:%S IST"
    )
)

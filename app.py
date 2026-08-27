# ============================================================
# NIFTY + STOCK SIGNAL PRO V4
# Robust / safer / completed-candle / liquidity-aware
# Streamlit + Yahoo Finance
#
# IMPORTANT:
# - This is an analysis/scanner tool, NOT guaranteed-profit software.
# - Yahoo Finance is not a broker-grade execution feed.
# - Always verify live broker price, bid/ask, liquidity and slippage.
# ============================================================

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo
import logging
import math
import time

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

INTRADAY_INTERVAL = "5m"
INTRADAY_PERIOD = "5d"
DAILY_PERIOD = "2y"

DATA_TTL = 45
DAILY_TTL = 300
OPTION_TTL = 45
FUNDAMENTAL_TTL = 3600

MIN_INTRADAY_CANDLES = 100
MIN_DAILY_CANDLES = 220

MAX_WORKERS = 5

MARKET_OPEN = dt_time(9, 15)
MARKET_CLOSE = dt_time(15, 30)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


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
# SAFE HELPERS
# ============================================================

def safe_float(value, default=np.nan):
    """Convert scalar/Series/DataFrame value safely to finite float."""
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

    except (TypeError, ValueError, IndexError, OverflowError):
        return default


def finite(value):
    value = safe_float(value)
    return bool(np.isfinite(value))


def positive(value):
    value = safe_float(value)
    return bool(np.isfinite(value) and value > 0)


def fmt(value, digits=2):
    value = safe_float(value)

    if not np.isfinite(value):
        return "-"

    return f"{value:,.{digits}f}"


def pct(value, digits=1):
    value = safe_float(value)

    if not np.isfinite(value):
        return "-"

    return f"{value:.{digits}f}%"


def clamp(value, low, high):
    value = safe_float(value)

    if not finite(value):
        return np.nan

    return max(low, min(high, value))


# ============================================================
# MARKET STATUS
# ============================================================

def is_weekday():
    return datetime.now(IST).weekday() < 5


def is_market_open_now():
    now = datetime.now(IST)

    return (
        now.weekday() < 5
        and MARKET_OPEN <= now.time() <= MARKET_CLOSE
    )


def market_status():
    now = datetime.now(IST)

    if now.weekday() >= 5:
        return "🔴 NSE closed — Weekend"

    if MARKET_OPEN <= now.time() <= MARKET_CLOSE:
        return "🟢 NSE market hours"

    return "🟡 NSE market closed"


# ============================================================
# OHLCV CLEANING
# ============================================================

def clean_ohlcv(data):
    if data is None or data.empty:
        return pd.DataFrame()

    try:
        df = data.copy()

        # ----------------------------------------------------
        # Handle Yahoo MultiIndex safely
        # ----------------------------------------------------
        if isinstance(df.columns, pd.MultiIndex):
            levels = [
                [str(x) for x in df.columns.get_level_values(i)]
                for i in range(df.columns.nlevels)
            ]

            required = {
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            }

            selected = {}

            for field in required:
                found = None

                for level in levels:
                    if field in level:
                        found = field
                        break

                if found is not None:
                    selected[field] = found

            if len(selected) < 5:
                # Try flattening as fallback.
                flat = []

                for col in df.columns:
                    parts = [str(x) for x in col]
                    flat.append("_".join(parts))

                df.columns = flat

                rename_map = {}

                for wanted in required:
                    for col in df.columns:
                        if col == wanted or col.endswith(f"_{wanted}"):
                            rename_map[col] = wanted
                            break

                df = df.rename(columns=rename_map)

            else:
                # Pick exact OHLCV fields from whichever MultiIndex
                # level contains them.
                new_df = pd.DataFrame(index=df.index)

                for field in required:
                    positions = []

                    for i, col in enumerate(df.columns):
                        if field in [str(x) for x in col]:
                            positions.append(i)

                    if positions:
                        new_df[field] = df.iloc[:, positions[0]]

                df = new_df

        # ----------------------------------------------------
        # Normal single-level columns
        # ----------------------------------------------------
        df.columns = [str(c) for c in df.columns]

        required = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]

        if not all(col in df.columns for col in required):
            return pd.DataFrame()

        df = df[required].copy()

        # ----------------------------------------------------
        # Numeric conversion
        # ----------------------------------------------------
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

        # ----------------------------------------------------
        # OHLC sanity
        # ----------------------------------------------------
        df = df[
            (df["High"] >= df["Low"])
            & (df["High"] >= df["Open"])
            & (df["High"] >= df["Close"])
            & (df["Low"] <= df["Open"])
            & (df["Low"] <= df["Close"])
        ]

        if isinstance(df.index, pd.DatetimeIndex):
            try:
                if df.index.tz is None:
                    df.index = df.index.tz_localize(
                        IST,
                        ambiguous="NaT",
                        nonexistent="shift_forward",
                    )
                else:
                    df.index = df.index.tz_convert(IST)
            except Exception:
                pass

            df = df[
                ~df.index.duplicated(keep="last")
            ].sort_index()

        return df

    except Exception as exc:
        logging.warning(
            "OHLCV cleaning failed: %s",
            exc,
        )
        return pd.DataFrame()


# ============================================================
# SYMBOL EXTRACTION
# ============================================================

def extract_symbol(raw, symbol):
    if raw is None or raw.empty:
        return pd.DataFrame()

    try:
        if isinstance(raw.columns, pd.MultiIndex):

            # Case 1:
            # ticker is first level
            if symbol in raw.columns.get_level_values(0):
                return clean_ohlcv(
                    raw.xs(
                        symbol,
                        axis=1,
                        level=0,
                        drop_level=True,
                    )
                )

            # Case 2:
            # ticker is second level
            if symbol in raw.columns.get_level_values(1):
                return clean_ohlcv(
                    raw.xs(
                        symbol,
                        axis=1,
                        level=1,
                        drop_level=True,
                    )
                )

            # Case 3: search each column tuple
            matching = []

            for col in raw.columns:
                if symbol in [str(x) for x in col]:
                    matching.append(col)

            if matching:
                return clean_ohlcv(
                    raw.loc[:, matching]
                )

            return pd.DataFrame()

        return clean_ohlcv(raw)

    except Exception as exc:
        logging.warning(
            "Symbol extraction failed for %s: %s",
            symbol,
            exc,
        )
        return pd.DataFrame()


# ============================================================
# YAHOO DOWNLOAD
# ============================================================

def _download_batch(
    symbols,
    period,
    interval,
):
    symbols = tuple(
        dict.fromkeys(
            str(s) for s in symbols if s
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
            prepost=False,
        )

    except Exception as exc:
        logging.warning(
            "Yahoo batch download failed: %s",
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
        try:
            df = extract_symbol(
                raw,
                symbol,
            )

            if len(df) >= minimum:
                result[symbol] = df

        except Exception as exc:
            logging.warning(
                "Data processing failed for %s: %s",
                symbol,
                exc,
            )

    return result


@st.cache_data(
    ttl=DATA_TTL,
    show_spinner=False,
)
def download_intraday_batch(symbols):
    return _download_batch(
        tuple(symbols),
        INTRADAY_PERIOD,
        INTRADAY_INTERVAL,
    )


@st.cache_data(
    ttl=DAILY_TTL,
    show_spinner=False,
)
def download_daily_batch(symbols):
    return _download_batch(
        tuple(symbols),
        DAILY_PERIOD,
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
# COMPLETED CANDLE
# ============================================================

def get_completed_intraday_df(df):
    """
    Return only completed 5-minute candles.

    Yahoo may return the currently forming candle during
    market hours. That candle is removed.

    After market close we retain the latest candle because
    it should already represent the completed session.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    if len(out) < 3:
        return pd.DataFrame()

    if not isinstance(out.index, pd.DatetimeIndex):
        return out.iloc[:-1].copy()

    try:
        out = out.sort_index()

        now = datetime.now(IST)

        last_ts = out.index[-1]

        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(
                tzinfo=IST
            )
        else:
            last_ts = last_ts.astimezone(
                IST
            )

        # During market hours the last 5-minute bucket
        # is potentially still forming.
        if is_market_open_now():

            bucket_minute = (
                now.minute // 5
            ) * 5

            current_bucket = now.replace(
                minute=bucket_minute,
                second=0,
                microsecond=0,
            )

            if last_ts >= current_bucket:
                out = out.iloc[:-1].copy()

        # Extra protection:
        # don't use future timestamps.
        out = out[
            out.index <= now
        ].copy()

        return out

    except Exception as exc:
        logging.warning(
            "Completed candle handling failed: %s",
            exc,
        )

        return out.iloc[:-1].copy()


# ============================================================
# RSI
# ============================================================

def rsi_wilder(close, period=14):
    close = pd.to_numeric(
        close,
        errors="coerce",
    )

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

    rsi = pd.Series(
        np.nan,
        index=close.index,
        dtype=float,
    )

    normal = (
        avg_loss > 0
    ) & avg_gain.notna()

    rs = pd.Series(
        np.nan,
        index=close.index,
        dtype=float,
    )

    rs.loc[normal] = (
        avg_gain.loc[normal]
        / avg_loss.loc[normal]
    )

    rsi.loc[normal] = (
        100
        - (
            100
            / (1 + rs.loc[normal])
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
# ATR
# ============================================================

def atr_wilder(
    high,
    low,
    close,
    period=14,
):
    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


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
    # EMAs
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

    df["ATR"] = atr_wilder(
        high,
        low,
        close,
        14,
    )

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
        session_key = pd.Series(
            df.index.date,
            index=df.index,
        )

        cumulative_pv = (
            pv.groupby(
                session_key
            ).cumsum()
        )

        cumulative_volume = (
            volume.groupby(
                session_key
            ).cumsum()
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
    # RELATIVE VOLUME
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
        volume / df["VOL_AVG20"],
        np.nan,
    )

    # --------------------------------------------------------
    # BREAKOUT LEVELS
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
        close
        .pct_change(5)
        * 100
    )

    df["ROC10"] = (
        close
        .pct_change(10)
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
            close - df["Open"]
        ).abs()
        / candle_range
    )

    df["CLOSE_LOCATION"] = (
        (
            close - low
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

MAX_SCORE = sum(BULL_WEIGHTS.values())


# ============================================================
# SIGNAL CLASSIFICATION
# ============================================================

def classify_signal(
    bull_pct,
    bear_pct,
):
    bull_pct = safe_float(bull_pct, 0)
    bear_pct = safe_float(bear_pct, 0)

    edge = bull_pct - bear_pct
    difference = abs(edge)

    if (
        bull_pct >= 72
        and edge >= 18
    ):
        return "STRONG BUY"

    if (
        bull_pct >= 62
        and edge >= 12
    ):
        return "BUY WATCH"

    if (
        bear_pct >= 72
        and edge <= -18
    ):
        return "STRONG SELL"

    if (
        bear_pct >= 62
        and edge <= -12
    ):
        return "SELL WATCH"

    if (
        bull_pct >= 52
        and edge > 0
    ):
        return "WATCH BUY"

    if (
        bear_pct >= 52
        and edge < 0
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

    try:
        # ----------------------------------------------------
        # Completed candles FIRST
        # ----------------------------------------------------

        intraday = get_completed_intraday_df(
            intraday
        )

        if len(intraday) < MIN_INTRADAY_CANDLES:
            return None

        if len(daily) < MIN_DAILY_CANDLES:
            return None

        # ----------------------------------------------------
        # Indicators
        # ----------------------------------------------------

        intraday = calculate_intraday_indicators(
            intraday
        )

        daily = calculate_daily_indicators(
            daily
        )

        if intraday.empty or daily.empty:
            return None

        last = intraday.iloc[-1]
        previous = intraday.iloc[-2]
        daily_last = daily.iloc[-1]

        price = safe_float(
            last["Close"]
        )

        if not positive(price):
            return None

        # ----------------------------------------------------
        # Indicator values
        # ----------------------------------------------------

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
            daily_last["EMA200_DAILY"]
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

        # ----------------------------------------------------
        # Bullish conditions
        # ----------------------------------------------------

        bullish = {
            "Price above VWAP": (
                finite(values["VWAP"])
                and price > values["VWAP"]
            ),

            "EMA9 > EMA20": (
                finite(values["EMA9"])
                and finite(values["EMA20"])
                and values["EMA9"]
                > values["EMA20"]
            ),

            "EMA20 > EMA50": (
                finite(values["EMA20"])
                and finite(values["EMA50"])
                and values["EMA20"]
                > values["EMA50"]
            ),

            "EMA20 rising": (
                finite(values["EMA20"])
                and finite(previous_ema20)
                and values["EMA20"]
                > previous_ema20
            ),

            "EMA50 rising": (
                finite(values["EMA50"])
                and finite(previous_ema50)
                and values["EMA50"]
                > previous_ema50
            ),

            "Above daily EMA200": (
                finite(daily_ema200)
                and price > daily_ema200
            ),

            "RSI 52-72": (
                finite(values["RSI"])
                and 52 <= values["RSI"] <= 72
            ),

            "ROC5 positive": (
                finite(values["ROC5"])
                and values["ROC5"] > 0
            ),

            "ROC10 positive": (
                finite(values["ROC10"])
                and values["ROC10"] > 0
            ),

            "Volume >= 1.2x": (
                finite(values["VOL_RATIO"])
                and values["VOL_RATIO"] >= 1.20
            ),

            "10-candle breakout": (
                finite(values["HIGH10"])
                and price > values["HIGH10"]
            ),

            "Strong bullish candle": (
                finite(open_price)
                and price > open_price
                and finite(values["BODY_PCT"])
                and finite(values["CLOSE_LOCATION"])
                and values["BODY_PCT"] >= 0.45
                and values["CLOSE_LOCATION"] >= 0.65
            ),
        }

        # ----------------------------------------------------
        # Bearish conditions
        # ----------------------------------------------------

        bearish = {
            "Price below VWAP": (
                finite(values["VWAP"])
                and price < values["VWAP"]
            ),

            "EMA9 < EMA20": (
                finite(values["EMA9"])
                and finite(values["EMA20"])
                and values["EMA9"]
                < values["EMA20"]
            ),

            "EMA20 < EMA50": (
                finite(values["EMA20"])
                and finite(values["EMA50"])
                and values["EMA20"]
                < values["EMA50"]
            ),

            "EMA20 falling": (
                finite(values["EMA20"])
                and finite(previous_ema20)
                and values["EMA20"]
                < previous_ema20
            ),

            "EMA50 falling": (
                finite(values["EMA50"])
                and finite(previous_ema50)
                and values["EMA50"]
                < previous_ema50
            ),

            "Below daily EMA200": (
                finite(daily_ema200)
                and price < daily_ema200
            ),

            "RSI 28-48": (
                finite(values["RSI"])
                and 28 <= values["RSI"] <= 48
            ),

            "ROC5 negative": (
                finite(values["ROC5"])
                and values["ROC5"] < 0
            ),

            "ROC10 negative": (
                finite(values["ROC10"])
                and values["ROC10"] < 0
            ),

            "Volume >= 1.2x": (
                finite(values["VOL_RATIO"])
                and values["VOL_RATIO"] >= 1.20
            ),

            "10-candle breakdown": (
                finite(values["LOW10"])
                and price < values["LOW10"]
            ),

            "Strong bearish candle": (
                finite(open_price)
                and price < open_price
                and finite(values["BODY_PCT"])
                and finite(values["CLOSE_LOCATION"])
                and values["BODY_PCT"] >= 0.45
                and values["CLOSE_LOCATION"] <= 0.35
            ),
        }

        # ----------------------------------------------------
        # Scores
        #
        # IMPORTANT:
        # Score is based only on conditions whose required
        # indicators exist. Missing data is NOT treated as
        # bearish or bullish.
        # ----------------------------------------------------

        bull_available = sum(
            BULL_WEIGHTS[key]
            for key, condition in bullish.items()
            if (
                condition
                or key not in {
                    "Price above VWAP",
                    "EMA9 > EMA20",
                    "EMA20 > EMA50",
                    "EMA20 rising",
                    "EMA50 rising",
                    "Above daily EMA200",
                    "RSI 52-72",
                    "ROC5 positive",
                    "ROC10 positive",
                    "Volume >= 1.2x",
                    "10-candle breakout",
                    "Strong bullish candle",
                }
            )
        )

        bear_available = sum(
            BEAR_WEIGHTS[key]
            for key, condition in bearish.items()
            if (
                condition
                or key not in {
                    "Price below VWAP",
                    "EMA9 < EMA20",
                    "EMA20 < EMA50",
                    "EMA20 falling",
                    "EMA50 falling",
                    "Below daily EMA200",
                    "RSI 28-48",
                    "ROC5 negative",
                    "ROC10 negative",
                    "Volume >= 1.2x",
                    "10-candle breakdown",
                    "Strong bearish candle",
                }
            )
        )

        # In this model all conditions have a defined value.
        # Keep the explicit denominator stable for ranking.
        bull_score = sum(
            BULL_WEIGHTS[key]
            for key, condition in bullish.items()
            if condition
        )

        bear_score = sum(
            BEAR_WEIGHTS[key]
            for key, condition in bearish.items()
            if condition
        )

        bull_pct = (
            bull_score / MAX_SCORE
        ) * 100

        bear_pct = (
            bear_score / MAX_SCORE
        ) * 100

        score_difference = (
            bull_pct - bear_pct
        )

        signal = classify_signal(
            bull_pct,
            bear_pct,
        )

        symbol_name = (
            symbol.replace(".NS", "")
            if symbol
            else "NIFTY"
        )

        return {
            "symbol": symbol_name,
            "ticker": symbol if symbol else NIFTY,

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
            "ema200_intraday": values["EMA200"],

            "daily_ema200": daily_ema200,

            "atr": values["ATR"],

            "roc5": values["ROC5"],
            "roc10": values["ROC10"],

            "volume_ratio": values["VOL_RATIO"],

            "bullish": bullish,
            "bearish": bearish,

            "reasons": [
                key
                for key, value in bullish.items()
                if value
            ],

            "warnings": [
                key
                for key, value in bearish.items()
                if value
            ],

            "candle_time": intraday.index[-1],

            "chart_df": intraday,
        }

    except Exception as exc:
        logging.warning(
            "Frame analysis failed for %s: %s",
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
    if (
        daily_df is None
        or daily_df.empty
    ):
        return np.nan

    current_price = safe_float(
        current_price
    )

    if not positive(current_price):
        return np.nan

    df = daily_df.copy()

    try:
        if isinstance(
            df.index,
            pd.DatetimeIndex,
        ):
            dates = [
                ts.date()
                for ts in df.index
            ]

            today = datetime.now(
                IST
            ).date()

            if today in dates:
                idx = dates.index(today)

                if idx > 0:
                    previous_close = safe_float(
                        df.iloc[idx - 1]["Close"]
                    )

                    if positive(previous_close):
                        return (
                            (
                                current_price
                                / previous_close
                            ) - 1
                        ) * 100

            # Latest available daily candle may be
            # today's close after market close.
            if len(df) >= 2:
                previous_close = safe_float(
                    df.iloc[-2]["Close"]
                )

                if positive(previous_close):
                    latest_date = (
                        df.index[-1].date()
                        if isinstance(
                            df.index[-1],
                            pd.Timestamp,
                        )
                        else None
                    )

                    if (
                        latest_date != today
                        or not is_market_open_now()
                    ):
                        return (
                            (
                                current_price
                                / previous_close
                            ) - 1
                        ) * 100

        previous_close = safe_float(
            df.iloc[-1]["Close"]
        )

        if positive(previous_close):
            return (
                (
                    current_price
                    / previous_close
                ) - 1
            ) * 100

    except Exception as exc:
        logging.warning(
            "Day change calculation failed: %s",
            exc,
        )

    return np.nan


# ============================================================
# STOCK ANALYSIS
# ============================================================

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
            "Stock analysis failed for %s: %s",
            symbol,
            exc,
        )
        return None


# ============================================================
# ATM
# ============================================================

def get_atm(price):
    price = safe_float(price)

    if not positive(price):
        return np.nan

    return int(
        math.floor(
            (price / 50) + 0.5
        ) * 50
    )


# ============================================================
# OPTIONS CLEANING
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
            out[col] = 0

        out[col] = (
            out[col]
            .fillna(0)
            .clip(lower=0)
        )

    return out


# ============================================================
# OPTION CHAIN
# ============================================================

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

        expirations = sorted(
            set(expirations)
        )

        expiry = expirations[0]

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
            expirations,
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

    try:
        work = df.copy()

        required = [
            "strike",
            "lastPrice",
            "bid",
            "ask",
            "volume",
            "openInterest",
        ]

        for col in required:
            if col not in work.columns:
                work[col] = 0

            work[col] = pd.to_numeric(
                work[col],
                errors="coerce",
            ).fillna(0)

        work = work[
            work["strike"] > 0
        ].copy()

        if work.empty:
            return None

        work["distance"] = (
            work["strike"] - atm
        ).abs()

        # Maximum 300 points from ATM.
        work = work[
            work["distance"] <= 300
        ].copy()

        if work.empty:
            return None

        # ----------------------------------------------------
        # Direction-specific strike window
        # ----------------------------------------------------

        if direction == "CE":
            work = work[
                (work["strike"] >= atm - 50)
                & (work["strike"] <= atm + 100)
            ].copy()

        elif direction == "PE":
            work = work[
                (work["strike"] >= atm - 100)
                & (work["strike"] <= atm + 50)
            ].copy()

        else:
            return None

        if work.empty:
            return None

        bid = work["bid"]
        ask = work["ask"]
        last_price = work["lastPrice"]

        valid_market = (
            (bid > 0)
            & (ask > 0)
            & (ask >= bid)
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
            & np.isfinite(
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

        # Reject extremely poor markets.
        work = work[
            work["spread_pct"] <= 12
        ].copy()

        if work.empty:
            return None

        volume = (
            work["volume"]
            .fillna(0)
            .clip(lower=0)
        )

        oi = (
            work["openInterest"]
            .fillna(0)
            .clip(lower=0)
        )

        # ----------------------------------------------------
        # Liquidity score
        # ----------------------------------------------------

        work["liquidity_score"] = (
            np.log1p(volume) * 2.0
            + np.log1p(oi) * 1.0
        )

        work["distance_penalty"] = (
            work["distance"] / 50
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

        # Prefer reasonable OI/volume.
        work = work.sort_values(
            [
                "selection_score",
                "openInterest",
                "volume",
            ],
            ascending=False,
        )

        if work.empty:
            return None

        return work.iloc[0]

    except Exception as exc:
        logging.warning(
            "Option selection failed: %s",
            exc,
        )
        return None


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

    try:
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
            put_volume / call_volume
            if call_volume > 0
            else np.nan
        )

        return (
            oi_pcr,
            volume_pcr,
        )

    except Exception:
        return (
            np.nan,
            np.nan,
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

    try:
        calls = calls.copy()
        puts = puts.copy()

        for df in (calls, puts):
            df["strike"] = pd.to_numeric(
                df["strike"],
                errors="coerce",
            )

            df["openInterest"] = (
                pd.to_numeric(
                    df["openInterest"],
                    errors="coerce",
                )
                .fillna(0)
                .clip(lower=0)
            )

        calls = calls.dropna(
            subset=["strike"]
        )

        puts = puts.dropna(
            subset=["strike"]
        )

        if calls.empty or puts.empty:
            return np.nan

        strikes = sorted(
            set(
                calls["strike"].astype(float)
            ).union(
                set(
                    puts["strike"].astype(float)
                )
            )
        )

        if not strikes:
            return np.nan

        call_strikes = (
            calls["strike"].to_numpy()
        )

        call_oi = (
            calls["openInterest"].to_numpy()
        )

        put_strikes = (
            puts["strike"].to_numpy()
        )

        put_oi = (
            puts["openInterest"].to_numpy()
        )

        lowest_loss = np.inf
        best_strike = np.nan

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
                lowest_loss = total_loss
                best_strike = settlement

        return best_strike

    except Exception as exc:
        logging.warning(
            "Max pain failed: %s",
            exc,
        )
        return np.nan


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

    price = safe_float(price)

    if not positive(price):
        return (
            np.nan,
            np.nan,
        )

    try:
        calls_near = calls[
            (calls["strike"] >= price)
            & (calls["strike"] <= price + 500)
            & (calls["openInterest"] > 0)
        ].copy()

        puts_near = puts[
            (puts["strike"] <= price)
            & (puts["strike"] >= price - 500)
            & (puts["openInterest"] > 0)
        ].copy()

        resistance = np.nan
        support = np.nan

        if not calls_near.empty:
            distance = (
                calls_near["strike"]
                - price
            ).clip(lower=0)

            calls_near["level_score"] = (
                calls_near["openInterest"]
                / (1 + distance / 50)
            )

            resistance = safe_float(
                calls_near
                .sort_values(
                    "level_score",
                    ascending=False,
                )
                .iloc[0]["strike"]
            )

        if not puts_near.empty:
            distance = (
                price
                - puts_near["strike"]
            ).clip(lower=0)

            puts_near["level_score"] = (
                puts_near["openInterest"]
                / (1 + distance / 50)
            )

            support = safe_float(
                puts_near
                .sort_values(
                    "level_score",
                    ascending=False,
                )
                .iloc[0]["strike"]
            )

        return (
            resistance,
            support,
        )

    except Exception as exc:
        logging.warning(
            "OI levels failed: %s",
            exc,
        )

        return (
            np.nan,
            np.nan,
        )


# ============================================================
# TRADE PLAN
# ============================================================

def build_trade_plan(
    action,
    option_row,
    nifty_price,
    atr,
):
    if option_row is None:
        return None

    nifty_price = safe_float(
        nifty_price
    )

    atr = safe_float(
        atr
    )

    if (
        not positive(nifty_price)
        or not positive(atr)
    ):
        return None

    if action not in {
        "BUY CE",
        "BUY PE",
    }:
        return None

    try:
        bid = safe_float(
            option_row.get("bid")
        )

        ask = safe_float(
            option_row.get("ask")
        )

        last_price = safe_float(
            option_row.get("lastPrice")
        )

        # ----------------------------------------------------
        # Entry
        # ----------------------------------------------------

        if (
            positive(bid)
            and positive(ask)
            and ask >= bid
        ):
            entry = (
                bid + ask
            ) / 2

        elif positive(last_price):
            entry = last_price

        else:
            return None

        if not positive(entry):
            return None

        # ----------------------------------------------------
        # Premium risk
        #
        # 18% SL
        # R:R:
        # T1 = 1.5R
        # T2 = 2.5R
        # T3 = 4R
        # ----------------------------------------------------

        stop_loss = entry * 0.82

        risk = (
            entry
            - stop_loss
        )

        if risk <= 0:
            return None

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

        # ----------------------------------------------------
        # Underlying invalidation
        # ----------------------------------------------------

        if action == "BUY CE":
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
            "stop_loss": stop_loss,
            "target1": target1,
            "target2": target2,
            "target3": target3,
            "nifty_stop": nifty_stop,
        }

    except Exception as exc:
        logging.warning(
            "Trade plan failed: %s",
            exc,
        )
        return None


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
        ticker = yf.Ticker(
            symbol
        )

        info = ticker.info or {}

        market_cap = safe_float(
            info.get("marketCap")
        )

        roe = safe_float(
            info.get("returnOnEquity")
        )

        debt_equity = safe_float(
            info.get("debtToEquity")
        )

        profit_margin = safe_float(
            info.get("profitMargins")
        )

        revenue_growth = safe_float(
            info.get("revenueGrowth")
        )

        earnings_growth = safe_float(
            info.get("earningsGrowth")
        )

        if finite(roe):
            roe *= 100

        if finite(profit_margin):
            profit_margin *= 100

        if finite(revenue_growth):
            revenue_growth *= 100

        if finite(earnings_growth):
            earnings_growth *= 100

        return {
            "market_cap": market_cap,
            "roe": roe,
            "debt_equity": debt_equity,
            "profit_margin": profit_margin,
            "revenue_growth": revenue_growth,
            "earnings_growth": earnings_growth,
        }

    except Exception as exc:
        logging.warning(
            "Fundamental data failed for %s: %s",
            symbol,
            exc,
        )

        return empty


# ============================================================
# CAP LABEL
# ============================================================

def get_cap_label(market_cap):
    market_cap = safe_float(
        market_cap
    )

    if not finite(market_cap):
        return "-"

    # Yahoo marketCap for Indian stocks is normally INR.
    if market_cap >= 2e12:
        return "Large Cap"

    if market_cap >= 5e11:
        return "Mid Cap"

    return "Small Cap"


# ============================================================
# CACHE CLEAR
# ============================================================

def clear_market_cache():
    functions = [
        download_intraday_batch,
        download_daily_batch,
        get_nifty_intraday,
        get_nifty_daily,
        get_nifty_options,
        get_fundamentals,
    ]

    for func in functions:
        try:
            func.clear()
        except Exception:
            pass


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

    max_pain = calculate_max_pain(
        calls,
        puts,
    )

    resistance, support = (
        get_oi_levels(
            calls,
            puts,
            price,
        )
    )

    signal = market["signal"]

    # ========================================================
    # TOP METRICS
    # ========================================================

    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric(
        "NIFTY",
        fmt(price),
    )

    m2.metric(
        "ATM",
        str(atm) if finite(atm) else "-",
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
        signal,
    )

    candle_time = market.get(
        "candle_time"
    )

    if candle_time is not None:
        st.caption(
            "Signal last completed 5-minute candle "
            f"({candle_time.strftime('%d-%m-%Y %H:%M IST')}) "
            "par based hai."
        )

    # ========================================================
    # ACTION
    # ========================================================

    st.subheader(
        "🚨 FINAL SIGNAL"
    )

    if signal == "STRONG BUY":

        st.success(
            "🟢 STRONG BUY CE — "
            "Bullish confirmation strong hai."
        )

    elif signal == "STRONG SELL":

        st.error(
            "🔴 STRONG SELL — "
            "Bearish confirmation strong hai."
        )

    elif signal == "BUY WATCH":

        st.info(
            "🟢 BUY WATCH — "
            "Bullish setup hai; final confirmation wait karein."
        )

    elif signal == "SELL WATCH":

        st.info(
            "🔴 SELL WATCH — "
            "Bearish setup hai; final confirmation wait karein."
        )

    elif signal == "WATCH BUY":

        st.info(
            "🟢 WATCH BUY — "
            "Bullish bias hai, lekin confirmation incomplete hai."
        )

    elif signal == "WATCH SELL":

        st.info(
            "🔴 WATCH SELL — "
            "Bearish bias hai, lekin confirmation incomplete hai."
        )

    else:

        st.warning(
            "🟡 WAIT — "
            "Strong directional edge nahi mili."
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

    a1, a2, a3, a4, a5, a6 = st.columns(6)

    a1.metric(
        "EMA20",
        fmt(market["ema20"]),
    )

    a2.metric(
        "EMA50",
        fmt(market["ema50"]),
    )

    a3.metric(
        "Intraday EMA200",
        fmt(
            market["ema200_intraday"]
        ),
    )

    a4.metric(
        "Daily EMA200",
        fmt(
            market["daily_ema200"]
        ),
    )

    a5.metric(
        "VWAP",
        fmt(market["vwap"]),
    )

    a6.metric(
        "RSI",
        fmt(market["rsi"]),
    )

    # ========================================================
    # MOMENTUM
    # ========================================================

    st.subheader(
        "⚡ Momentum / Volume"
    )

    q1, q2, q3 = st.columns(3)

    q1.metric(
        "ROC5",
        pct(market["roc5"]),
    )

    q2.metric(
        "ROC10",
        pct(market["roc10"]),
    )

    volume_ratio = safe_float(
        market["volume_ratio"]
    )

    q3.metric(
        "Volume Ratio",
        (
            f"{volume_ratio:.2f}x"
            if finite(volume_ratio)
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
            "⚠️ Yahoo Finance se option chain "
            "available nahi hui."
        )

    else:

        o1, o2, o3 = st.columns(3)

        o1.metric(
            "Expiry",
            str(expiry),
        )

        o2.metric(
            "OI PCR",
            fmt(oi_pcr),
        )

        o3.metric(
            "Volume PCR",
            fmt(volume_pcr),
        )

        o4, o5, o6 = st.columns(3)

        o4.metric(
            "Max Pain",
            fmt(max_pain, 0),
        )

        o5.metric(
            "OI Resistance",
            fmt(resistance, 0),
        )

        o6.metric(
            "OI Support",
            fmt(support, 0),
        )

        # ----------------------------------------------------
        # Selected options
        # ----------------------------------------------------

        st.subheader(
            "⚔️ Selected CE / PE"
        )

        option_rows = []

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

            option_rows.append(
                {
                    "Side": side,
                    "Strike": safe_float(
                        row.get("strike")
                    ),
                    "Last": safe_float(
                        row.get("lastPrice")
                    ),
                    "Bid": safe_float(
                        row.get("bid")
                    ),
                    "Ask": safe_float(
                        row.get("ask")
                    ),
                    "Volume": safe_float(
                        row.get("volume"),
                        0,
                    ),
                    "OI": safe_float(
                        row.get("openInterest"),
                        0,
                    ),
                    "IV %": (
                        iv * 100
                        if finite(iv)
                        else np.nan
                    ),
                }
            )

        if option_rows:

            option_df = pd.DataFrame(
                option_rows
            )

            st.dataframe(
                option_df.round(2),
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

    selected_option = None

    if action == "BUY CE":
        selected_option = ce_row

    elif action == "BUY PE":
        selected_option = pe_row

    plan = build_trade_plan(
        action,
        selected_option,
        price,
        market["atr"],
    )

    if plan is None:

        st.info(
            "Trade plan unavailable — "
            "strong signal + valid option liquidity "
            "required."
        )

    else:

        p1, p2, p3, p4 = st.columns(4)

        p1.metric(
            "Entry",
            fmt(plan["entry"]),
        )

        p2.metric(
            "SL",
            fmt(plan["stop_loss"]),
        )

        p3.metric(
            "T1",
            fmt(plan["target1"]),
        )

        p4.metric(
            "T2",
            fmt(plan["target2"]),
        )

        st.caption(
            f'T3: {fmt(plan["target3"])} | '
            f'Underlying NIFTY invalidation: '
            f'{fmt(plan["nifty_stop"])}'
        )

    # ========================================================
    # CHECKLIST
    # ========================================================

    st.subheader(
        "✅ Confirmation Checklist"
    )

    left, right = st.columns(2)

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

    chart_source = market.get(
        "chart_df",
        pd.DataFrame(),
    )

    chart_columns = [
        "Close",
        "EMA20",
        "EMA50",
        "VWAP",
    ]

    available_columns = [
        col
        for col in chart_columns
        if col in chart_source.columns
    ]

    if available_columns:

        chart_df = (
            chart_source[
                available_columns
            ]
            .tail(150)
            .copy()
        )

        st.line_chart(
            chart_df,
            height=450,
        )

    else:

        st.info(
            "Chart ke liye sufficient indicator data nahi hai."
        )


# ============================================================
# PAGE TABS
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

    c1, c2 = st.columns(2)

    with c1:

        refresh_nifty = st.button(
            "🔄 Refresh NIFTY",
            use_container_width=True,
            key="refresh_nifty_button",
        )

    with c2:

        auto_refresh = st.checkbox(
            "⏱️ Auto Refresh (45s)",
            value=False,
            key="auto_refresh_checkbox",
        )

    if refresh_nifty:

        clear_market_cache()

        st.session_state.nifty_last_refresh = (
            datetime.now(IST)
        )

        st.rerun()

    # --------------------------------------------------------
    # Auto refresh
    # --------------------------------------------------------

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

    elif auto_refresh:

        st.info(
            "Current Streamlit version native "
            "fragment auto-refresh support nahi karta. "
            "Manual refresh use karein."
        )

        render_nifty()

    else:

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

    s1, s2, s3 = st.columns(3)

    with s1:

        scan_count = st.selectbox(
            "Stocks to scan",
            [25, 50],
            index=1,
            key="scan_count_select",
        )

    with s2:

        min_buy_score = st.slider(
            "Minimum bullish score %",
            40,
            90,
            62,
            key="min_buy_score_slider",
        )

    with s3:

        run_scan = st.button(
            "🚀 SCAN STOCKS",
            use_container_width=True,
            key="scan_stocks_button",
        )

    symbols = DEFAULT_STOCKS[
        :min(
            scan_count,
            len(DEFAULT_STOCKS),
        )
    ]

    # ========================================================
    # RUN SCAN
    # ========================================================

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

                future = executor.submit(
                    worker,
                    symbol,
                )

                futures[future] = symbol

            total = len(futures)

            for done, future in enumerate(
                as_completed(futures),
                start=1,
            ):

                symbol = futures[
                    future
                ]

                try:

                    result = future.result()

                    if result is not None:
                        results.append(
                            result
                        )

                except Exception as exc:

                    logging.warning(
                        "Stock future failed for %s: %s",
                        symbol,
                        exc,
                    )

                progress.progress(
                    done / total
                    if total
                    else 1.0
                )

                status.write(
                    f"Analysing "
                    f"{done}/{total}..."
                )

        progress.empty()
        status.empty()

        # ====================================================
        # STORE RESULTS
        # ====================================================

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

        last_scan = (
            st.session_state.last_stock_scan
        )

        if last_scan:

            st.caption(
                "Last scan: "
                + last_scan.strftime(
                    "%d-%m-%Y %H:%M:%S IST"
                )
            )

        # ====================================================
        # TOP SETUPS
        # ====================================================

        st.subheader(
            "🏆 Top Current Setups"
        )

        top = result_df[
            (
                result_df["bull_score"]
                >= min_buy_score
            )
            & (
                result_df["score"]
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

            display_columns = [
                col
                for col in display_columns
                if col in top.columns
            ]

            st.dataframe(
                top[
                    display_columns
                ].round(2),
                use_container_width=True,
                hide_index=True,
            )

        # ====================================================
        # TOP 5 DETAILS
        # ====================================================

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
                f"Bull {row['bull_score']:.0f}%"
            ):

                d1, d2, d3, d4, d5 = (
                    st.columns(5)
                )

                d1.metric(
                    "Price",
                    fmt(row["price"]),
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
                    fmt(row["rsi"]),
                )

                # ------------------------------------------------
                # Bullish
                # ------------------------------------------------

                st.write(
                    "### 🟢 Bullish Confirmation"
                )

                reasons = row.get(
                    "reasons",
                    [],
                )

                if isinstance(
                    reasons,
                    (list, tuple),
                ) and reasons:

                    for reason in reasons:

                        st.success(
                            "✓ " + str(reason)
                        )

                else:

                    st.info(
                        "Strong bullish confirmation nahi."
                    )

                # ------------------------------------------------
                # Bearish
                # ------------------------------------------------

                warnings = row.get(
                    "warnings",
                    [],
                )

                if isinstance(
                    warnings,
                    (list, tuple),
                ) and warnings:

                    st.write(
                        "### ⚠️ Bearish / Risk Factors"
                    )

                    for warning in warnings:

                        st.warning(
                            str(warning)
                        )

                # =================================================
                # FUNDAMENTALS
                # =================================================

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
                        f"₹{market_cap / 1e7:,.0f} Cr"
                        if finite(market_cap)
                        else "-"
                    ),
                )

                f2.metric(
                    "ROE",
                    pct(
                        fundamentals["roe"]
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

        # ====================================================
        # COMPLETE RESULTS
        # ====================================================

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

        complete_columns = [
            col
            for col in complete_columns
            if col in result_df.columns
        ]

        st.dataframe(
            result_df[
                complete_columns
            ].round(2),
            use_container_width=True,
            hide_index=True,
        )

        # ====================================================
        # BULLISH CHART
        # ====================================================

        st.subheader(
            "📊 Strongest Bullish Setups"
        )

        chart_df = (
            result_df
            .sort_values(
                "bull_score",
                ascending=False,
            )
            .head(10)
            .set_index("symbol")[
                ["bull_score"]
            ]
        )

        if not chart_df.empty:

            st.bar_chart(
                chart_df,
                height=400,
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
• Completed 5-minute candle handling improved.
• Yahoo MultiIndex OHLCV parsing hardened.
• Invalid OHLC rows filtered.
• Session VWAP calculation protected.
• Intraday EMA200 aur Daily EMA200 separate.
• Bullish/bearish scores independently calculated.
• Bull-vs-bear edge ranking mein included.
• Strong signal par hi CE/PE action activate hota hai.
• STRONG SELL UI ko correctly BUY PE action se map kiya gaya.
• Option selection spread + liquidity + OI + distance based hai.
• Extremely wide option spreads rejected.
• Max Pain calculation protected.
• OI support/resistance nearby liquid strikes se calculate hota hai.
• Yahoo download failures graceful handling ke saath hain.
• Fundamental data cached hai.
• Missing/invalid data se application crash avoid kiya gaya.
• Chart missing columns ke against protected hai.
• Cache refresh centralized hai.
• Stock scanner mein individual stock failure poore scan ko stop nahi karta.
• Trade plan invalid inputs par safely unavailable hota hai.

Actual trade se pehle broker ka live price,
bid/ask, liquidity, slippage, position size,
market conditions aur risk verify karein.
"""
)

st.caption(
    "Last app refresh: "
    + datetime.now(IST).strftime(
        "%d-%m-%Y %H:%M:%S IST"
    )
)

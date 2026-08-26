import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="NIFTY Signal Pro",
    page_icon="📈",
    layout="wide"
)

IST = ZoneInfo("Asia/Kolkata")
NIFTY = "^NSEI"

st.title("📈 NIFTY Signal Pro")
st.caption(
    "5-minute NIFTY + Option Chain confirmation dashboard"
)

# ============================================================
# SETTINGS
# ============================================================

REFRESH_SECONDS = 60
STRIKE_STEP = 50

# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=np.nan):
    try:
        if value is None:
            return default

        if isinstance(value, (pd.Series, pd.DataFrame)):
            value = value.iloc[0]

        value = float(value)

        if np.isfinite(value):
            return value

        return default

    except Exception:
        return default


def clean_ohlcv(df):

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # Fix yfinance MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    for col in required:
        if col not in df.columns:
            return pd.DataFrame()

    df = df[required].copy()

    for col in required:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna()

    return df


# ============================================================
# NIFTY DATA
# ============================================================

@st.cache_data(ttl=55, show_spinner=False)
def get_nifty_data():

    try:

        df = yf.download(
            NIFTY,
            period="5d",
            interval="5m",
            progress=False,
            auto_adjust=False,
            threads=False
        )

        return clean_ohlcv(df)

    except Exception:
        return pd.DataFrame()


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(df):

    df = df.copy()

    # EMA
    df["EMA20"] = (
        df["Close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    df["EMA50"] = (
        df["Close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    # RSI
    change = df["Close"].diff()

    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    df["RSI"] = (
        100 -
        (100 / (1 + rs))
    )

    # ATR
    tr1 = df["High"] - df["Low"]

    tr2 = (
        df["High"] -
        df["Close"].shift()
    ).abs()

    tr3 = (
        df["Low"] -
        df["Close"].shift()
    ).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["ATR"] = (
        tr.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean()
    )

    # VWAP
    typical = (
        df["High"] +
        df["Low"] +
        df["Close"]
    ) / 3

    volume = (
        df["Volume"]
        .replace(0, np.nan)
    )

    df["VWAP"] = (
        (typical * volume).cumsum() /
        volume.cumsum()
    )

    # Volume average
    df["VOL_AVG"] = (
        df["Volume"]
        .rolling(20)
        .mean()
    )

    # Recent breakout levels
    df["HIGH10"] = (
        df["High"]
        .shift(1)
        .rolling(10)
        .max()
    )

    df["LOW10"] = (
        df["Low"]
        .shift(1)
        .rolling(10)
        .min()
    )

    # Recent support/resistance
    df["HIGH20"] = (
        df["High"]
        .shift(1)
        .rolling(20)
        .max()
    )

    df["LOW20"] = (
        df["Low"]
        .shift(1)
        .rolling(20)
        .min()
    )

    return df.dropna()


# ============================================================
# OPTION CHAIN
# ============================================================

@st.cache_data(ttl=55, show_spinner=False)
def get_option_chain():

    try:

        ticker = yf.Ticker(NIFTY)

        expirations = list(
            ticker.options
        )

        if not expirations:
            return None, None, None, None

        # nearest available expiry
        expiry = expirations[0]

        chain = ticker.option_chain(
            expiry
        )

        calls = chain.calls.copy()
        puts = chain.puts.copy()

        return (
            expiry,
            calls,
            puts,
            expirations
        )

    except Exception:

        return None, None, None, None


# ============================================================
# OPTION CLEANING
# ============================================================

def clean_options(df):

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    numeric_cols = [
        "strike",
        "lastPrice",
        "bid",
        "ask",
        "change",
        "percentChange",
        "volume",
        "openInterest",
        "impliedVolatility"
    ]

    for col in numeric_cols:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    if "volume" not in df.columns:
        df["volume"] = 0

    if "openInterest" not in df.columns:
        df["openInterest"] = 0

    if "impliedVolatility" not in df.columns:
        df["impliedVolatility"] = np.nan

    df["volume"] = df["volume"].fillna(0)
    df["openInterest"] = (
        df["openInterest"]
        .fillna(0)
    )

    return df


# ============================================================
# ATM
# ============================================================

def get_atm(price):

    return int(
        round(price / STRIKE_STEP) *
        STRIKE_STEP
    )


# ============================================================
# OPTION SELECTION
# ============================================================

def select_option(
    df,
    atm,
    direction
):

    if df is None or df.empty:
        return None

    df = df.copy()

    # Keep strikes near ATM
    df["distance"] = (
        df["strike"] - atm
    ).abs()

    # Prefer ATM / slightly ITM
    if direction == "CE":

        candidates = df[
            df["strike"] <= atm + 100
        ].copy()

    else:

        candidates = df[
            df["strike"] >= atm - 100
        ].copy()

    if candidates.empty:
        candidates = df.copy()

    # Liquidity score
    candidates["spread"] = (
        candidates["ask"] -
        candidates["bid"]
    )

    candidates["mid"] = (
        candidates["bid"] +
        candidates["ask"]
    ) / 2

    candidates["spread_pct"] = np.where(
        candidates["mid"] > 0,
        (
            candidates["spread"] /
            candidates["mid"]
        ) * 100,
        999
    )

    # Score
    candidates["liq_score"] = (

        np.log1p(
            candidates["volume"]
        ) * 2

        +

        np.log1p(
            candidates["openInterest"]
        )

        -

        candidates["spread_pct"] * 2

        -

        candidates["distance"] / 50
    )

    candidates = candidates.sort_values(
        "liq_score",
        ascending=False
    )

    return candidates.iloc[0]


# ============================================================
# OPTION METRICS
# ============================================================

def option_price(row):

    if row is None:
        return np.nan

    bid = safe_float(
        row.get("bid")
    )

    ask = safe_float(
        row.get("ask")
    )

    last = safe_float(
        row.get("lastPrice")
    )

    # Best executable estimate = ask for BUY
    if np.isfinite(ask) and ask > 0:
        return ask

    if np.isfinite(bid) and bid > 0:
        return bid

    if np.isfinite(last) and last > 0:
        return last

    return np.nan


def option_mid(row):

    bid = safe_float(
        row.get("bid")
    )

    ask = safe_float(
        row.get("ask")
    )

    if (
        np.isfinite(bid)
        and np.isfinite(ask)
        and bid > 0
        and ask > 0
    ):
        return (bid + ask) / 2

    return option_price(row)


# ============================================================
# PCR
# ============================================================

def calculate_pcr(calls, puts):

    call_oi = safe_float(
        calls["openInterest"].sum()
        if not calls.empty else 0,
        0
    )

    put_oi = safe_float(
        puts["openInterest"].sum()
        if not puts.empty else 0,
        0
    )

    call_vol = safe_float(
        calls["volume"].sum()
        if not calls.empty else 0,
        0
    )

    put_vol = safe_float(
        puts["volume"].sum()
        if not puts.empty else 0,
        0
    )

    oi_pcr = (
        put_oi / call_oi
        if call_oi > 0
        else np.nan
    )

    vol_pcr = (
        put_vol / call_vol
        if call_vol > 0
        else np.nan
    )

    return (
        oi_pcr,
        vol_pcr,
        call_oi,
        put_oi
    )


# ============================================================
# MAX PAIN
# ============================================================

def calculate_max_pain(calls, puts):

    if calls.empty or puts.empty:
        return np.nan

    strikes = sorted(
        set(
            calls["strike"].dropna()
        ).intersection(
            set(
                puts["strike"].dropna()
            )
        )
    )

    if not strikes:
        return np.nan

    best_strike = None
    lowest_loss = None

    for expiry_strike in strikes:

        call_loss = (
            np.maximum(
                expiry_strike -
                calls["strike"],
                0
            )
            * calls["openInterest"]
        ).sum()

        put_loss = (
            np.maximum(
                puts["strike"] -
                expiry_strike,
                0
            )
            * puts["openInterest"]
        ).sum()

        total_loss = (
            call_loss +
            put_loss
        )

        if (
            lowest_loss is None
            or total_loss < lowest_loss
        ):

            lowest_loss = total_loss
            best_strike = expiry_strike

    return best_strike


# ============================================================
# OI LEVELS
# ============================================================

def get_oi_levels(calls, puts):

    call_resistance = np.nan
    put_support = np.nan

    if (
        not calls.empty
        and "openInterest" in calls
    ):

        c = calls.sort_values(
            "openInterest",
            ascending=False
        )

        if not c.empty:
            call_resistance = safe_float(
                c.iloc[0]["strike"]
            )

    if (
        not puts.empty
        and "openInterest" in puts
    ):

        p = puts.sort_values(
            "openInterest",
            ascending=False
        )

        if not p.empty:
            put_support = safe_float(
                p.iloc[0]["strike"]
            )

    return (
        call_resistance,
        put_support
    )


# ============================================================
# MARKET SIGNAL
# ============================================================

def calculate_signal(
    df,
    calls,
    puts
):

    last = df.iloc[-1]
    prev = df.iloc[-2]

    price = safe_float(last["Close"])
    ema20 = safe_float(last["EMA20"])
    ema50 = safe_float(last["EMA50"])
    vwap = safe_float(last["VWAP"])
    rsi = safe_float(last["RSI"])
    atr = safe_float(last["ATR"])
    volume = safe_float(last["Volume"])
    volavg = safe_float(last["VOL_AVG"])

    high10 = safe_float(last["HIGH10"])
    low10 = safe_float(last["LOW10"])

    prev_ema50 = safe_float(
        prev["EMA50"]
    )

    atr_avg = safe_float(
        df["ATR"]
        .rolling(20)
        .mean()
        .iloc[-1]
    )

    # -------------------------
    # CE
    # -------------------------

    ce_checks = {

        "Price above VWAP":
            price > vwap,

        "EMA20 above EMA50":
            ema20 > ema50,

        "EMA50 rising":
            ema50 > prev_ema50,

        "RSI bullish":
            rsi > 55,

        "Volume confirmation":
            volume > volavg * 1.10,

        "10-candle breakout":
            price > high10,

        "Bullish candle":
            price > safe_float(
                last["Open"]
            ),

        "ATR expansion":
            atr > atr_avg
    }

    # -------------------------
    # PE
    # -------------------------

    pe_checks = {

        "Price below VWAP":
            price < vwap,

        "EMA20 below EMA50":
            ema20 < ema50,

        "EMA50 falling":
            ema50 < prev_ema50,

        "RSI bearish":
            rsi < 45,

        "Volume confirmation":
            volume > volavg * 1.10,

        "10-candle breakdown":
            price < low10,

        "Bearish candle":
            price < safe_float(
                last["Open"]
            ),

        "ATR expansion":
            atr > atr_avg
    }

    ce_score = sum(
        ce_checks.values()
    )

    pe_score = sum(
        pe_checks.values()
    )

    return {
        "price": price,
        "ema20": ema20,
        "ema50": ema50,
        "vwap": vwap,
        "rsi": rsi,
        "atr": atr,
        "volume": volume,
        "volavg": volavg,
        "ce_checks": ce_checks,
        "pe_checks": pe_checks,
        "ce_score": ce_score,
        "pe_score": pe_score
    }


# ============================================================
# OPTION CONFIRMATION
# ============================================================

def option_confirmation(
    row,
    direction
):

    if row is None:
        return 0, []

    score = 0
    reasons = []

    volume = safe_float(
        row.get("volume"),
        0
    )

    oi = safe_float(
        row.get("openInterest"),
        0
    )

    bid = safe_float(
        row.get("bid")
    )

    ask = safe_float(
        row.get("ask")
    )

    last = safe_float(
        row.get("lastPrice")
    )

    change = safe_float(
        row.get("change"),
        0
    )

    mid = option_mid(row)

    # Volume
    if volume >= 500:
        score += 1
        reasons.append(
            "Good option volume"
        )

    # OI
    if oi >= 1000:
        score += 1
        reasons.append(
            "Good open interest"
        )

    # Spread
    if (
        np.isfinite(bid)
        and np.isfinite(ask)
        and mid > 0
    ):

        spread_pct = (
            (ask - bid) / mid
        ) * 100

        if spread_pct <= 3:
            score += 1
            reasons.append(
                "Tight bid-ask spread"
            )

    # Premium momentum
    if change > 0:
        score += 1
        reasons.append(
            "Premium momentum positive"
        )

    # Valid price
    if (
        np.isfinite(last)
        and last > 0
    ):
        score += 1
        reasons.append(
            "Valid option price"
        )

    return score, reasons


# ============================================================
# TRADE PLAN
# ============================================================

def build_trade_plan(
    signal,
    option_row,
    nifty_price,
    atr
):

    if option_row is None:
        return None

    entry = option_price(
        option_row
    )

    if not np.isfinite(entry) or entry <= 0:
        return None

    # Conservative premium risk
    sl_pct = 0.18

    if signal == "BUY CE":

        sl = entry * (
            1 - sl_pct
        )

        t1 = entry * 1.20
        t2 = entry * 1.35
        t3 = entry * 1.50

        direction = 1

    elif signal == "BUY PE":

        sl = entry * (
            1 - sl_pct
        )

        t1 = entry * 1.20
        t2 = entry * 1.35
        t3 = entry * 1.50

        direction = -1

    else:
        return None

    # NIFTY invalidation level
    if direction == 1:
        nifty_sl = nifty_price - (
            atr * 0.80
        )
    else:
        nifty_sl = nifty_price + (
            atr * 0.80
        )

    return {
        "entry": entry,
        "sl": sl,
        "t1": t1,
        "t2": t2,
        "t3": t3,
        "nifty_sl": nifty_sl
    }


# ============================================================
# REFRESH
# ============================================================

col_a, col_b = st.columns(2)

with col_a:

    if st.button(
        "🔄 Refresh NIFTY Data",
        use_container_width=True
    ):

        st.cache_data.clear()
        st.rerun()

with col_b:

    auto_refresh = st.checkbox(
        "⏱️ Auto refresh every 60 sec",
        value=True
    )


# ============================================================
# AUTO REFRESH
# ============================================================

if auto_refresh:

    try:

        @st.fragment(
            run_every="60s"
        )
        def refresh_marker():

            st.caption(
                "🟢 Auto refresh ON — data refreshes every 60 seconds."
            )

        refresh_marker()

    except Exception:

        st.caption(
            "Refresh manually if auto-refresh is unavailable."
        )


# ============================================================
# LOAD DATA
# ============================================================

with st.spinner(
    "NIFTY + Option Chain data load ho raha hai..."
):

    df = get_nifty_data()

    (
        expiry,
        calls,
        puts,
        expirations
    ) = get_option_chain()


# ============================================================
# DATA VALIDATION
# ============================================================

if df.empty:

    st.error(
        "❌ NIFTY data nahi mila. "
        "Refresh karke dobara try karein."
    )

    st.stop()


df = calculate_indicators(
    df
)

if len(df) < 60:

    st.error(
        "❌ Indicators calculate karne ke liye "
        "sufficient 5-minute candles nahi mili."
    )

    st.stop()


calls = clean_options(
    calls
)

puts = clean_options(
    puts
)


# ============================================================
# BASIC DATA
# ============================================================

market = calculate_signal(
    df,
    calls,
    puts
)

price = market["price"]
atr = market["atr"]

atm = get_atm(
    price
)


# ============================================================
# OPTION CHAIN ANALYSIS
# ============================================================

oi_pcr, vol_pcr, call_oi, put_oi = (
    calculate_pcr(
        calls,
        puts
    )
)

max_pain = calculate_max_pain(
    calls,
    puts
)

call_resistance, put_support = (
    get_oi_levels(
        calls,
        puts
    )
)


# ============================================================
# SELECT CE / PE
# ============================================================

ce_row = select_option(
    calls,
    atm,
    "CE"
)

pe_row = select_option(
    puts,
    atm,
    "PE"
)


ce_option_score, ce_reasons = (
    option_confirmation(
        ce_row,
        "CE"
    )
)

pe_option_score, pe_reasons = (
    option_confirmation(
        pe_row,
        "PE"
    )
)


# ============================================================
# FINAL SCORE
# ============================================================

ce_total = (
    market["ce_score"] +
    ce_option_score
)

pe_total = (
    market["pe_score"] +
    pe_option_score
)


# ============================================================
# FINAL DECISION
# ============================================================

signal = "WAIT / NO TRADE"
confidence = 0

if (
    ce_total >= 10
    and ce_total > pe_total
    and market["ce_score"] >= 6
    and ce_option_score >= 3
):

    signal = "BUY CE"
    confidence = min(
        95,
        55 + ce_total * 2
    )

elif (
    pe_total >= 10
    and pe_total > ce_total
    and market["pe_score"] >= 6
    and pe_option_score >= 3
):

    signal = "BUY PE"
    confidence = min(
        95,
        55 + pe_total * 2
    )


# ============================================================
# TRADE OPTION
# ============================================================

if signal == "BUY CE":

    selected = ce_row
    option_name = f"{atm} CE"

elif signal == "BUY PE":

    selected = pe_row
    option_name = f"{atm} PE"

else:

    selected = None
    option_name = "NO TRADE"


# ============================================================
# TRADE PLAN
# ============================================================

trade = build_trade_plan(
    signal,
    selected,
    price,
    atr
)


# ============================================================
# HEADER
# ============================================================

st.divider()

h1, h2, h3, h4 = st.columns(4)

h1.metric(
    "NIFTY",
    f"{price:,.2f}"
)

h2.metric(
    "ATM",
    f"{atm}"
)

h3.metric(
    "CE Score",
    f"{ce_total}/13"
)

h4.metric(
    "PE Score",
    f"{pe_total}/13"
)


# ============================================================
# SIGNAL
# ============================================================

st.subheader(
    "🚨 FINAL TRADING SIGNAL"
)

if signal == "BUY CE":

    st.success(
        f"""
🟢 STRONG BUY CE

Option: {option_name}

Confidence: {confidence}%

NIFTY trigger: {price:,.2f} ke upar strength maintain honi chahiye.
"""
    )

elif signal == "BUY PE":

    st.error(
        f"""
🔴 STRONG BUY PE

Option: {option_name}

Confidence: {confidence}%

NIFTY trigger: {price:,.2f} ke neeche weakness maintain honi chahiye.
"""
    )

else:

    st.warning(
        """
🟡 WAIT / NO TRADE

Abhi CE aur PE confirmation me sufficient edge nahi hai.

False entry se bachne ke liye system trade block kar raha hai.
"""
    )


# ============================================================
# OPTION PRICE
# ============================================================

st.subheader(
    "💰 Selected Option Details"
)

if selected is not None:

    option_strike = safe_float(
        selected.get("strike")
    )

    last_price = safe_float(
        selected.get("lastPrice")
    )

    bid = safe_float(
        selected.get("bid")
    )

    ask = safe_float(
        selected.get("ask")
    )

    mid = option_mid(
        selected
    )

    volume = safe_float(
        selected.get("volume"),
        0
    )

    oi = safe_float(
        selected.get("openInterest"),
        0
    )

    iv = safe_float(
        selected.get(
            "impliedVolatility"
        )
    )

    o1, o2, o3, o4 = st.columns(4)

    o1.metric(
        "Strike",
        "-" if not np.isfinite(option_strike)
        else f"{option_strike:.0f}"
    )

    o2.metric(
        "Last Price",
        "-" if not np.isfinite(last_price)
        else f"₹{last_price:.2f}"
    )

    o3.metric(
        "Bid",
        "-" if not np.isfinite(bid)
        else f"₹{bid:.2f}"
    )

    o4.metric(
        "Ask",
        "-" if not np.isfinite(ask)
        else f"₹{ask:.2f}"
    )

    o5, o6, o7, o8 = st.columns(4)

    o5.metric(
        "Mid",
        "-" if not np.isfinite(mid)
        else f"₹{mid:.2f}"
    )

    o6.metric(
        "Volume",
        f"{volume:,.0f}"
    )

    o7.metric(
        "Open Interest",
        f"{oi:,.0f}"
    )

    o8.metric(
        "IV",
        "-" if not np.isfinite(iv)
        else f"{iv * 100:.2f}%"
    )

else:

    st.info(
        "No confirmed option selected — WAIT."
    )


# ============================================================
# TRADE PLAN
# ============================================================

st.subheader(
    "🎯 Point-to-Point Trade Plan"
)

if trade is not None:

    t1, t2, t3, t4, t5 = st.columns(5)

    t1.metric(
        "ENTRY",
        f"₹{trade['entry']:.2f}"
    )

    t2.metric(
        "SL",
        f"₹{trade['sl']:.2f}"
    )

    t3.metric(
        "TARGET 1",
        f"₹{trade['t1']:.2f}"
    )

    t4.metric(
        "TARGET 2",
        f"₹{trade['t2']:.2f}"
    )

    t5.metric(
        "TARGET 3",
        f"₹{trade['t3']:.2f}"
    )

    st.info(
        f"""
NIFTY invalidation level:
{trade['nifty_sl']:.2f}

Option entry:
₹{trade['entry']:.2f}

Hard option SL:
₹{trade['sl']:.2f}

T1:
₹{trade['t1']:.2f}

T2:
₹{trade['t2']:.2f}

T3:
₹{trade['t3']:.2f}

T1 hit hone ke baad SL ko entry ke paas shift karna
better risk-control approach hai.
"""
    )

else:

    st.warning(
        "Trade plan available nahi hai because setup confirmed nahi hai."
    )


# ============================================================
# OPTION CHAIN SUMMARY
# ============================================================

st.subheader(
    "⛓️ Option Chain Intelligence"
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Expiry",
    "-" if expiry is None else expiry
)

c2.metric(
    "OI PCR",
    "-" if not np.isfinite(oi_pcr)
    else f"{oi_pcr:.2f}"
)

c3.metric(
    "Volume PCR",
    "-" if not np.isfinite(vol_pcr)
    else f"{vol_pcr:.2f}"
)

c4.metric(
    "Max Pain",
    "-" if not np.isfinite(max_pain)
    else f"{max_pain:.0f}"
)


c5, c6 = st.columns(2)

with c5:

    st.metric(
        "Highest Call OI Resistance",
        "-" if not np.isfinite(
            call_resistance
        )
        else f"{call_resistance:.0f}"
    )

with c6:

    st.metric(
        "Highest Put OI Support",
        "-" if not np.isfinite(
            put_support
        )
        else f"{put_support:.0f}"
    )


# ============================================================
# CE / PE COMPARISON
# ============================================================

st.subheader(
    "⚔️ CE vs PE Comparison"
)

comparison = pd.DataFrame({

    "Metric": [
        "NIFTY Score",
        "Option Score",
        "Total Score",
        "Strike",
        "Last Price",
        "Bid",
        "Ask",
        "Volume",
        "Open Interest",
        "IV"
    ],

    "CE": [
        market["ce_score"],
        ce_option_score,
        ce_total,
        safe_float(
            ce_row.get("strike")
            if ce_row is not None
            else np.nan
        ),
        safe_float(
            ce_row.get("lastPrice")
            if ce_row is not None
            else np.nan
        ),
        safe_float(
            ce_row.get("bid")
            if ce_row is not None
            else np.nan
        ),
        safe_float(
            ce_row.get("ask")
            if ce_row is not None
            else np.nan
        ),
        safe_float(
            ce_row.get("volume")
            if ce_row is not None
            else np.nan
        ),
        safe_float(
            ce_row.get("openInterest")
            if ce_row is not None
            else np.nan
        ),
        safe_float(
            ce_row.get(
                "impliedVolatility"
            )
            if ce_row is not None
            else np.nan
        )
    ],

    "PE": [
        market["pe_score"],
        pe_option_score,
        pe_total,
        safe_float(
            pe_row.get("strike")
            if pe_row is not None
            else np.nan
        ),
        safe_float(
            pe_row.get("lastPrice")
            if pe_row is not None
            else np.nan
        ),
        safe_float(
            pe_row.get("bid")
            if pe_row is not None
            else np.nan
        ),
        safe_float(
            pe_row.get("ask")
            if pe_row is not None
            else np.nan
        ),
        safe_float(
            pe_row.get("volume")
            if pe_row is not None
            else np.nan
        ),
        safe_float(
            pe_row.get("openInterest")
            if pe_row is not None
            else np.nan
        ),
        safe_float(
            pe_row.get(
                "impliedVolatility"
            )
            if pe_row is not None
            else np.nan
        )
    ]
})

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# MARKET ANALYSIS
# ============================================================

st.subheader(
    "🔎 NIFTY Market Analysis"
)

analysis = pd.DataFrame({

    "Indicator": [
        "NIFTY Price",
        "EMA 20",
        "EMA 50",
        "VWAP",
        "RSI",
        "ATR",
        "Volume",
        "Average Volume",
        "CE NIFTY Score",
        "PE NIFTY Score",
        "CE Option Score",
        "PE Option Score",
        "CE Final Score",
        "PE Final Score"
    ],

    "Value": [
        f"{price:.2f}",
        f"{market['ema20']:.2f}",
        f"{market['ema50']:.2f}",
        f"{market['vwap']:.2f}",
        f"{market['rsi']:.2f}",
        f"{market['atr']:.2f}",
        f"{market['volume']:.0f}",
        f"{market['volavg']:.0f}",
        f"{market['ce_score']}/8",
        f"{market['pe_score']}/8",
        f"{ce_option_score}/5",
        f"{pe_option_score}/5",
        f"{ce_total}/13",
        f"{pe_total}/13"
    ]
})

st.dataframe(
    analysis,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# SIGNAL CHECKLIST
# ============================================================

st.subheader(
    "✅ Signal Checklist"
)

left, right = st.columns(2)

with left:

    st.markdown(
        "### 🟢 CE Confirmation"
    )

    for name, result in market[
        "ce_checks"
    ].items():

        if result:
            st.success(
                f"✓ {name}"
            )
        else:
            st.error(
                f"✗ {name}"
            )

    for reason in ce_reasons:
        st.info(
            f"CE option: {reason}"
        )


with right:

    st.markdown(
        "### 🔴 PE Confirmation"
    )

    for name, result in market[
        "pe_checks"
    ].items():

        if result:
            st.success(
                f"✓ {name}"
            )
        else:
            st.error(
                f"✗ {name}"
            )

    for reason in pe_reasons:
        st.info(
            f"PE option: {reason}"
        )


# ============================================================
# CHART
# ============================================================

st.subheader(
    "📊 NIFTY 5-Minute Chart"
)

chart = df[
    [
        "Close",
        "EMA20",
        "EMA50",
        "VWAP"
    ]
].tail(100)

st.line_chart(
    chart,
    height=450
)


# ============================================================
# EXPIRATIONS
# ============================================================

with st.expander(
    "📅 Available Expirations"
):

    if expirations:

        st.write(
            expirations[:10]
        )

    else:

        st.write(
            "No expiration data available."
        )


# ============================================================
# DATA WARNING
# ============================================================

st.divider()

st.warning(
    """
⚠️ IMPORTANT

Ye system 100% guaranteed profit nahi de sakta.

Signal tabhi strong maana jayega jab:
1. NIFTY trend confirm ho
2. VWAP/EMA direction agree kare
3. RSI + volume confirmation mile
4. Option-chain side same direction support kare
5. Selected option liquid ho
6. Bid/Ask spread acceptable ho
7. CE aur PE scores me clear difference ho

Agar conditions conflict karti hain to system WAIT karega.

Option premium NIFTY se alag move karta hai because of
delta, gamma, IV aur time decay.

Isliye actual order place karne se pehle broker ke
live bid/ask ko verify karna zaroori hai.
"""
)

st.caption(
    f"Last dashboard refresh: "
    f"{datetime.now(IST).strftime('%d-%m-%Y %H:%M:%S')} IST"
)

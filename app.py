import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import math

# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="NIFTY Signal Pro",
    page_icon="📈",
    layout="wide"
)

st.title("📈 NIFTY Signal Pro")
st.caption("5-minute rule-based NIFTY signal dashboard")


# ============================================================
# REFRESH
# ============================================================

if st.button("🔄 Refresh NIFTY Data"):
    st.cache_data.clear()
    st.rerun()


# ============================================================
# GET NIFTY DATA
# ============================================================

@st.cache_data(ttl=60)
def get_data():

    try:

        df = yf.download(
            "^NSEI",
            period="5d",
            interval="5m",
            progress=False,
            auto_adjust=False,
            threads=False
        )

        # Empty data protection
        if df is None or df.empty:
            return pd.DataFrame()

        # Handle MultiIndex returned by yfinance
        if isinstance(df.columns, pd.MultiIndex):

            try:
                df.columns = df.columns.get_level_values(0)
            except Exception:
                return pd.DataFrame()

        required_columns = [
            "Open",
            "High",
            "Low",
            "Close"
        ]

        # Check required columns
        for column in required_columns:

            if column not in df.columns:
                return pd.DataFrame()

        # Volume may not always be available for index data
        if "Volume" not in df.columns:
            df["Volume"] = 0

        # Keep only required columns
        df = df[
            [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume"
            ]
        ].copy()

        # Convert numeric columns
        for column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        # Remove invalid rows
        df = df.dropna(
            subset=[
                "Open",
                "High",
                "Low",
                "Close"
            ]
        )

        return df

    except Exception:

        return pd.DataFrame()


# ============================================================
# CALCULATE INDICATORS
# ============================================================

def calculate(df):

    if df.empty:
        return pd.DataFrame()

    df = df.copy()

    # --------------------------------------------------------
    # EMA 20
    # --------------------------------------------------------

    df["EMA20"] = (
        df["Close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    # --------------------------------------------------------
    # EMA 50
    # --------------------------------------------------------

    df["EMA50"] = (
        df["Close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    change = df["Close"].diff()

    gain = change.clip(lower=0)

    loss = -change.clip(upper=0)

    avg_gain = (
        gain
        .ewm(
            alpha=1 / 14,
            adjust=False
        )
        .mean()
    )

    avg_loss = (
        loss
        .ewm(
            alpha=1 / 14,
            adjust=False
        )
        .mean()
    )

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    df["RSI"] = (
        100 -
        (
            100 /
            (1 + rs)
        )
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    tr1 = (
        df["High"] -
        df["Low"]
    )

    tr2 = (
        df["High"] -
        df["Close"].shift()
    ).abs()

    tr3 = (
        df["Low"] -
        df["Close"].shift()
    ).abs()

    tr = pd.concat(
        [
            tr1,
            tr2,
            tr3
        ],
        axis=1
    ).max(axis=1)

    df["ATR"] = (
        tr
        .ewm(
            alpha=1 / 14,
            adjust=False
        )
        .mean()
    )

    # --------------------------------------------------------
    # VWAP
    # --------------------------------------------------------

    typical_price = (
        df["High"] +
        df["Low"] +
        df["Close"]
    ) / 3

    volume = (
        df["Volume"]
        .fillna(0)
    )

    # If volume exists, calculate VWAP
    if volume.sum() > 0:

        cumulative_volume = (
            volume.cumsum()
        )

        df["VWAP"] = (
            typical_price *
            volume
        ).cumsum() / cumulative_volume.replace(
            0,
            np.nan
        )

    else:

        # Fallback when NIFTY index volume is unavailable
        df["VWAP"] = (
            typical_price
            .rolling(20)
            .mean()
        )

    # --------------------------------------------------------
    # Volume Average
    # --------------------------------------------------------

    df["VOL_AVG"] = (
        df["Volume"]
        .rolling(20)
        .mean()
    )

    # --------------------------------------------------------
    # Breakout levels
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Average ATR
    # --------------------------------------------------------

    df["ATR_AVG"] = (
        df["ATR"]
        .rolling(20)
        .mean()
    )

    # Remove rows where indicators are unavailable
    df = df.dropna(
        subset=[
            "EMA20",
            "EMA50",
            "RSI",
            "ATR",
            "VWAP",
            "HIGH10",
            "LOW10",
            "ATR_AVG"
        ]
    )

    return df


# ============================================================
# LOAD DATA
# ============================================================

df = get_data()


# ============================================================
# EMPTY DATA PROTECTION
# ============================================================

if df.empty:

    st.error(
        "⚠️ Yahoo Finance se NIFTY 5-minute data nahi mil raha."
    )

    st.info(
        "🔄 1-2 minute baad 'Refresh NIFTY Data' button dabao."
    )

    st.stop()


# ============================================================
# CALCULATE
# ============================================================

df = calculate(df)


# ============================================================
# INSUFFICIENT DATA PROTECTION
# ============================================================

if df.empty or len(df) < 30:

    st.error(
        "⚠️ Indicators calculate karne ke liye sufficient NIFTY data nahi hai."
    )

    st.info(
        "Please refresh the app and try again."
    )

    st.stop()


# ============================================================
# LAST CANDLES
# ============================================================

last = df.iloc[-1]

prev = df.iloc[-2]


# ============================================================
# CURRENT VALUES
# ============================================================

price = float(last["Close"])

ema20 = float(last["EMA20"])

ema50 = float(last["EMA50"])

vwap = float(last["VWAP"])

rsi = float(last["RSI"])

atr = float(last["ATR"])

volume = float(last["Volume"])

volavg = float(last["VOL_AVG"])

atr_avg = float(last["ATR_AVG"])

high10 = float(last["HIGH10"])

low10 = float(last["LOW10"])

previous_ema50 = float(
    prev["EMA50"]
)


# ============================================================
# VOLUME CHECK
# ============================================================

volume_available = (
    volume > 0 and
    volavg > 0
)

if volume_available:

    volume_ce_ok = (
        volume >
        volavg * 1.10
    )

    volume_pe_ok = (
        volume >
        volavg * 1.10
    )

else:

    volume_ce_ok = None

    volume_pe_ok = None


# ============================================================
# ATR CHECK
# ============================================================

atr_ok = (
    atr >
    atr_avg
)


# ============================================================
# BUY CE CONDITIONS
# ============================================================

ce_conditions = [

    price > vwap,

    ema20 > ema50,

    ema50 > previous_ema50,

    rsi > 55,

    price > high10,

    price > float(last["Open"]),

    atr_ok
]


# Add volume only when available
if volume_available:

    ce_conditions.append(
        volume_ce_ok
    )


# ============================================================
# BUY PE CONDITIONS
# ============================================================

pe_conditions = [

    price < vwap,

    ema20 < ema50,

    ema50 < previous_ema50,

    rsi < 45,

    price < low10,

    price < float(last["Open"]),

    atr_ok
]


# Add volume only when available
if volume_available:

    pe_conditions.append(
        volume_pe_ok
    )


# ============================================================
# SCORES
# ============================================================

ce_score = sum(
    bool(x)
    for x in ce_conditions
)

pe_score = sum(
    bool(x)
    for x in pe_conditions
)

total_conditions = len(
    ce_conditions
)

required_score = math.ceil(
    total_conditions * 0.75
)


# ============================================================
# FINAL SIGNAL
# ============================================================

if (
    ce_score >= required_score
    and
    ce_score > pe_score
):

    signal = "BUY CE"

    signal_type = "success"

    entry = price

    sl = (
        price -
        atr * 1.35
    )

    risk = (
        entry -
        sl
    )

    t1 = (
        entry +
        risk * 1.5
    )

    t2 = (
        entry +
        risk * 2.5
    )

    trail = (
        entry +
        risk
    )


elif (
    pe_score >= required_score
    and
    pe_score > ce_score
):

    signal = "BUY PE"

    signal_type = "error"

    entry = price

    sl = (
        price +
        atr * 1.35
    )

    risk = (
        sl -
        entry
    )

    t1 = (
        entry -
        risk * 1.5
    )

    t2 = (
        entry -
        risk * 2.5
    )

    trail = (
        entry -
        risk
    )


else:

    signal = "WAIT / NO TRADE"

    signal_type = "warning"

    entry = None

    sl = None

    t1 = None

    t2 = None

    trail = None


# ============================================================
# ATM STRIKE
# ============================================================

atm = int(
    round(price / 50) * 50
)


if signal == "BUY CE":

    option = f"{atm} CE"


elif signal == "BUY PE":

    option = f"{atm} PE"


else:

    option = "No trade"


# ============================================================
# MAIN PRICE
# ============================================================

st.metric(
    "NIFTY",
    f"{price:,.2f}"
)


# ============================================================
# SIGNAL DISPLAY
# ============================================================

if signal_type == "success":

    st.success(
        f"🟢 {signal} | Suggested: {option}"
    )


elif signal_type == "error":

    st.error(
        f"🔴 {signal} | Suggested: {option}"
    )


else:

    st.warning(
        "🟡 WAIT — Abhi entry mat lo"
    )


# ============================================================
# TRADE PLAN
# ============================================================

st.subheader(
    "🎯 Trade Plan"
)


c1, c2, c3, c4, c5 = st.columns(5)


c1.metric(
    "Entry",
    "-"
    if entry is None
    else f"{entry:.2f}"
)


c2.metric(
    "SL",
    "-"
    if sl is None
    else f"{sl:.2f}"
)


c3.metric(
    "T1",
    "-"
    if t1 is None
    else f"{t1:.2f}"
)


c4.metric(
    "T2",
    "-"
    if t2 is None
    else f"{t2:.2f}"
)


c5.metric(
    "Trailing SL",
    "-"
    if trail is None
    else f"{trail:.2f}"
)


# ============================================================
# MARKET ANALYSIS
# ============================================================

st.subheader(
    "🔎 Market Analysis"
)


analysis = pd.DataFrame({

    "Indicator": [

        "Price",

        "EMA 20",

        "EMA 50",

        "VWAP",

        "RSI",

        "ATR",

        "Volume",

        "Average Volume",

        "CE Score",

        "PE Score",

        "Required Score"

    ],

    "Value": [

        f"{price:.2f}",

        f"{ema20:.2f}",

        f"{ema50:.2f}",

        f"{vwap:.2f}",

        f"{rsi:.2f}",

        f"{atr:.2f}",

        f"{volume:.0f}",

        f"{volavg:.0f}",

        f"{ce_score}/{total_conditions}",

        f"{pe_score}/{total_conditions}",

        f"{required_score}/{total_conditions}"

    ]

})


st.dataframe(
    analysis,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# CONDITION STATUS
# ============================================================

st.subheader(
    "📋 Signal Conditions"
)


condition_data = pd.DataFrame({

    "Condition": [

        "Price > VWAP",

        "EMA20 > EMA50",

        "EMA50 rising",

        "RSI > 55",

        "10-candle breakout",

        "Bullish candle",

        "ATR above average"

    ],

    "CE": [

        price > vwap,

        ema20 > ema50,

        ema50 > previous_ema50,

        rsi > 55,

        price > high10,

        price > float(last["Open"]),

        atr_ok

    ],

    "PE": [

        price < vwap,

        ema20 < ema50,

        ema50 < previous_ema50,

        rsi < 45,

        price < low10,

        price < float(last["Open"]),

        atr_ok

    ]

})


if volume_available:

    volume_row = pd.DataFrame({

        "Condition": ["Volume > 1.1 × Average"],

        "CE": [volume_ce_ok],

        "PE": [volume_pe_ok]

    })

    condition_data = pd.concat(
        [
            condition_data,
            volume_row
        ],
        ignore_index=True
    )


st.dataframe(
    condition_data,
    use_container_width=True,
    hide_index=True
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
# DATA TIME
# ============================================================

if hasattr(df.index, "tz"):

    last_time = df.index[-1]

else:

    last_time = df.index[-1]


st.caption(
    f"Latest candle: {last_time}"
)


# ============================================================
# IMPORTANT DISCLAIMER
# ============================================================

st.info(
    "Signal NIFTY underlying ke basis par hai. "
    "Option premium ka movement alag ho sakta hai. "
    "Entry se pehle actual option price, liquidity aur spread check karein."
)


st.warning(
    "⚠️ Ye rules-based system hai. Guaranteed profit nahi hai. "
    "Signal ko blindly follow karke trade na karein."
)

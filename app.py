import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="NIFTY Signal Pro",
    page_icon="📈",
    layout="wide"
)

st.title("📈 NIFTY Signal Pro")
st.caption("5-minute rule-based NIFTY signal dashboard")

@st.cache_data(ttl=60)
def get_data():
    df = yf.download(
        "^NSEI",
        period="5d",
        interval="5m",
        progress=False,
        auto_adjust=False
    )

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df.dropna()


def calculate(df):

    df["EMA20"] = df["Close"].ewm(
        span=20, adjust=False
    ).mean()

    df["EMA50"] = df["Close"].ewm(
        span=50, adjust=False
    ).mean()

    # RSI
    change = df["Close"].diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI"] = 100 - (100 / (1 + rs))

    # ATR
    tr1 = df["High"] - df["Low"]
    tr2 = abs(df["High"] - df["Close"].shift())
    tr3 = abs(df["Low"] - df["Close"].shift())

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["ATR"] = tr.ewm(
        alpha=1/14,
        adjust=False
    ).mean()

    # VWAP
    typical = (
        df["High"] +
        df["Low"] +
        df["Close"]
    ) / 3

    volume = df["Volume"].replace(0, np.nan)

    df["VWAP"] = (
        typical * volume
    ).cumsum() / volume.cumsum()

    # Volume
    df["VOL_AVG"] = df["Volume"].rolling(20).mean()

    # Breakout levels
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

    return df.dropna()


df = get_data()

if df.empty:
    st.error(
        "NIFTY data nahi mil raha. Page refresh karke dobara try karo."
    )
    st.stop()

df = calculate(df)

last = df.iloc[-1]
prev = df.iloc[-2]

price = float(last["Close"])
ema20 = float(last["EMA20"])
ema50 = float(last["EMA50"])
vwap = float(last["VWAP"])
rsi = float(last["RSI"])
atr = float(last["ATR"])
volume = float(last["Volume"])
volavg = float(last["VOL_AVG"])

# -----------------------------
# BUY CE CONDITIONS
# -----------------------------

ce_conditions = [

    price > vwap,

    ema20 > ema50,

    ema50 > float(prev["EMA50"]),

    rsi > 55,

    volume > volavg * 1.10,

    price > float(last["HIGH10"]),

    price > float(last["Open"]),

    atr > float(df["ATR"].rolling(20).mean().iloc[-1])
]

# -----------------------------
# BUY PE CONDITIONS
# -----------------------------

pe_conditions = [

    price < vwap,

    ema20 < ema50,

    ema50 < float(prev["EMA50"]),

    rsi < 45,

    volume > volavg * 1.10,

    price < float(last["LOW10"]),

    price < float(last["Open"]),

    atr > float(df["ATR"].rolling(20).mean().iloc[-1])
]

ce_score = sum(ce_conditions)
pe_score = sum(pe_conditions)

# -----------------------------
# FINAL DECISION
# -----------------------------

if ce_score >= 6 and ce_score > pe_score:

    signal = "BUY CE"
    signal_type = "success"

    entry = price
    sl = price - atr * 1.35
    risk = entry - sl

    t1 = entry + risk * 1.5
    t2 = entry + risk * 2.5
    trail = entry + risk

elif pe_score >= 6 and pe_score > ce_score:

    signal = "BUY PE"
    signal_type = "error"

    entry = price
    sl = price + atr * 1.35
    risk = sl - entry

    t1 = entry - risk * 1.5
    t2 = entry - risk * 2.5
    trail = entry - risk

else:

    signal = "WAIT / NO TRADE"
    signal_type = "warning"

    entry = sl = t1 = t2 = trail = None


# -----------------------------
# STRIKE
# -----------------------------

atm = round(price / 50) * 50

if signal == "BUY CE":
    option = f"{atm} CE"

elif signal == "BUY PE":
    option = f"{atm} PE"

else:
    option = "No trade"


# -----------------------------
# MAIN DISPLAY
# -----------------------------

st.metric(
    "NIFTY",
    f"{price:,.2f}"
)

if signal_type == "success":

    st.success(
        f"🟢 {signal}  |  Suggested: {option}"
    )

elif signal_type == "error":

    st.error(
        f"🔴 {signal}  |  Suggested: {option}"
    )

else:

    st.warning(
        "🟡 WAIT — Abhi entry mat lo"
    )


# -----------------------------
# TRADE LEVELS
# -----------------------------

st.subheader("🎯 Trade Plan")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Entry",
    "-" if entry is None else f"{entry:.2f}"
)

c2.metric(
    "SL",
    "-" if sl is None else f"{sl:.2f}"
)

c3.metric(
    "T1",
    "-" if t1 is None else f"{t1:.2f}"
)

c4.metric(
    "T2",
    "-" if t2 is None else f"{t2:.2f}"
)

c5.metric(
    "Trailing SL",
    "-" if trail is None else f"{trail:.2f}"
)


# -----------------------------
# ANALYSIS
# -----------------------------

st.subheader("🔎 Market Analysis")

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
        "PE Score"
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
        f"{ce_score}/8",
        f"{pe_score}/8"
    ]
})

st.dataframe(
    analysis,
    use_container_width=True,
    hide_index=True
)


# -----------------------------
# CHART
# -----------------------------

st.subheader("📊 NIFTY 5-Minute Chart")

chart = df[
    ["Close", "EMA20", "EMA50", "VWAP"]
].tail(100)

st.line_chart(
    chart,
    height=450
)


# -----------------------------
# IMPORTANT
# -----------------------------

st.info(
    "Signal NIFTY underlying ke basis par hai. "
    "Option premium ka movement alag ho sakta hai. "
    "Entry se pehle actual option price, liquidity aur spread check karein."
)

st.caption(
    "Rules-based system hai; guaranteed profit nahi hai. "
    "Signal ko blindly follow karke trade na karein."
)

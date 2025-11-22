# Signal  — Binary

import logging
import os
import json
import io
from datetime import datetime, timedelta

import requests
import pandas as pd
import ta
import matplotlib.pyplot as plt

from telegram import (
    Update,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    ReplyKeyboardMarkup,
    KeyboardButton,
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    CallbackQueryHandler,
    MessageHandler,
    filters,
)

from openai import OpenAI  # OpenAI SDK
import asyncio

# ===========================
# 🔐 CONFIG — EDIT THESE
# ===========================

SIGNAL_VERSION = "2.0.0"

# 1) Telegram bot token
BOT_TOKEN = os.getenv("BOT_TOKEN", "")  # <-- PUT YOUR TELEGRAM BOT TOKEN HERE

# 2) Data providers (keys can also come from env vars)
DATA_PROVIDERS = {
    "ALPHAVANTAGE": {
        "api_key": os.getenv("ALPHAVANTAGE_KEY", ""),   # or hardcode string
        "base_url": "https://www.alphavantage.co/query",
    },
    "TWELVEDATA": {
        "api_key": os.getenv("TWELVEDATA_KEY", ""),     # or hardcode string
        "base_url": "https://api.twelvedata.com",
    },
}

# 3) OpenAI config (for AI confirmation)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # or put your key here as string
OPENAI_MODEL = "gpt-4o-mini"

# Default timeframe (data)
DEFAULT_TIMEFRAME = "1h"

# Bot display timezone (WAT = UTC+1)
BOT_TZ_NAME = "WAT"
BOT_TZ_OFFSET_HOURS = 1

# Stagnation detection
STAGNATION_WINDOW_CANDLES = 10
STAGNATION_RANGE_THRESHOLD_PCT = 0.05  # 0.05% range → considered stagnant

# Volatility filters (ATR as % of price)
MIN_ATR_PCT = 0.03   # too quiet if below this
MAX_ATR_PCT = 1.50   # too wild if above this

# ===========================
# MARKETS & CATEGORIES
# ===========================

# Base lists
FOREX_PAIRS = [
    "EURUSD", "GBPUSD", "XAUUSD", "USDJPY",
    "USDCAD", "AUDUSD", "EURGBP",
]

CRYPTO_PAIRS = [
    "BTCUSD", "ETHUSD", "LTCUSD",
]

INDEX_SYMBOLS = [
    "NAS100", "SPX500", "US30",
]

COMMODITY_SYMBOLS = [
    "OILUSD", "XAGUSD",  # Oil, Silver
]

STOCK_SYMBOLS = [
    "AAPL", "TSLA", "MSFT",
]

# Add 10 more per category
FOREX_PAIRS_EXTRA = [
    "NZDUSD", "EURJPY", "GBPJPY", "AUDJPY", "EURCAD",
    "GBPCAD", "AUDCAD", "NZDJPY", "CHFJPY", "USDCHF",
]

CRYPTO_PAIRS_EXTRA = [
    "XRPUSD", "SOLUSD", "ADAUSD", "BNBUSD", "DOGEUSD",
    "AVAXUSD", "TRXUSD", "DOTUSD", "MATICUSD", "LINKUSD",
]

INDEX_SYMBOLS_EXTRA = [
    "GER40", "FRA40", "UK100", "JP225", "AUS200",
    "HK50", "ES35", "STOXX50", "RUS2000", "VIX",
]

COMMODITY_SYMBOLS_EXTRA = [
    "UKOIL", "NGAS", "COPPER", "CORN", "WHEAT",
    "SOYBEAN", "SUGAR", "COFFEE", "COTTON", "PLATINUM",
]

STOCK_SYMBOLS_EXTRA = [
    "GOOGL", "AMZN", "META", "NVDA", "NFLX",
    "JPM", "BAC", "ORCL", "INTC", "IBM",
]

FOREX_PAIRS = FOREX_PAIRS + FOREX_PAIRS_EXTRA
CRYPTO_PAIRS = CRYPTO_PAIRS + CRYPTO_PAIRS_EXTRA
INDEX_SYMBOLS = INDEX_SYMBOLS + INDEX_SYMBOLS_EXTRA
COMMODITY_SYMBOLS = COMMODITY_SYMBOLS + COMMODITY_SYMBOLS_EXTRA
STOCK_SYMBOLS = STOCK_SYMBOLS + STOCK_SYMBOLS_EXTRA

MARKET_CATEGORIES = {
    "Forex": FOREX_PAIRS,
    "Crypto": CRYPTO_PAIRS,
    "Indices": INDEX_SYMBOLS,
    "Commodities": COMMODITY_SYMBOLS,
    "Stocks": STOCK_SYMBOLS,
}

# For quick validation of any symbol
ALL_SYMBOLS = set(
    FOREX_PAIRS
    + CRYPTO_PAIRS
    + INDEX_SYMBOLS
    + COMMODITY_SYMBOLS
    + STOCK_SYMBOLS
)

# Mapping from user-facing symbol → TwelveData internal symbol
TWELVEDATA_SYMBOL_MAP = {
    "NAS100": "NDX",        # NASDAQ 100
    "SPX500": "SPX",        # S&P 500
    "US30": "DJI",          # Dow Jones
    "OILUSD": "WTI",        # Crude oil
    "XAGUSD": "XAG/USD",    # Silver
    "XAUUSD": "XAU/USD",    # Gold
}

SUBSCRIPTIONS_FILE = "subscriptions.json"
PREMIUM_FILE = "premium_users.json"
SIGNALS_LOG_FILE = "signals_log.csv"

PREMIUM_ACCESS_CODE = "FOREA3URPO12S"

# Premium-only bot
MAX_FREE_SUBSCRIPTIONS = 0

# ===========================
# LOGGING
# ===========================
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

subscriptions = {}
premium_users = set()
pending_unsub = {}

# ===========================
# STORAGE HELPERS
# ===========================
def load_json_file(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Error loading %s: %s", path, e)
        return default


def save_json_file(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error("Error saving %s: %s", path, e)


def load_data():
    global subscriptions, premium_users
    subscriptions = load_json_file(SUBSCRIPTIONS_FILE, {})
    premium_list = load_json_file(PREMIUM_FILE, [])
    premium_users.clear()
    for uid in premium_list:
        premium_users.add(str(uid))
    logger.info(
        "Loaded %d subscriptions, %d premium users",
        len(subscriptions),
        len(premium_users),
    )


def save_subscriptions():
    save_json_file(SUBSCRIPTIONS_FILE, subscriptions)


def save_premium():
    save_json_file(PREMIUM_FILE, list(premium_users))


def is_premium(user_id: int) -> bool:
    return str(user_id) in premium_users


def log_signal(pair: str, source: str):
    header_needed = not os.path.exists(SIGNALS_LOG_FILE)
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{now},{pair},{source}\n"
    try:
        with open(SIGNALS_LOG_FILE, "a", encoding="utf-8") as f:
            if header_needed:
                f.write("timestamp_utc,pair,source\n")
            f.write(line)
    except Exception as e:
        logger.error("Error logging signal: %s", e)

# ===========================
# TIME & WEEKEND / OTC
# ===========================
def utc_now() -> datetime:
    return datetime.utcnow()


def wat_now() -> datetime:
    return utc_now() + timedelta(hours=BOT_TZ_OFFSET_HOURS)


def format_time_local(dt: datetime) -> str:
    """Return '8:05 am' style (no leading 0, lowercase am/pm)."""
    s = dt.strftime("%I:%M %p").lstrip("0")
    s = s.replace("AM", "am").replace("PM", "pm")
    return s


def is_weekend_utc(now_utc: datetime | None = None) -> bool:
    if now_utc is None:
        now_utc = utc_now()
    # Saturday = 5, Sunday = 6
    return now_utc.weekday() >= 5

# ===========================
# FLAG & NAME HELPERS
# ===========================
CURRENCY_FLAGS = {
    "EUR": "🇪🇺",
    "GBP": "🇬🇧",
    "USD": "🇺🇸",
    "JPY": "🇯🇵",
    "CAD": "🇨🇦",
    "AUD": "🇦🇺",
    "NZD": "🇳🇿",
    "CHF": "🇨🇭",
    "XAU": "🟡",
    "XAG": "⚪",
}

def pretty_pair_name(symbol: str, weekend_otc: bool) -> tuple[str, str]:
    """
    Build display name and flags.
    For FX: EURUSD -> 'EUR/USD OTC 🇪🇺 / 🇬🇧' style.
    For others: just symbol + a generic flag if available.
    """
    is_fx = len(symbol) == 6 and symbol[:3].isalpha() and symbol[3:].isalpha()

    if is_fx:
        base = symbol[:3]
        quote = symbol[3:]
        base_flag = CURRENCY_FLAGS.get(base, "")
        quote_flag = CURRENCY_FLAGS.get(quote, "")
        name = f"{base}/{quote}"
        if weekend_otc:
            name += " OTC"
        flag_str = ""
        if base_flag or quote_flag:
            if base_flag and quote_flag:
                flag_str = f"{base_flag} / {quote_flag}"
            else:
                flag_str = base_flag or quote_flag
        return name, flag_str

    # Non-FX: just show symbol, maybe OTC label
    name = symbol
    if weekend_otc:
        name += " OTC"
    # simple global default
    flag_str = "🇺🇸"
    return name, flag_str

# ===========================
# MENU / KEYBOARD HELPERS
# ===========================
def build_main_menu(is_premium_user: bool) -> ReplyKeyboardMarkup:
    if is_premium_user:
        keyboard = [
            [KeyboardButton("/signal"), KeyboardButton("/pairs")],
            [KeyboardButton("/subscribe"), KeyboardButton("/unsubscribe")],
            [KeyboardButton("/help"), KeyboardButton("/debugsymbol ETHUSD")],
        ]
    else:
        keyboard = [
            [KeyboardButton("/register")],
            [KeyboardButton("/help")],
        ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)


async def ensure_premium(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    user = update.effective_user
    if is_premium(user.id):
        return True

    reply_markup = build_main_menu(False)
    await update.message.reply_markdown(
        "🔐 *Premium-only feature.*\n\n"
        "To activate the bot, tap `/register` and enter your premium code.",
        reply_markup=reply_markup,
    )
    return False

# ===========================
# DATA PROVIDER SELECTION
# ===========================
def get_active_provider_for_symbol(symbol: str) -> str | None:
    td_key = DATA_PROVIDERS["TWELVEDATA"]["api_key"].strip()
    av_key = DATA_PROVIDERS["ALPHAVANTAGE"]["api_key"].strip()

    if td_key:
        return "TWELVEDATA"
    if av_key and symbol in FOREX_PAIRS:
        return "ALPHAVANTAGE"
    return None

# ===========================
# MARKET DATA FETCHING
# ===========================
def fetch_from_twelvedata(symbol: str, timeframe: str = "1h") -> pd.DataFrame:
    api_key = DATA_PROVIDERS["TWELVEDATA"]["api_key"].strip()
    base_url = DATA_PROVIDERS["TWELVEDATA"]["base_url"]

    if not api_key:
        raise ValueError("TwelveData API key is not configured.")

    if symbol in TWELVEDATA_SYMBOL_MAP:
        td_symbol = TWELVEDATA_SYMBOL_MAP[symbol]
    else:
        if len(symbol) == 6:
            td_symbol = f"{symbol[:3]}/{symbol[3:]}"
        else:
            td_symbol = symbol

    params = {
        "symbol": td_symbol,
        "interval": timeframe,
        "outputsize": 200,
        "apikey": api_key,
    }

    url = f"{base_url}/time_series"
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    if "status" in data and data["status"] == "error":
        raise ValueError(f"TwelveData error: {data.get('message')}")

    if "values" not in data:
        raise ValueError(f"Unexpected TwelveData response: {data}")

    values = data["values"]
    df = pd.DataFrame(values)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)

    return df


def fetch_from_alphavantage_intraday(pair: str, interval: str = "60min") -> pd.DataFrame:
    api_key = DATA_PROVIDERS["ALPHAVANTAGE"]["api_key"].strip()
    base_url = DATA_PROVIDERS["ALPHAVANTAGE"]["base_url"]

    if not api_key:
        raise ValueError("Alpha Vantage API key is not configured.")

    from_symbol = pair[:3]
    to_symbol = pair[3:]

    params = {
        "function": "FX_INTRADAY",
        "from_symbol": from_symbol,
        "to_symbol": to_symbol,
        "interval": interval,
        "outputsize": "compact",
        "apikey": api_key,
    }

    r = requests.get(base_url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    if "Information" in data:
        raise ValueError(f"Alpha Vantage intraday info: {data['Information']}")

    key_candidates = [k for k in data.keys() if "Time Series" in k]
    if not key_candidates:
        raise ValueError(f"Unexpected Alpha Vantage response: {data}")
    key = key_candidates[0]

    ts = data[key]
    df = (
        pd.DataFrame(ts).T
        .rename(
            columns={
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
            }
        )
        .astype(float)
    )
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def fetch_from_alphavantage_daily(pair: str) -> pd.DataFrame:
    api_key = DATA_PROVIDERS["ALPHAVANTAGE"]["api_key"].strip()
    base_url = DATA_PROVIDERS["ALPHAVANTAGE"]["base_url"]

    if not api_key:
        raise ValueError("Alpha Vantage API key is not configured.")

    from_symbol = pair[:3]
    to_symbol = pair[3:]

    params = {
        "function": "FX_DAILY",
        "from_symbol": from_symbol,
        "to_symbol": to_symbol,
        "outputsize": "compact",
        "apikey": api_key,
    }

    r = requests.get(base_url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    key_candidates = [k for k in data.keys() if "Time Series" in k]
    if not key_candidates:
        raise ValueError(f"Unexpected Alpha Vantage response: {data}")
    key = key_candidates[0]

    ts = data[key]
    df = (
        pd.DataFrame(ts).T
        .rename(
            columns={
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
            }
        )
        .astype(float)
    )
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def fetch_market_data(symbol: str, timeframe: str = DEFAULT_TIMEFRAME) -> pd.DataFrame:
    provider = get_active_provider_for_symbol(symbol)
    if provider is None:
        raise ValueError(
            "No data provider configured for this symbol. "
            "Please set at least one API key in DATA_PROVIDERS."
        )

    if provider == "TWELVEDATA":
        return fetch_from_twelvedata(symbol, timeframe=timeframe)

    # AlphaVantage for FX only
    try:
        return fetch_from_alphavantage_intraday(symbol, interval="60min")
    except Exception as e:
        logger.warning("Alpha Vantage intraday failed, falling back to daily: %s", e)
        df_daily = fetch_from_alphavantage_daily(symbol)
        return df_daily

# ===========================
# ANALYSIS LOGIC (ULTIMATE)
# ===========================
def detect_trend_from_close(close: pd.Series) -> str:
    if len(close) < 100:
        return "unclear"

    ema50 = close.ewm(span=50).mean()
    ema100 = close.ewm(span=100).mean()

    c = close.iloc[-1]
    e50 = ema50.iloc[-1]
    e100 = ema100.iloc[-1]

    if c > e50 > e100:
        return "bullish"
    elif c < e50 < e100:
        return "bearish"
    else:
        return "sideways"


def compute_confidence(bias: str, reasons: list[str]) -> int:
    score = 50

    for r in reasons:
        rl = r.lower()
        if "uptrend" in rl or "bullish" in rl:
            score += 5
        if "downtrend" in rl or "bearish" in rl:
            score += 5
        if "overbought" in rl or "oversold" in rl:
            score += 5
        if "crossover" in rl:
            score += 5
        if "alignment" in rl or "structure" in rl:
            score += 3
        if "divergence" in rl:
            score += 4
        if "rejection" in rl or "wick" in rl:
            score += 3

    if bias.upper() in ("BUY", "SELL"):
        score += 5
    else:
        score -= 5

    return max(0, min(100, score))


def detect_stagnation(df: pd.DataFrame) -> dict:
    if len(df) < 2:
        return {"detected": False, "minutes": 0.0, "range_pct": 0.0}

    window = df.tail(STAGNATION_WINDOW_CANDLES)
    if len(window) < 2:
        return {"detected": False, "minutes": 0.0, "range_pct": 0.0}

    high = float(window["high"].max())
    low = float(window["low"].min())
    last_price = float(window["close"].iloc[-1])

    if last_price <= 0:
        return {"detected": False, "minutes": 0.0, "range_pct": 0.0}

    range_pct = (high - low) / last_price * 100.0
    t0 = window.index[0]
    t1 = window.index[-1]
    minutes = max(0.0, (t1 - t0).total_seconds() / 60.0)

    detected = range_pct < STAGNATION_RANGE_THRESHOLD_PCT

    return {
        "detected": detected,
        "minutes": minutes,
        "range_pct": range_pct,
    }


def detect_candle_pattern(df: pd.DataFrame) -> tuple[str, str]:
    """
    Very simple candle pattern detection:
    - Bullish/Bearish Engulfing
    - Hammer / Shooting Star
    Returns (pattern_name, pattern_bias).
    """
    if len(df) < 2:
        return "", "neutral"

    last = df.iloc[-1]
    prev = df.iloc[-2]

    o1, h1, l1, c1 = prev["open"], prev["high"], prev["low"], prev["close"]
    o2, h2, l2, c2 = last["open"], last["high"], last["low"], last["close"]

    body1 = abs(c1 - o1)
    body2 = abs(c2 - o2)
    upper_wick2 = h2 - max(o2, c2)
    lower_wick2 = min(o2, c2) - l2

    # Avoid division by zero
    body2_nonzero = body2 if body2 != 0 else 1e-8

    # Engulfing
    bullish_engulf = (c1 < o1) and (c2 > o2) and (body2 > body1) and (o2 < c1) and (c2 > o1)
    bearish_engulf = (c1 > o1) and (c2 < o2) and (body2 > body1) and (o2 > c1) and (c2 < o1)

    # Hammer: small body, long lower wick
    hammer = (body2 < (h2 - l2) * 0.3) and (lower_wick2 > body2 * 2) and (upper_wick2 < body2)
    # Shooting star: small body, long upper wick
    shooting_star = (body2 < (h2 - l2) * 0.3) and (upper_wick2 > body2 * 2) and (lower_wick2 < body2)

    if bullish_engulf:
        return "Bullish Engulfing", "bullish"
    if bearish_engulf:
        return "Bearish Engulfing", "bearish"
    if hammer:
        return "Hammer", "bullish"
    if shooting_star:
        return "Shooting Star", "bearish"

    return "", "neutral"


def detect_simple_divergence(df: pd.DataFrame) -> tuple[bool, bool]:
    """
    Simple RSI divergence detection over last 10 candles.
    Returns (bullish_divergence, bearish_divergence).
    """
    if len(df) < 11 or "rsi" not in df.columns:
        return False, False

    recent = df.tail(10)
    price_series = recent["close"]
    rsi_series = recent["rsi"]

    last_price = float(price_series.iloc[-1])
    last_rsi = float(rsi_series.iloc[-1])

    prev_price_high = float(price_series.iloc[:-1].max())
    prev_price_low = float(price_series.iloc[:-1].min())
    prev_rsi_high = float(rsi_series.iloc[:-1].max())
    prev_rsi_low = float(rsi_series.iloc[:-1].min())

    bullish_div = (last_price < prev_price_low) and (last_rsi > prev_rsi_low)
    bearish_div = (last_price > prev_price_high) and (last_rsi < prev_rsi_high)

    return bullish_div, bearish_div


def ai_evaluate_signal(features: dict, rule_decision: str, rule_reasons: list[str]) -> dict:
    if not OPENAI_API_KEY:
        confidence = compute_confidence(rule_decision, rule_reasons)
        return {
            "decision": rule_decision,
            "confidence": confidence,
            "explanation": "AI module not configured. Using rule-based technical logic only.",
            "risk": "Treat this as a standard technical setup without AI confirmation.",
        }

    client = OpenAI(api_key=OPENAI_API_KEY)

    payload = {
        "features": features,
        "rule_based_decision": rule_decision,
        "rule_based_reasons": rule_reasons,
    }

    system_msg = (
        "You are a professional trading analyst. "
        "You receive technical indicator data and a rule-based suggestion. "
        "You must respond ONLY in valid JSON with fields: "
        "`decision` (BUY/SELL/NO TRADE), `confidence` (0-100), "
        "`explanation` (1-3 sentences), `risk` (one short sentence). "
        "Base your reasoning ONLY on the given data."
    )

    user_msg = (
        "Here is the current market technical snapshot:\n"
        + json.dumps(payload, ensure_ascii=False)
    )

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content.strip()

        # Be robust about extra text around JSON
        json_str = content
        if not json_str.startswith("{"):
            first = json_str.find("{")
            last = json_str.rfind("}")
            if first != -1 and last != -1 and last > first:
                json_str = json_str[first:last+1]

        data = json.loads(json_str)

        decision = str(data.get("decision", rule_decision)).upper()
        if decision not in ("BUY", "SELL", "NO TRADE"):
            decision = rule_decision

        confidence_raw = data.get("confidence", None)
        if confidence_raw is None:
            confidence = compute_confidence(decision, rule_reasons)
        else:
            try:
                confidence = int(confidence_raw)
            except Exception:
                confidence = compute_confidence(decision, rule_reasons)
        confidence = max(0, min(100, confidence))

        explanation = str(data.get("explanation", "AI explanation not available."))
        risk = str(data.get("risk", "Risk not specified by AI."))

        return {
            "decision": decision,
            "confidence": confidence,
            "explanation": explanation,
            "risk": risk,
        }
    except Exception as e:
        logger.error("AI evaluation error: %s", e)
        confidence = compute_confidence(rule_decision, rule_reasons)
        return {
            "decision": rule_decision,
            "confidence": confidence,
            "explanation": "AI module could not process this snapshot. Using rule-based technical logic only.",
            "risk": "Treat this as a normal technical setup without AI confirmation.",
        }


def choose_expiry_minutes(final_decision: str, confidence: int) -> int:
    """
    Hybrid expiry 1–5 minutes ONLY.
    """
    if final_decision == "NO TRADE":
        return 3

    if confidence >= 85:
        return 5
    elif confidence >= 70:
        return 4
    elif confidence >= 55:
        return 3
    elif confidence >= 40:
        return 2
    else:
        return 1

# ===========================
# TECHNICAL SNAPSHOT (ULTIMATE)
# ===========================
def compute_technical_snapshot(df: pd.DataFrame, symbol: str) -> dict:
    df = df.copy()
    if len(df) < 120:
        raise ValueError("Not enough candles to analyse this market yet.")

    close = df["close"]

    # Indicators
    df["rsi"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    df["ema_20"] = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(close=close, window=50).ema_indicator()
    df["ema_100"] = ta.trend.EMAIndicator(close=close, window=100).ema_indicator()
    df["ema_200"] = ta.trend.EMAIndicator(close=close, window=200).ema_indicator()

    macd_obj = ta.trend.MACD(close=close)
    df["macd"] = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()

    atr_obj = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=close, window=14
    )
    df["atr"] = atr_obj.average_true_range()

    last = df.iloc[-1]

    price = float(last["close"])
    rsi = float(last["rsi"])
    ema_20 = float(last["ema_20"])
    ema_50 = float(last["ema_50"])
    ema_100 = float(last["ema_100"])
    ema_200 = float(last["ema_200"])
    macd_val = float(last["macd"])
    macd_sig = float(last["macd_signal"])
    atr = float(last["atr"])

    atr_pct = (atr / price * 100.0) if price > 0 else 0.0

    # S/R from recent candles
    recent = df.tail(60)
    recent_high = float(recent["high"].max())
    recent_low = float(recent["low"].min())

    # Trend
    trend = detect_trend_from_close(df["close"])

    # Stagnation
    stagnation_info = detect_stagnation(df)

    # Long-term bias (approx "higher timeframe")
    if price > ema_100 > ema_200:
        long_bias = "bullish"
    elif price < ema_100 < ema_200:
        long_bias = "bearish"
    else:
        long_bias = "neutral"

    # Short-term momentum bias using ema_20 slope
    short_bias = "neutral"
    if len(df) >= 25:
        ema_20_series = df["ema_20"]
        ema_20_prev = float(ema_20_series.iloc[-5])
        slope = ema_20 - ema_20_prev
        if slope > 0:
            short_bias = "bullish"
        elif slope < 0:
            short_bias = "bearish"

    # Candle pattern
    pattern_name, pattern_bias = detect_candle_pattern(df)

    # Divergence
    bullish_div, bearish_div = detect_simple_divergence(df)

    # Rule-based decision with ultimate filters
    rule_decision = "NO TRADE"
    rule_reasons: list[str] = []
    buy_checks = {}
    sell_checks = {}

    # Volatility filter
    vol_ok = MIN_ATR_PCT <= atr_pct <= MAX_ATR_PCT
    if not vol_ok:
        rule_reasons.append(
            f"Volatility ({atr_pct:.3f}%) is outside the preferred range ({MIN_ATR_PCT}–{MAX_ATR_PCT}%)."
        )

    # Stagnation
    if stagnation_info["detected"]:
        minutes = stagnation_info["minutes"]
        rng = stagnation_info["range_pct"]
        rule_reasons.append(
            f"Market has moved only {rng:.4f}% over ~{minutes:.0f} minutes → price action is stagnant."
        )

    # BUY conditions (multi-filter)
    buy_checks = {
        "Price above EMA50 & EMA100 (bullish structure)": price > ema_50 > ema_100,
        "Higher-timeframe bias bullish": long_bias == "bullish",
        "Short-term momentum up": short_bias == "bullish",
        "RSI below 70 (not extreme overbought)": rsi < 70,
        "MACD above signal (bullish momentum)": macd_val > macd_sig,
        "Price above recent low (not at extreme bottom)": price > recent_low,
        "No bearish divergence": not bearish_div,
        "Pattern not strongly bearish": pattern_bias != "bearish",
        "Volatility acceptable": vol_ok,
        "No strong stagnation": not stagnation_info["detected"],
    }

    # SELL conditions (multi-filter)
    sell_checks = {
        "Price below EMA50 & EMA100 (bearish structure)": price < ema_50 < ema_100,
        "Higher-timeframe bias bearish": long_bias == "bearish",
        "Short-term momentum down": short_bias == "bearish",
        "RSI above 30 (not extreme oversold)": rsi > 30,
        "MACD below signal (bearish momentum)": macd_val < macd_sig,
        "Price below recent high (not at extreme top)": price < recent_high,
        "No bullish divergence": not bullish_div,
        "Pattern not strongly bullish": pattern_bias != "bullish",
        "Volatility acceptable": vol_ok,
        "No strong stagnation": not stagnation_info["detected"],
    }

    buy_conditions = all(buy_checks.values())
    sell_conditions = all(sell_checks.values())

    buy_fails = [name for name, ok in buy_checks.items() if not ok]
    sell_fails = [name for name, ok in sell_checks.items() if not ok]

    if buy_conditions and not sell_conditions:
        rule_decision = "BUY"
        rule_reasons.append("All bullish filters aligned across structure, momentum and volatility.")
        if pattern_bias == "bullish":
            rule_reasons.append(f"Candle pattern supports bullish bias ({pattern_name}).")
        if bullish_div:
            rule_reasons.append("Bullish RSI divergence supports potential upside continuation.")
    elif sell_conditions and not buy_conditions:
        rule_decision = "SELL"
        rule_reasons.append("All bearish filters aligned across structure, momentum and volatility.")
        if pattern_bias == "bearish":
            rule_reasons.append(f"Candle pattern supports bearish bias ({pattern_name}).")
        if bearish_div:
            rule_reasons.append("Bearish RSI divergence supports potential downside continuation.")
    else:
        rule_decision = "NO TRADE"
        rule_reasons.append("Filters are not aligned enough for a high-probability BUY or SELL setup.")

    features = {
        "symbol": symbol,
        "timeframe": "H1",
        "price": price,
        "rsi": rsi,
        "ema_20": ema_20,
        "ema_50": ema_50,
        "ema_100": ema_100,
        "ema_200": ema_200,
        "macd": macd_val,
        "macd_signal": macd_sig,
        "atr": atr,
        "atr_pct": atr_pct,
        "recent_high": recent_high,
        "recent_low": recent_low,
        "trend": trend,
        "stagnation_minutes": stagnation_info["minutes"],
        "stagnation_range_pct": stagnation_info["range_pct"],
        "stagnation_detected": stagnation_info["detected"],
        "long_bias": long_bias,
        "short_bias": short_bias,
        "pattern_name": pattern_name,
        "pattern_bias": pattern_bias,
        "bullish_divergence": bullish_div,
        "bearish_divergence": bearish_div,
        "volatility_ok": vol_ok,
    }

    return {
        "df": df,
        "price": price,
        "rsi": rsi,
        "ema_20": ema_20,
        "ema_50": ema_50,
        "ema_100": ema_100,
        "ema_200": ema_200,
        "macd_val": macd_val,
        "macd_sig": macd_sig,
        "atr": atr,
        "atr_pct": atr_pct,
        "recent_high": recent_high,
        "recent_low": recent_low,
        "trend": trend,
        "stagnation": stagnation_info,
        "long_bias": long_bias,
        "short_bias": short_bias,
        "pattern_name": pattern_name,
        "pattern_bias": pattern_bias,
        "bullish_divergence": bullish_div,
        "bearish_divergence": bearish_div,
        "buy_conditions": buy_conditions,
        "sell_conditions": sell_conditions,
        "buy_fails": buy_fails,
        "sell_fails": sell_fails,
        "rule_decision": rule_decision,
        "rule_reasons": rule_reasons,
        "features": features,
    }

# ===========================
# BINARY-STYLE SIGNAL GENERATION
# ===========================
def generate_binary_signal_from_df(df: pd.DataFrame, symbol: str) -> str:
    snap = compute_technical_snapshot(df, symbol)

    price = snap["price"]
    rsi = snap["rsi"]
    ema_50 = snap["ema_50"]
    ema_100 = snap["ema_100"]
    macd_val = snap["macd_val"]
    macd_sig = snap["macd_sig"]
    atr = snap["atr"]
    recent_high = snap["recent_high"]
    recent_low = snap["recent_low"]
    trend = snap["trend"]
    atr_pct = snap["atr_pct"]
    stagnation_info = snap["stagnation"]
    pattern_name = snap["pattern_name"]
    pattern_bias = snap["pattern_bias"]
    bullish_div = snap["bullish_divergence"]
    bearish_div = snap["bearish_divergence"]
    rule_decision = snap["rule_decision"]
    rule_reasons = snap["rule_reasons"]
    features = snap["features"]

    # AI refinement
    ai_result = ai_evaluate_signal(features, rule_decision, rule_reasons)
    final_decision = ai_result["decision"]
    confidence = ai_result["confidence"]
    ai_explanation = ai_result["explanation"]
    ai_risk = ai_result["risk"]

    # Expiry minutes (1–5)
    expiry_minutes = choose_expiry_minutes(final_decision, confidence)

    now_local = wat_now()
    entry_dt = now_local + timedelta(minutes=5)  # 5-minute future entry
    expiry_dt = entry_dt + timedelta(minutes=expiry_minutes)

    entry_str = f"{format_time_local(entry_dt)} {BOT_TZ_NAME}"
    expiry_str = f"{format_time_local(expiry_dt)} {BOT_TZ_NAME}"

    # Martingale times
    m1_dt = expiry_dt
    m2_dt = expiry_dt + timedelta(minutes=1)
    m3_dt = expiry_dt + timedelta(minutes=2)

    m1_str = f"{format_time_local(m1_dt)} {BOT_TZ_NAME}"
    m2_str = f"{format_time_local(m2_dt)} {BOT_TZ_NAME}"
    m3_str = f"{format_time_local(m3_dt)} {BOT_TZ_NAME}"

    weekend_otc = is_weekend_utc()
    pair_name, flags = pretty_pair_name(symbol, weekend_otc)

    lines: list[str] = []

    # Header with pair & flags
    if flags:
        lines.append(f"📈 {pair_name} {flags}")
    else:
        lines.append(f"📈 {pair_name}")
    lines.append("")

    # NO TRADE case
    if final_decision == "NO TRADE":
        lines.append(f"⏱ Timeframe: {DEFAULT_TIMEFRAME} analysis")
        lines.append(f"🤖 AI Confidence: {confidence}%")
        lines.append(f"🕒 Checked Time: {format_time_local(now_local)} {BOT_TZ_NAME}")
        lines.append("")
        lines.append("🚫 NO TRADE")
        lines.append("")
        lines.append("📌 Reason(s):")
        for r in rule_reasons:
            lines.append(f"• {r}")
        if stagnation_info["detected"]:
            minutes = stagnation_info["minutes"]
            rng = stagnation_info["range_pct"]
            lines.append(
                f"• Price has stayed within ~{rng:.4f}% over ~{minutes:.0f} minutes → very little movement."
            )
        if pattern_name:
            lines.append(f"• Candle pattern: {pattern_name} ({pattern_bias})")
        if bullish_div:
            lines.append("• Bullish divergence present (not ideal for clean direction yet).")
        if bearish_div:
            lines.append("• Bearish divergence present (not ideal for clean direction yet).")
        lines.append(f"• AI note: {ai_explanation}")
        lines.append("")
        lines.append("⚠️ Wait for a cleaner move before entering.")
        return "\n".join(lines)

    # BUY or SELL
    lines.append(f"⏱ Timeframe: {expiry_minutes}-min expiry")
    lines.append(f"🤖 AI Confidence: {confidence}%")
    lines.append(f"🕒 Entry Time: {entry_str}")
    lines.append(f"⌛ Expiry: {expiry_str}")
    lines.append("")

    if final_decision == "BUY":
        lines.append("🟩 BUY")
    elif final_decision == "SELL":
        lines.append("🟥 SELL")
    else:
        lines.append(f"📌 {final_decision}")

    lines.append("📊 Martingale Levels:")
    lines.append(f"• Level 1 → {m1_str}")
    lines.append(f"• Level 2 → {m2_str}")
    lines.append(f"• Level 3 → {m3_str}")
    lines.append("")

    # Short technical footnote
    lines.append("ℹ️ Technical snapshot:")
    lines.append(f"• Price: {price:.5f}")
    lines.append(f"• Trend: {trend}")
    lines.append(f"• ATR: {atr:.5f} ({atr_pct:.3f}%)")
    lines.append(f"• RSI(14): {rsi:.2f}")
    lines.append(f"• EMA50 / EMA100: {ema_50:.5f} / {ema_100:.5f}")
    lines.append(f"• MACD vs Signal: {macd_val:.5f} / {macd_sig:.5f}")
    lines.append(f"• Recent low / high: {recent_low:.5f} / {recent_high:.5f}")
    if pattern_name:
        lines.append(f"• Candle pattern: {pattern_name} ({pattern_bias})")
    if bullish_div:
        lines.append("• Bullish divergence detected.")
    if bearish_div:
        lines.append("• Bearish divergence detected.")
    if stagnation_info["detected"]:
        minutes = stagnation_info["minutes"]
        rng = stagnation_info["range_pct"]
        lines.append(
            f"• Recent movement: ~{rng:.4f}% over ~{minutes:.0f} minutes (low volatility)."
        )
    lines.append("")
    lines.append(f"🤖 AI note: {ai_explanation}")
    lines.append("⚠️ Use strict risk management. This is not financial advice.")

    return "\n".join(lines)


def create_chart_image(df: pd.DataFrame, symbol: str) -> io.BytesIO:
    df = df.copy()
    close = df["close"]

    df["ema_50"] = ta.trend.EMAIndicator(close=close, window=50).ema_indicator()
    df["ema_100"] = ta.trend.EMAIndicator(close=close, window=100).ema_indicator()

    last = df.tail(150)

    plt.figure(figsize=(9, 4))
    plt.plot(last.index, last["close"], label="Close")
    plt.plot(last.index, last["ema_50"], label="EMA 50")
    plt.plot(last.index, last["ema_100"], label="EMA 100")
    plt.title(f"{symbol} — Price with EMA50 & EMA100")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return buf


async def get_signal_for_symbol(symbol: str, source: str = "manual") -> str:
    try:
        df = fetch_market_data(symbol, timeframe=DEFAULT_TIMEFRAME)
        msg = generate_binary_signal_from_df(df, symbol)
        log_signal(symbol, source)
        return msg
    except Exception as e:
        logger.exception("Error generating signal")
        return f"❌ Error generating signal for {symbol}: {e}"

# ===========================
# TELEGRAM HANDLERS — COMMANDS
# ===========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    premium = is_premium(user.id)
    reply_markup = build_main_menu(premium)

    if premium:
        txt = f"""
Hi {user.first_name or ''} 👋

You are already *activated* as a premium user. ✅

I can:
• Show supported markets → /pairs  
• Give binary-style AI signals → `/signal`  
• Show EMA chart snapshots → `/chart`  
• Manage auto signals → `/subscribe` and `/unsubscribe`  
• Debug a symbol's logic → `/debugsymbol ETHUSD`  

On weekends, I label signals as *OTC* and keep the style ready for binary trading.
"""
    else:
        txt = f"""
Hi {user.first_name or ''} 👋

Welcome to the *Pro Binary & OTC AI Bot*.

Features:
• Forex, Crypto, Indices, Commodities & Stocks  
• Binary-style signals with Entry Time, Expiry & Martingale levels  
• AI-backed direction & confidence  
• Auto signals per user  

🔐 To activate premium access:
1. Tap `/register`
2. Enter your premium code
"""

    await update.message.reply_markdown(txt, reply_markup=reply_markup)


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    premium = is_premium(user.id)
    reply_markup = build_main_menu(premium)

    txt = f"""
*Help — Signal_{SIGNAL_VERSION}*

/start - Welcome & status
/help - This help message
/pairs - Show all supported markets (grouped by category)
/signal - Open signal selector menu (binary-style)
/chart - Open chart selector menu (EMA50/EMA100)
/subscribe - Subscribe to auto signals
/unsubscribe - Unsubscribe from auto signals
/list - List your current subscriptions
/register - Enter your premium access code
/debugsymbol SYMBOL - Debug full technical view (e.g. `/debugsymbol ETHUSD`)

Entry Time is always *5 minutes ahead* in *{BOT_TZ_NAME}* and Expiry uses a 1–5 minute hybrid logic.
"""
    await update.message.reply_markdown(txt, reply_markup=reply_markup)


async def pairs_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    premium = is_premium(user.id)
    reply_markup = build_main_menu(premium)

    lines = ["*Supported markets (grouped):*"]
    for cat, items in MARKET_CATEGORIES.items():
        if not items:
            continue
        lines.append(f"\n*{cat}:*")
        lines.append(", ".join(sorted(items)))

    await update.message.reply_markdown("\n".join(lines), reply_markup=reply_markup)

# ===========================
# INLINE KEYBOARDS
# ===========================
def build_category_rows(items: list[str], prefix: str, per_row: int = 3) -> list[list[InlineKeyboardButton]]:
    rows: list[list[InlineKeyboardButton]] = []
    row: list[InlineKeyboardButton] = []
    for sym in sorted(items):
        row.append(InlineKeyboardButton(sym, callback_data=f"{prefix}|{sym}"))
        if len(row) == per_row:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    return rows


def build_signal_inline_keyboard() -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    for cat, items in MARKET_CATEGORIES.items():
        if not items:
            continue
        rows.append([InlineKeyboardButton(f"— {cat} —", callback_data="noop")])
        rows.extend(build_category_rows(items, prefix="sig", per_row=3))
    return InlineKeyboardMarkup(rows)


def build_subscribe_keyboard(user_id: int) -> InlineKeyboardMarkup:
    user_id_str = str(user_id)
    current = set(subscriptions.get(user_id_str, []))
    rows: list[list[InlineKeyboardButton]] = []
    for cat, items in MARKET_CATEGORIES.items():
        available = [s for s in items if s not in current]
        if not available:
            continue
        rows.append([InlineKeyboardButton(f"— {cat} —", callback_data="noop")])
        rows.extend(build_category_rows(available, prefix="sub", per_row=3))
    if not rows:
        rows = [[InlineKeyboardButton("No available markets (all subscribed).", callback_data="noop")]]
    return InlineKeyboardMarkup(rows)


def build_unsubscribe_keyboard(user_id: int) -> InlineKeyboardMarkup:
    user_id_str = str(user_id)
    current = subscriptions.get(user_id_str, [])
    rows: list[list[InlineKeyboardButton]] = []
    if not current:
        rows = [[InlineKeyboardButton("No active subscriptions.", callback_data="noop")]]
        return InlineKeyboardMarkup(rows)

    rows.append([InlineKeyboardButton("— Your Subscriptions —", callback_data="noop")])
    rows.extend(build_category_rows(current, prefix="unsub", per_row=3))
    return InlineKeyboardMarkup(rows)

# ===========================
# COMMAND HANDLERS
# ===========================
async def signal_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await ensure_premium(update, context):
        return

    kb = build_signal_inline_keyboard()
    await update.message.reply_markdown(
        "🔍 *Select a market to analyse (binary-style signal):*",
        reply_markup=kb,
    )


async def chart_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await ensure_premium(update, context):
        return

    rows: list[list[InlineKeyboardButton]] = []
    for cat, items in MARKET_CATEGORIES.items():
        if not items:
            continue
        rows.append([InlineKeyboardButton(f"— {cat} —", callback_data="noop")])
        rows.extend(build_category_rows(items, prefix="chart", per_row=3))

    kb = InlineKeyboardMarkup(rows)
    await update.message.reply_markdown(
        "📈 *Select a market to view EMA chart:*",
        reply_markup=kb,
    )


async def subscribe_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await ensure_premium(update, context):
        return

    user = update.effective_user
    kb = build_subscribe_keyboard(user.id)
    await update.message.reply_markdown(
        "✅ *Select a market to subscribe to auto signals:*",
        reply_markup=kb,
    )


async def unsubscribe_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await ensure_premium(update, context):
        return

    user = update.effective_user
    kb = build_unsubscribe_keyboard(user.id)
    await update.message.reply_markdown(
        "❌ *Select a market to start unsubscribe process:*",
        reply_markup=kb,
    )


async def list_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await ensure_premium(update, context):
        return

    user = update.effective_user
    user_id_str = str(user.id)
    current = subscriptions.get(user_id_str, [])

    premium = is_premium(user.id)
    reply_markup = build_main_menu(premium)

    if not current:
        await update.message.reply_markdown(
            "You don't have any active subscriptions yet.\nUse `/subscribe` to add some.",
            reply_markup=reply_markup,
        )
        return

    pairs_str = ", ".join(sorted(current))
    await update.message.reply_markdown(
        f"*Your active auto signals:*\n{pairs_str}",
        reply_markup=reply_markup,
    )


async def register_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id_str = str(user.id)

    if is_premium(user.id):
        reply_markup = build_main_menu(True)
        await update.message.reply_markdown(
            "✅ You are already registered as premium.",
            reply_markup=reply_markup,
        )
        return

    context.user_data["awaiting_premium_code"] = True

    reply_markup = build_main_menu(False)
    await update.message.reply_markdown(
        "🔑 Please send your premium code as a message now.",
        reply_markup=reply_markup,
    )

# ===========================
# DEBUG SYMBOL COMMAND
# ===========================
async def debugsymbol_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await ensure_premium(update, context):
        return

    user = update.effective_user
    premium = is_premium(user.id)
    reply_markup = build_main_menu(premium)

    if not context.args:
        await update.message.reply_markdown(
            "Usage: `/debugsymbol ETHUSD`",
            reply_markup=reply_markup,
        )
        return

    symbol = context.args[0].upper()
    if symbol not in ALL_SYMBOLS:
        await update.message.reply_markdown(
            f"❌ Unknown symbol: *{symbol}*",
            reply_markup=reply_markup,
        )
        return

    try:
        df = fetch_market_data(symbol, timeframe=DEFAULT_TIMEFRAME)
        snap = compute_technical_snapshot(df, symbol)
        ai_result = ai_evaluate_signal(
            snap["features"], snap["rule_decision"], snap["rule_reasons"]
        )

        stagnation = snap["stagnation"]
        minutes = stagnation["minutes"]
        range_pct = stagnation["range_pct"]

        text_lines = []
        text_lines.append(f"🔍 *Debug for {symbol}*")
        text_lines.append("")
        text_lines.append(f"Stagnation detected: *{'Yes' if stagnation['detected'] else 'No'}*")
        text_lines.append(f"Stagnation window: ~{minutes:.0f} minutes")
        text_lines.append(f"Price range in window: {range_pct:.4f}%")
        text_lines.append("")
        text_lines.append(f"ATR%: {snap['atr_pct']:.3f}% (min {MIN_ATR_PCT}, max {MAX_ATR_PCT})")
        text_lines.append(f"Long bias: *{snap['long_bias']}*  |  Short bias: *{snap['short_bias']}*")
        text_lines.append(f"Pattern: {snap['pattern_name']} ({snap['pattern_bias']})")
        text_lines.append(f"Bullish divergence: *{snap['bullish_divergence']}*")
        text_lines.append(f"Bearish divergence: *{snap['bearish_divergence']}*")
        text_lines.append("")
        text_lines.append(f"Rule-based decision: *{snap['rule_decision']}*")
        text_lines.append("Rule-based reasons:")
        for r in snap["rule_reasons"]:
            text_lines.append(f"• {r}")
        text_lines.append("")
        text_lines.append(f"BUY conditions pass: *{snap['buy_conditions']}*")
        if snap["buy_fails"]:
            text_lines.append("BUY failed checks:")
            for fdesc in snap["buy_fails"]:
                text_lines.append(f"• {fdesc}")
        text_lines.append("")
        text_lines.append(f"SELL conditions pass: *{snap['sell_conditions']}*")
        if snap["sell_fails"]:
            text_lines.append("SELL failed checks:")
            for fdesc in snap["sell_fails"]:
                text_lines.append(f"• {fdesc}")
        text_lines.append("")
        text_lines.append(f"AI final decision: *{ai_result['decision']}*")
        text_lines.append(f"AI confidence: *{ai_result['confidence']}%*")
        text_lines.append(f"AI explanation: {ai_result['explanation']}")
        text_lines.append(f"AI risk note: {ai_result['risk']}")

        await update.message.reply_markdown(
            "\n".join(text_lines),
            reply_markup=reply_markup,
        )

    except Exception as e:
        logger.exception("Error in debugsymbol")
        await update.message.reply_markdown(
            f"❌ Error debugging {symbol}: {e}",
            reply_markup=reply_markup,
        )

# ===========================
# CALLBACK ROUTER
# ===========================
async def callback_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data = query.data or ""
    user_id = query.from_user.id
    chat_id = query.message.chat_id
    user_id_str = str(user_id)

    if data == "noop":
        return

    # SIGNAL from inline menu
    if data.startswith("sig|"):
        symbol = data.split("|", 1)[1]
        if symbol not in ALL_SYMBOLS:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"❌ Unknown symbol: {symbol}",
                reply_markup=build_main_menu(is_premium(user_id)),
            )
            return

        try:
            await query.edit_message_text(
                text=f"⏳ Generating signal for *{symbol}*...\nPlease wait...",
                parse_mode="Markdown",
            )
        except Exception:
            pass

        text = await get_signal_for_symbol(symbol, source="manual")
        await context.bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode="Markdown",
            reply_markup=build_main_menu(is_premium(user_id)),
        )
        return

    # CHART from inline menu
    if data.startswith("chart|"):
        symbol = data.split("|", 1)[1]
        if symbol not in ALL_SYMBOLS:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"❌ Unknown symbol: {symbol}",
                reply_markup=build_main_menu(is_premium(user_id)),
            )
            return
        try:
            await query.edit_message_text(
                text=f"⏳ Building chart for *{symbol}*...",
                parse_mode="Markdown",
            )
        except Exception:
            pass

        try:
            df = fetch_market_data(symbol, timeframe=DEFAULT_TIMEFRAME)
            buf = create_chart_image(df, symbol)
            await context.bot.send_photo(
                chat_id=chat_id,
                photo=buf,
                caption=f"{symbol} — {DEFAULT_TIMEFRAME} chart with EMA50 & EMA100",
                reply_markup=build_main_menu(is_premium(user_id)),
            )
        except Exception as e:
            logger.exception("Error generating chart")
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"❌ Error generating chart for {symbol}: {e}",
                reply_markup=build_main_menu(is_premium(user_id)),
            )
        return

    # SUBSCRIBE from inline menu
    if data.startswith("sub|"):
        symbol = data.split("|", 1)[1]
        current = subscriptions.get(user_id_str, [])
        if symbol not in current:
            current.append(symbol)
            subscriptions[user_id_str] = current
            save_subscriptions()
            await query.answer(text=f"Subscribed to {symbol}", show_alert=False)
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"✅ You are now subscribed to auto signals for *{symbol}*.",
                parse_mode="Markdown",
                reply_markup=build_main_menu(is_premium(user_id)),
            )
        else:
            await query.answer(text=f"Already subscribed to {symbol}", show_alert=False)
        return

    # UNSUBSCRIBE selection
    if data.startswith("unsub|"):
        symbol = data.split("|", 1)[1]
        current = subscriptions.get(user_id_str, [])
        if symbol not in current:
            await query.answer(text=f"Not subscribed to {symbol}", show_alert=False)
            return

        pending_unsub[user_id_str] = symbol
        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                f"❗ You requested to unsubscribe from *{symbol}*.\n\n"
                f"Type `YES` to confirm, or anything else to cancel."
            ),
            parse_mode="Markdown",
            reply_markup=build_main_menu(is_premium(user_id)),
        )
        return

# ===========================
# TELEGRAM HANDLERS — TEXT
# ===========================
async def text_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id_str = str(user.id)
    text = (update.message.text or "").strip()

    # Premium code flow
    if context.user_data.get("awaiting_premium_code"):
        if text == PREMIUM_ACCESS_CODE:
            premium_users.add(user_id_str)
            save_premium()
            context.user_data["awaiting_premium_code"] = False
            await update.message.reply_markdown(
                "✅ *Premium activated successfully!*\n\n"
                "You can now use /signal, /subscribe, /unsubscribe, /debugsymbol and more.",
                reply_markup=build_main_menu(True),
            )
        else:
            await update.message.reply_markdown(
                "❌ Invalid premium code. Please try again or contact the bot owner.",
                reply_markup=build_main_menu(False),
            )
        return

    # Unsubscribe YES confirmation
    if user_id_str in pending_unsub:
        symbol = pending_unsub[user_id_str]
        if text.upper() == "YES":
            current = subscriptions.get(user_id_str, [])
            if symbol in current:
                current = [s for s in current if s != symbol]
                subscriptions[user_id_str] = current
                save_subscriptions()
                await update.message.reply_markdown(
                    f"✅ You have unsubscribed from auto signals for *{symbol}*.",
                    reply_markup=build_main_menu(is_premium(user.id)),
                )
            else:
                await update.message.reply_markdown(
                    f"ℹ️ You were not subscribed to *{symbol}*.",
                    reply_markup=build_main_menu(is_premium(user.id)),
                )
        else:
            await update.message.reply_markdown(
                "❎ Unsubscribe cancelled.",
                reply_markup=build_main_menu(is_premium(user.id)),
            )
        pending_unsub.pop(user_id_str, None)
        return

    # Default: just guide user back to menu
    await update.message.reply_markdown(
        "Use the menu below to choose what to do next.",
        reply_markup=build_main_menu(is_premium(user.id)),
    )

# ===========================
# AUTO SIGNAL JOB
# ===========================
async def periodic_signals_job(context: ContextTypes.DEFAULT_TYPE):
    if not subscriptions:
        return

    all_symbols = set()
    for pairs in subscriptions.values():
        all_symbols.update(pairs)

    if not all_symbols:
        return

    pair_to_msg: dict[str, str] = {}

    for symbol in all_symbols:
        try:
            pair_to_msg[symbol] = await get_signal_for_symbol(symbol, source="auto")
            await asyncio.sleep(1.5)
        except Exception as e:
            pair_to_msg[symbol] = f"❌ Error generating auto signal for {symbol}: {e}"

    for user_id_str, pairs in subscriptions.items():
        chat_id = int(user_id_str)
        for symbol in pairs:
            msg = pair_to_msg.get(symbol)
            if not msg:
                continue
            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"🕒 *Auto signal update for {symbol}:*\n\n{msg}",
                    parse_mode="Markdown",
                    reply_markup=build_main_menu(is_premium(chat_id)),
                )
            except Exception as e:
                logger.error("Error sending auto signal to %s: %s", chat_id, e)

# ===========================
# MAIN ENTRY
# ===========================
def main():
    print(f"🚀 DEBUG: Pro Binary/OTC Bot (Signal_{SIGNAL_VERSION}) starting...")

    if BOT_TOKEN in ("", "YOUR_TELEGRAM_BOT_TOKEN_HERE"):
        print("❌ Please set BOT_TOKEN at the top of the file.")
        return

    load_data()

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # Command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("pairs", pairs_cmd))
    app.add_handler(CommandHandler("signal", signal_cmd))
    app.add_handler(CommandHandler("chart", chart_cmd))
    app.add_handler(CommandHandler("subscribe", subscribe_cmd))
    app.add_handler(CommandHandler("unsubscribe", unsubscribe_cmd))
    app.add_handler(CommandHandler("list", list_cmd))
    app.add_handler(CommandHandler("register", register_cmd))
    app.add_handler(CommandHandler("debugsymbol", debugsymbol_cmd))

    # Callback queries
    app.add_handler(CallbackQueryHandler(callback_router))

    # Text messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message_handler))

    # Auto signals every 60 minutes
    job_queue = app.job_queue
    if job_queue is None:
        print('⚠️ JobQueue not available. Install with: pip install "python-telegram-bot[job-queue]"')
    else:
        job_queue.run_repeating(periodic_signals_job, interval=60 * 60, first=60)

    logger.info(f"🚀 Pro Binary/OTC Bot Started Successfully (Signal_{SIGNAL_VERSION})!")
    app.run_polling()


if __name__ == "__main__":
    main()

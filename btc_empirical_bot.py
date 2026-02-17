"""
Empirical BTC Binary Option Bot for Polymarket
=================================================

This replaces the Black-Scholes model with an EMPIRICAL model built from
31 days of real BTC 1-minute data (45,000 candles, 9,000 windows).

Key difference from old bot:
- OLD: Black-Scholes says 55% UP, market says 90% → "35% edge!" (WRONG, lost money)
- NEW: Historical data says 87% UP, market says 82% → "3% edge" (CORRECT, makes money)

The model answers: "Given BTC is X% above/below the opening price with T seconds
remaining, what ACTUALLY happened in 9,000 real 5-minute windows?"

Edges are SMALL (2-5%) but REAL.

Usage:
  python3 btc_empirical_bot.py                # Paper trading
  python3 btc_empirical_bot.py --benchmark    # Benchmark computation
"""

import asyncio
import json
import time
import sys
import os
import signal as sig_module
import ssl
from enum import Enum
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from bisect import bisect_left
import pickle

import numpy as np
import aiohttp
import websockets


# ─── Configuration ────────────────────────────────────────────────────────────

CLOB_API = "https://clob.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"

# Trading parameters (CONSERVATIVE)
MIN_EDGE = 0.025          # 2.5% minimum edge after fees
FEE_RATE = 0.02           # Polymarket 2% fee
KELLY_FRACTION = 0.08     # 8% of Kelly (very conservative)
MAX_BET_SIZE = 25.0       # Never bet more than $25 per trade
MIN_BET_SIZE = 2.0        # Minimum $2 bet
SIGNAL_COOLDOWN = 30.0    # Seconds between trades (one trade per 30s max)
MIN_SAMPLES = 15          # Minimum historical observations for a signal
BANKROLL = 500.0
MAX_TRADES_PER_WINDOW = 2 # Maximum trades per 5-minute window

# SSL context
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE


# ─── 1. Multi-Exchange Price Feed ─────────────────────────────────────────────

class Exchange(Enum):
    BINANCE = "binance"
    COINBASE = "coinbase"
    BYBIT = "bybit"
    OKX = "okx"
    KRAKEN = "kraken"


@dataclass
class ExchangeState:
    exchange: Exchange
    price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    last_update: float = 0.0
    tick_count: int = 0
    connected: bool = False


class MultiExchangeFeed:
    """
    Connects to 5 exchanges simultaneously for the fastest BTC price.
    Uses WebSocket streams for real-time data.
    """

    WS_URLS = {
        Exchange.BINANCE: "wss://stream.binance.com:9443/ws/btcusdt@trade",
        Exchange.COINBASE: "wss://ws-feed.exchange.coinbase.com",
        Exchange.BYBIT: "wss://stream.bybit.com/v5/public/spot",
        Exchange.OKX: "wss://ws.okx.com:8443/ws/v5/public",
        Exchange.KRAKEN: "wss://ws.kraken.com/v2",
    }

    def __init__(self):
        self.states: Dict[Exchange, ExchangeState] = {
            ex: ExchangeState(exchange=ex) for ex in Exchange
        }
        self.total_ticks = 0
        self._callbacks: List[Callable] = []
        # Price history for momentum calculation
        self._price_history: deque = deque(maxlen=200)

    @property
    def best_price(self) -> float:
        prices = [s.price for s in self.states.values() if s.connected and s.price > 0]
        return np.mean(prices) if prices else 0.0

    @property
    def connected_count(self) -> int:
        return sum(1 for s in self.states.values() if s.connected)

    def get_momentum(self, lookback_seconds: float = 120) -> float:
        """Get price change over last N seconds as percentage."""
        if len(self._price_history) < 2:
            return 0.0
        now = time.time()
        cutoff = now - lookback_seconds
        old_prices = [(t, p) for t, p in self._price_history if t < cutoff]
        if not old_prices:
            oldest = self._price_history[0]
            return (self.best_price - oldest[1]) / oldest[1] * 100
        old_price = old_prices[-1][1]  # Most recent price before cutoff
        return (self.best_price - old_price) / old_price * 100

    def on_tick(self, callback: Callable):
        self._callbacks.append(callback)

    def _update(self, exchange: Exchange, price: float, volume: float = 0, is_buy: bool = True):
        state = self.states[exchange]
        state.price = price
        state.last_update = time.time()
        state.tick_count += 1
        state.connected = True
        self.total_ticks += 1
        self._price_history.append((time.time(), price))
        for cb in self._callbacks:
            try:
                cb(price)
            except Exception:
                pass

    async def connect_all(self):
        tasks = [
            self._connect_binance(),
            self._connect_coinbase(),
            self._connect_bybit(),
            self._connect_okx(),
            self._connect_kraken(),
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _connect_binance(self):
        while True:
            try:
                async with websockets.connect(
                    self.WS_URLS[Exchange.BINANCE], ssl=SSL_CTX,
                    ping_interval=20, close_timeout=5,
                ) as ws:
                    self.states[Exchange.BINANCE].connected = True
                    async for msg in ws:
                        data = json.loads(msg)
                        self._update(Exchange.BINANCE, float(data["p"]), float(data["q"]), data.get("m", False))
            except Exception:
                self.states[Exchange.BINANCE].connected = False
                await asyncio.sleep(2)

    async def _connect_coinbase(self):
        while True:
            try:
                async with websockets.connect(
                    self.WS_URLS[Exchange.COINBASE], ssl=SSL_CTX,
                    ping_interval=20, close_timeout=5,
                ) as ws:
                    await ws.send(json.dumps({
                        "type": "subscribe",
                        "channels": [{"name": "ticker", "product_ids": ["BTC-USD"]}]
                    }))
                    self.states[Exchange.COINBASE].connected = True
                    async for msg in ws:
                        data = json.loads(msg)
                        if data.get("type") == "ticker" and "price" in data:
                            self._update(Exchange.COINBASE, float(data["price"]))
            except Exception:
                self.states[Exchange.COINBASE].connected = False
                await asyncio.sleep(2)

    async def _connect_bybit(self):
        while True:
            try:
                async with websockets.connect(
                    self.WS_URLS[Exchange.BYBIT], ssl=SSL_CTX,
                    ping_interval=20, close_timeout=5,
                ) as ws:
                    await ws.send(json.dumps({
                        "op": "subscribe",
                        "args": ["tickers.BTCUSDT"]
                    }))
                    self.states[Exchange.BYBIT].connected = True
                    async for msg in ws:
                        data = json.loads(msg)
                        if "data" in data:
                            d = data["data"]
                            if isinstance(d, list):
                                d = d[0]
                            if "lastPrice" in d:
                                self._update(Exchange.BYBIT, float(d["lastPrice"]))
            except Exception:
                self.states[Exchange.BYBIT].connected = False
                await asyncio.sleep(2)

    async def _connect_okx(self):
        while True:
            try:
                async with websockets.connect(
                    self.WS_URLS[Exchange.OKX], ssl=SSL_CTX,
                    ping_interval=20, close_timeout=5,
                ) as ws:
                    await ws.send(json.dumps({
                        "op": "subscribe",
                        "args": [{"channel": "tickers", "instId": "BTC-USDT"}]
                    }))
                    self.states[Exchange.OKX].connected = True
                    async for msg in ws:
                        data = json.loads(msg)
                        if "data" in data and data["data"]:
                            d = data["data"][0]
                            if "last" in d:
                                self._update(Exchange.OKX, float(d["last"]))
            except Exception:
                self.states[Exchange.OKX].connected = False
                await asyncio.sleep(2)

    async def _connect_kraken(self):
        while True:
            try:
                async with websockets.connect(
                    self.WS_URLS[Exchange.KRAKEN], ssl=SSL_CTX,
                    ping_interval=20, close_timeout=5,
                ) as ws:
                    await ws.send(json.dumps({
                        "method": "subscribe",
                        "params": {"channel": "ticker", "symbol": ["BTC/USD"]}
                    }))
                    self.states[Exchange.KRAKEN].connected = True
                    async for msg in ws:
                        data = json.loads(msg)
                        if "data" in data and data["data"]:
                            for d in data["data"]:
                                if "last" in d:
                                    self._update(Exchange.KRAKEN, float(d["last"]))
            except Exception:
                self.states[Exchange.KRAKEN].connected = False
                await asyncio.sleep(2)


# ─── 2. Empirical Probability Engine ─────────────────────────────────────────

class EmpiricalEngine:
    """
    Prices binary options using REAL historical data instead of theoretical models.

    Built from 31 days of 1-minute BTC candles (9,000 five-minute windows).
    For each (price_diff%, time_remaining) → we know the ACTUAL probability
    that BTC closed above its opening price.

    This is fundamentally more accurate than Black-Scholes because:
    1. It captures BTC's real distribution (fat tails, momentum, etc.)
    2. It doesn't assume log-normal returns
    3. It naturally accounts for microstructure effects
    4. It's calibrated to the EXACT product we're trading (5-min binary)
    """

    def __init__(self, candle_file: str = "btc_1m_candles.pkl"):
        self.prob_surface: Dict[Tuple[float, int], dict] = {}
        self.mom_surface: Dict[Tuple[float, float, int], dict] = {}
        self._pct_bins: List[float] = []
        self._time_bins: List[int] = []
        self._loaded = False

        try:
            self._build(candle_file)
            self._loaded = True
        except FileNotFoundError:
            print("  WARNING: btc_1m_candles.pkl not found. Run download script first.")
        except Exception as e:
            print(f"  WARNING: Could not load empirical data: {e}")

    def _build(self, candle_file: str):
        """Build empirical probability surface from historical candle data."""
        with open(candle_file, "rb") as f:
            candles = pickle.load(f)

        closes = [float(c[4]) for c in candles]
        opens = [float(c[1]) for c in candles]

        window_size = 5
        n_windows = len(closes) // window_size

        bins = defaultdict(lambda: {"up": 0, "total": 0})
        mom_bins = defaultdict(lambda: {"up": 0, "total": 0})

        for w in range(n_windows):
            start = w * window_size
            if start + window_size > len(closes):
                break

            strike = opens[start]
            final_close = closes[start + window_size - 1]
            resolved_up = final_close >= strike

            for minute in range(window_size):
                idx = start + minute
                current = closes[idx]
                pct_diff = (current - strike) / strike * 100

                # Bin to 0.005% resolution for fine-grained lookup
                pct_bin = round(pct_diff / 0.005) * 0.005
                pct_bin = max(-0.5, min(0.5, pct_bin))
                time_remaining = (window_size - minute - 1) * 60

                bins[(pct_bin, time_remaining)]["total"] += 1
                if resolved_up:
                    bins[(pct_bin, time_remaining)]["up"] += 1

                # Momentum-conditional probability
                if idx >= 2:
                    mom = (current - closes[idx - 2]) / closes[idx - 2] * 100
                    mom_bin = round(mom / 0.01) * 0.01
                    mom_bin = max(-0.2, min(0.2, mom_bin))
                    mom_bins[(pct_bin, mom_bin, time_remaining)]["total"] += 1
                    if resolved_up:
                        mom_bins[(pct_bin, mom_bin, time_remaining)]["up"] += 1

        self.prob_surface = dict(bins)
        self.mom_surface = dict(mom_bins)
        self._pct_bins = sorted(set(k[0] for k in bins.keys()))
        self._time_bins = sorted(set(k[1] for k in bins.keys()))

        total_obs = sum(v["total"] for v in bins.values())
        print(f"  Empirical engine: {total_obs:,} observations from {n_windows:,} windows")

    def lookup(self, pct_diff: float, time_remaining: float, momentum: float = 0.0) -> Tuple[float, int]:
        """
        Look up P(UP) from empirical data with time interpolation.

        Returns (probability_up, sample_count).
        """
        if not self._loaded:
            return 0.5, 0

        pct_bin = round(pct_diff / 0.005) * 0.005
        pct_bin = max(-0.5, min(0.5, pct_bin))

        # INTERPOLATE between time bins instead of rounding
        # Our bins are at 0, 60, 120, 180, 240 seconds
        time_lo = int(time_remaining // 60) * 60   # Floor to nearest minute
        time_hi = time_lo + 60                       # Next minute up
        time_lo = max(0, min(240, time_lo))
        time_hi = max(0, min(240, time_hi))

        # Fraction between bins (0.0 = exactly at time_lo, 1.0 = at time_hi)
        if time_hi > time_lo:
            frac = (time_remaining - time_lo) / (time_hi - time_lo)
        else:
            frac = 0.0
        frac = max(0.0, min(1.0, frac))

        # Get prob at both time bins
        prob_lo, count_lo = self._lookup_single(pct_bin, time_lo, momentum)
        prob_hi, count_hi = self._lookup_single(pct_bin, time_hi, momentum)

        # Linear interpolation between bins
        prob = prob_lo * (1 - frac) + prob_hi * frac
        count = min(count_lo, count_hi)

        # Don't return extreme probabilities at T=0 (that's just resolution)
        # Clamp to a reasonable range
        if time_remaining > 5:
            prob = max(0.02, min(0.98, prob))

        return prob, count

    def _lookup_single(self, pct_bin: float, time_bin: int, momentum: float) -> Tuple[float, int]:
        """Look up a single (pct_bin, time_bin) point."""
        # Always start with the BASE probability (most reliable)
        key = (pct_bin, time_bin)
        data = self.prob_surface.get(key)

        if data and data["total"] >= 5:
            base_prob = data["up"] / data["total"]
            count = data["total"]

            # Only use momentum surface if we have LOTS of data (30+ obs)
            # Otherwise it's too noisy
            if abs(momentum) > 0.005:
                mom_bin = round(momentum / 0.01) * 0.01
                mom_bin = max(-0.2, min(0.2, mom_bin))
                mom_key = (pct_bin, mom_bin, time_bin)
                mom_data = self.mom_surface.get(mom_key)

                if mom_data and mom_data["total"] >= 30:
                    # Blend base with momentum data (70% base, 30% momentum)
                    mom_prob = mom_data["up"] / mom_data["total"]
                    base_prob = 0.7 * base_prob + 0.3 * mom_prob
                elif abs(momentum) > 0.01:
                    # Small momentum nudge when no specific data
                    nudge = momentum * 0.15  # Reduced from 0.25
                    nudge = max(-0.03, min(0.03, nudge))
                    base_prob = max(0.02, min(0.98, base_prob + nudge))

            return base_prob, count

        return self._interpolate(pct_bin, time_bin), 0

    def _interpolate(self, pct_bin: float, time_bin: int) -> float:
        """Interpolate from neighboring bins."""
        idx = bisect_left(self._pct_bins, pct_bin)
        pct_lo = self._pct_bins[max(0, idx - 1)]
        pct_hi = self._pct_bins[min(len(self._pct_bins) - 1, idx)]

        tidx = bisect_left(self._time_bins, time_bin)
        time_lo = self._time_bins[max(0, tidx - 1)]
        time_hi = self._time_bins[min(len(self._time_bins) - 1, tidx)]

        total_up = 0
        total_count = 0
        for p in [pct_lo, pct_hi]:
            for t in [time_lo, time_hi]:
                data = self.prob_surface.get((p, t))
                if data and data["total"] > 0:
                    total_up += data["up"]
                    total_count += data["total"]

        return total_up / total_count if total_count > 0 else 0.5


# ─── 3. Edge Detector ─────────────────────────────────────────────────────────

@dataclass
class TradeSignal:
    """A validated trade signal with real edge."""
    side: str                # "UP" or "DOWN"
    empirical_prob: float    # Our probability (from 9,000 real observations)
    market_prob: float       # What Polymarket says
    edge: float              # Our prob - market prob - fees
    kelly_size: float        # Dollar amount to bet
    confidence: float        # Confidence score (0-1)
    pct_diff: float          # BTC vs strike (%)
    time_remaining: float    # Seconds left
    momentum: float          # 2-min momentum (%)
    sample_count: int        # Historical observations backing this


class EdgeDetector:
    """
    Compares empirical fair value vs Polymarket market price.

    Philosophy:
    - The market is usually RIGHT (market makers are smart)
    - We only trade when our empirical data DISAGREES by enough to cover fees
    - Edges are small (2-5%) but backed by thousands of real observations
    - We are CONSERVATIVE: better to miss opportunities than lose money
    """

    def __init__(self, engine: EmpiricalEngine):
        self.engine = engine

    def evaluate(
        self,
        btc_price: float,
        strike: float,
        time_remaining: float,
        market_price_up: float,
        market_price_down: float,
        momentum: float = 0.0,
    ) -> Optional[TradeSignal]:
        """
        Check if there's a genuine, data-backed edge.
        Returns None if no edge (which is MOST of the time - that's correct).
        """
        if strike <= 0 or btc_price <= 0:
            return None

        # Don't trade in last 15 seconds (spreads widen, slippage risk)
        if time_remaining < 15:
            return None

        # Don't trade in first 15 seconds (strike might not be set correctly)
        if time_remaining > 285:
            return None

        # Skip if market prices look invalid
        if market_price_up < 0.03 or market_price_up > 0.97:
            return None  # Too extreme, likely no liquidity on other side
        if market_price_down < 0.03 or market_price_down > 0.97:
            return None

        pct_diff = (btc_price - strike) / strike * 100

        # Get empirical probability
        emp_prob_up, sample_count = self.engine.lookup(pct_diff, time_remaining, momentum)
        emp_prob_down = 1.0 - emp_prob_up

        if sample_count < MIN_SAMPLES:
            return None  # Not enough data, don't guess

        # Calculate edge for both sides
        # Edge = our_probability - market_price - fee_rate
        # We need to beat the market price PLUS fees to be profitable
        edge_up = emp_prob_up - market_price_up - FEE_RATE
        edge_down = emp_prob_down - market_price_down - FEE_RATE

        # Pick the best side (or no trade)
        if edge_up > edge_down and edge_up > MIN_EDGE:
            side = "UP"
            our_prob = emp_prob_up
            market_prob = market_price_up
            edge = edge_up
        elif edge_down > MIN_EDGE:
            side = "DOWN"
            our_prob = emp_prob_down
            market_prob = market_price_down
            edge = edge_down
        else:
            return None  # No edge

        # SAFETY CHECK: Don't bet against strong trends
        # If BTC is clearly above strike and we want to buy DOWN, that's suspicious
        if side == "DOWN" and pct_diff > 0.05 and time_remaining < 120:
            return None  # BTC is significantly above strike with little time
        if side == "UP" and pct_diff < -0.05 and time_remaining < 120:
            return None  # BTC is significantly below strike with little time

        # Confidence based on sample size
        sample_conf = min(1.0, sample_count / 50)  # Max confidence at 50+ samples
        edge_conf = min(1.0, edge / 0.10)           # Max confidence at 10%+ edge
        confidence = 0.5 * sample_conf + 0.5 * edge_conf

        # Kelly sizing
        if market_prob > 0.02 and market_prob < 0.98:
            odds = (1.0 / market_prob) - 1
            kelly_pct = (our_prob * odds - (1 - our_prob)) / odds
            kelly_pct = max(0, kelly_pct) * KELLY_FRACTION
            kelly_size = kelly_pct * BANKROLL
        else:
            kelly_size = 0

        # Clamp bet size
        kelly_size = min(MAX_BET_SIZE, kelly_size)
        if kelly_size < MIN_BET_SIZE:
            return None  # Edge too small to bother

        return TradeSignal(
            side=side,
            empirical_prob=our_prob,
            market_prob=market_prob,
            edge=edge,
            kelly_size=kelly_size,
            confidence=confidence,
            pct_diff=pct_diff,
            time_remaining=time_remaining,
            momentum=momentum,
            sample_count=sample_count,
        )


# ─── 4. Paper Execution Engine ────────────────────────────────────────────────

@dataclass
class PaperTrade:
    """Record of a paper trade."""
    timestamp: float
    window_id: int
    side: str
    entry_price: float      # Market price we bought at
    empirical_prob: float    # Our probability estimate
    size: float              # Dollar amount
    edge: float              # Expected edge
    pct_diff: float          # BTC vs strike at entry
    time_remaining: float    # Seconds left at entry
    momentum: float
    # Filled after resolution
    resolved_up: Optional[bool] = None
    pnl: Optional[float] = None
    won: Optional[bool] = None


class PaperExecutor:
    """
    Simulates trade execution and tracks P&L.
    Resolves trades at the end of each 5-minute window.
    """

    def __init__(self):
        self.open_trades: Dict[int, List[PaperTrade]] = {}  # window_id -> trades
        self.closed_trades: List[PaperTrade] = []
        self.total_pnl: float = 0.0

    def execute(self, signal: TradeSignal, window_id: int) -> PaperTrade:
        """Record a paper trade."""
        trade = PaperTrade(
            timestamp=time.time(),
            window_id=window_id,
            side=signal.side,
            entry_price=signal.market_prob,
            empirical_prob=signal.empirical_prob,
            size=signal.kelly_size,
            edge=signal.edge,
            pct_diff=signal.pct_diff,
            time_remaining=signal.time_remaining,
            momentum=signal.momentum,
        )

        if window_id not in self.open_trades:
            self.open_trades[window_id] = []
        self.open_trades[window_id].append(trade)
        return trade

    def resolve_window(self, window_id: int, resolved_up: bool):
        """Resolve all trades in a window."""
        if window_id not in self.open_trades:
            return

        for trade in self.open_trades[window_id]:
            trade.resolved_up = resolved_up
            trade.won = (trade.side == "UP" and resolved_up) or (trade.side == "DOWN" and not resolved_up)

            if trade.won:
                # Payout: we bet $X at price P, payout is $X * (1/P - 1) * (1 - fee)
                payout = trade.size * (1.0 / trade.entry_price - 1) * (1 - FEE_RATE)
                trade.pnl = payout
            else:
                trade.pnl = -trade.size

            self.total_pnl += trade.pnl
            self.closed_trades.append(trade)

        del self.open_trades[window_id]

    @property
    def win_rate(self) -> float:
        if not self.closed_trades:
            return 0.0
        return sum(1 for t in self.closed_trades if t.won) / len(self.closed_trades)

    @property
    def total_trades(self) -> int:
        return len(self.closed_trades)

    def summary_str(self) -> str:
        if not self.closed_trades:
            return "No closed trades yet."

        wins = sum(1 for t in self.closed_trades if t.won)
        losses = self.total_trades - wins
        avg_win = np.mean([t.pnl for t in self.closed_trades if t.won]) if wins > 0 else 0
        avg_loss = np.mean([t.pnl for t in self.closed_trades if not t.won]) if losses > 0 else 0

        lines = [
            f"  Trades:     {self.total_trades}",
            f"  Wins:       {wins} ({self.win_rate:.1%})",
            f"  Losses:     {losses}",
            f"  Total P&L:  ${self.total_pnl:+,.2f}",
            f"  Avg Win:    ${avg_win:+,.2f}",
            f"  Avg Loss:   ${avg_loss:+,.2f}",
        ]

        # Per-trade details
        if self.closed_trades:
            lines.append("")
            lines.append("  Recent trades:")
            for t in self.closed_trades[-10:]:
                result = "WIN " if t.won else "LOSS"
                lines.append(
                    f"    {result} | {t.side:>4} @ {t.entry_price:.3f} "
                    f"| Emp: {t.empirical_prob:.3f} | Edge: {t.edge:.1%} "
                    f"| ${t.pnl:+,.2f} | Diff: {t.pct_diff:+.3f}%"
                )

        return "\n".join(lines)


# ─── 5. Main Strategy ─────────────────────────────────────────────────────────

class EmpiricalStrategy:
    """
    Main strategy: detect real Polymarket markets, compare empirical
    probabilities vs market prices, trade when we have genuine edge.
    """

    def __init__(self, bankroll: float = BANKROLL):
        self.bankroll = bankroll
        self.running = False

        # Components
        self.feed = MultiExchangeFeed()
        self.engine = EmpiricalEngine()
        self.detector = EdgeDetector(self.engine)
        self.executor = PaperExecutor()

        # Market state
        self.strike: float = 0.0
        self.market_price_up: float = 0.5
        self.market_price_down: float = 0.5
        self.token_id_up: str = ""
        self.token_id_down: str = ""
        self.current_market_title: str = ""

        # Window tracking
        self._current_window_id: int = 0
        self._window_strikes: Dict[int, float] = {}  # window_id -> strike price
        self._window_btc_at_start: Dict[int, float] = {}
        self._window_transitions_seen: int = 0  # Need 2 transitions before trading

        # Stats
        self.signals_generated = 0
        self.last_signal_time: float = 0
        self.start_time: float = 0
        self._trades_this_window: int = 0

    async def run(self):
        self.running = True
        self.start_time = time.time()

        print("=" * 70)
        print("  EMPIRICAL BTC BINARY OPTION BOT")
        print("=" * 70)
        print(f"  Model:       Empirical (31 days, 9,000 windows)")
        print(f"  Mode:        PAPER TRADING")
        print(f"  Bankroll:    ${self.bankroll:,.0f}")
        print(f"  Min edge:    {MIN_EDGE*100:.1f}%")
        print(f"  Kelly frac:  {KELLY_FRACTION*100:.0f}%")
        print(f"  Max bet:     ${MAX_BET_SIZE}")
        print(f"  Fees:        {FEE_RATE*100:.0f}%")
        print(f"  Min samples: {MIN_SAMPLES}")
        print(f"  Feeds:       5 exchanges (Binance, Coinbase, Bybit, OKX, Kraken)")
        print("=" * 70)
        print()

        try:
            await asyncio.gather(
                self.feed.connect_all(),
                self._strategy_loop(),
                self._market_monitor_loop(),
                self._display_loop(),
            )
        except asyncio.CancelledError:
            pass
        except KeyboardInterrupt:
            pass
        finally:
            self._print_summary()

    async def _market_monitor_loop(self):
        """
        Poll Polymarket for active BTC 5-min markets every 2s.
        Fetch real order book prices.
        """
        session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=SSL_CTX),
            headers={"Accept": "application/json"},
            timeout=aiohttp.ClientTimeout(total=10),
        )
        last_slug = ""
        try:
            while self.running:
                try:
                    now_ts = int(time.time())
                    boundary = now_ts - (now_ts % 300)
                    slug = f"btc-updown-5m-{boundary}"

                    if slug != last_slug:
                        async with session.get(
                            f"{GAMMA_API}/events",
                            params={"slug": slug},
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                if data:
                                    event = data[0] if isinstance(data, list) else data
                                    markets = event.get("markets", [])
                                    if markets:
                                        self._update_market(markets[0])
                                        last_slug = slug
                                        remaining = 300 - (now_ts % 300)
                                        self.current_market_title = event.get("title", slug)
                                        print(f"\n  MARKET: {self.current_market_title} | T-{remaining}s")
                                        print(f"    Token UP:   {self.token_id_up[:16]}...")
                                        print(f"    Token DOWN: {self.token_id_down[:16]}...")

                    # Fetch order book for UP token
                    if self.token_id_up:
                        async with session.get(
                            f"{CLOB_API}/book",
                            params={"token_id": self.token_id_up},
                        ) as resp:
                            if resp.status == 200:
                                book = await resp.json()
                                asks = book.get("asks", [])
                                bids = book.get("bids", [])
                                if asks:
                                    best_ask = min(float(a["price"]) for a in asks)
                                    best_bid = max(float(b["price"]) for b in bids) if bids else 0.0
                                    if best_bid > 0:
                                        self.market_price_up = (best_ask + best_bid) / 2
                                    else:
                                        self.market_price_up = best_ask

                    # Fetch order book for DOWN token
                    if self.token_id_down:
                        async with session.get(
                            f"{CLOB_API}/book",
                            params={"token_id": self.token_id_down},
                        ) as resp:
                            if resp.status == 200:
                                book = await resp.json()
                                asks = book.get("asks", [])
                                bids = book.get("bids", [])
                                if asks:
                                    best_ask_d = min(float(a["price"]) for a in asks)
                                    best_bid_d = max(float(b["price"]) for b in bids) if bids else 0.0
                                    if best_bid_d > 0:
                                        self.market_price_down = (best_ask_d + best_bid_d) / 2
                                    else:
                                        self.market_price_down = best_ask_d

                                    # Normalize if UP + DOWN != 1
                                    total = self.market_price_up + self.market_price_down
                                    if abs(total - 1.0) > 0.05:
                                        self.market_price_up /= total
                                        self.market_price_down /= total

                except Exception:
                    pass

                await asyncio.sleep(2)
        finally:
            await session.close()

    def _update_market(self, m: Dict):
        """Update market state from Gamma API data."""
        import re

        clob_ids = m.get("clobTokenIds", "[]")
        if isinstance(clob_ids, str):
            try:
                clob_ids = json.loads(clob_ids)
            except Exception:
                return
        if len(clob_ids) >= 2:
            self.token_id_up = str(clob_ids[0])
            self.token_id_down = str(clob_ids[1])

        prices = m.get("outcomePrices", "[]")
        if isinstance(prices, str):
            try:
                prices = [float(p) for p in json.loads(prices)]
            except Exception:
                prices = []
        if len(prices) >= 2:
            self.market_price_up = prices[0]
            self.market_price_down = prices[1]

    async def _strategy_loop(self):
        """Core strategy loop - runs every 50ms."""
        # Wait for price feeds
        while self.feed.connected_count == 0:
            await asyncio.sleep(0.1)

        print(f"  Feeds connected: {self.feed.connected_count}/5")
        print("  Waiting for first 5-min window...")

        while self.running:
            try:
                now = time.time()
                elapsed = now % 300
                time_remaining = 300 - elapsed
                window_id = int(now // 300)

                btc = self.feed.best_price
                if btc <= 0:
                    await asyncio.sleep(0.1)
                    continue

                # New window: set strike and resolve previous window
                if window_id != self._current_window_id:
                    old_window = self._current_window_id

                    # Resolve previous window's trades
                    if old_window > 0 and old_window in self._window_strikes:
                        old_strike = self._window_strikes[old_window]
                        resolved_up = btc >= old_strike
                        self.executor.resolve_window(old_window, resolved_up)

                        n_trades = len([t for t in self.executor.closed_trades if t.window_id == old_window])
                        if n_trades > 0:
                            result = "UP" if resolved_up else "DOWN"
                            print(f"\n  *** WINDOW RESOLVED: {result} "
                                  f"| BTC ${btc:,.2f} vs Strike ${old_strike:,.2f} "
                                  f"| P&L: ${self.executor.total_pnl:+,.2f} ***")

                    self._window_transitions_seen += 1
                    self._current_window_id = window_id
                    self.strike = btc
                    self._window_strikes[window_id] = btc
                    self._trades_this_window = 0

                    if self._window_transitions_seen == 1:
                        # First transition = we just saw the end of our initial partial window
                        # NOW we have a clean strike
                        print(f"\n  === FIRST CLEAN WINDOW | Strike: ${btc:,.2f} "
                              f"| Mkt UP: {self.market_price_up:.3f} DOWN: {self.market_price_down:.3f} ===")
                    else:
                        print(f"\n  === NEW WINDOW {window_id} | Strike: ${btc:,.2f} "
                              f"| Mkt UP: {self.market_price_up:.3f} DOWN: {self.market_price_down:.3f} ===")

                # First window: skip (partial), just note it
                if self._current_window_id == 0:
                    self._current_window_id = window_id
                    self.strike = 0  # Don't set strike for partial window
                    print(f"\n  === PARTIAL WINDOW (joining mid-window, will skip) ===")

                # Don't trade until we've seen at least 1 full window transition
                if self.strike <= 0 or self._window_transitions_seen < 1:
                    await asyncio.sleep(0.1)
                    continue

                # Get momentum
                momentum = self.feed.get_momentum(lookback_seconds=120)

                # Evaluate signal
                signal = self.detector.evaluate(
                    btc_price=btc,
                    strike=self.strike,
                    time_remaining=time_remaining,
                    market_price_up=self.market_price_up,
                    market_price_down=self.market_price_down,
                    momentum=momentum,
                )

                if signal:
                    self.signals_generated += 1

                    # Cooldown check + max trades per window
                    if ((now - self.last_signal_time) > SIGNAL_COOLDOWN
                            and self._trades_this_window < MAX_TRADES_PER_WINDOW):
                        self.last_signal_time = now

                        # Execute paper trade
                        trade = self.executor.execute(signal, window_id)
                        self._trades_this_window += 1

                        print(f"\n  >>> TRADE #{self._trades_this_window}: BUY {signal.side} "
                              f"| Edge: {signal.edge:.1%} "
                              f"| Emp: {signal.empirical_prob:.3f} vs Mkt: {signal.market_prob:.3f} "
                              f"| Size: ${signal.kelly_size:.1f} "
                              f"| BTC: ${btc:,.2f} vs K: ${self.strike:,.2f} "
                              f"| Diff: {signal.pct_diff:+.3f}% "
                              f"| Mom: {signal.momentum:+.3f}% "
                              f"| T-{signal.time_remaining:.0f}s "
                              f"| Samples: {signal.sample_count}")

                # Run at 20Hz (every 50ms)
                await asyncio.sleep(0.05)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"  [ERROR] Strategy: {e}")
                await asyncio.sleep(0.5)

    async def _display_loop(self):
        """Display status every 3 seconds."""
        while self.running:
            await asyncio.sleep(3.0)

            btc = self.feed.best_price
            if btc <= 0:
                continue

            now = time.time()
            time_rem = 300 - (now % 300)
            pct_diff = (btc - self.strike) / self.strike * 100 if self.strike > 0 else 0

            momentum = self.feed.get_momentum(120)

            # Get empirical probability
            emp_prob, samples = self.engine.lookup(pct_diff, time_rem, momentum)

            ts = time.strftime("%H:%M:%S")
            feeds = self.feed.connected_count

            edge_up = emp_prob - self.market_price_up - FEE_RATE
            edge_dn = (1 - emp_prob) - self.market_price_down - FEE_RATE
            best_edge = max(edge_up, edge_dn)
            edge_side = "UP" if edge_up > edge_dn else "DN"

            print(
                f"  {ts} | BTC ${btc:>10,.2f} | K ${self.strike:>10,.2f} | "
                f"Diff {pct_diff:+.3f}% | T-{time_rem:>3.0f}s | "
                f"Emp {emp_prob:.3f} | Mkt {self.market_price_up:.3f} | "
                f"Edge({edge_side}) {best_edge:+.3f} | "
                f"Mom {momentum:+.3f}% | "
                f"Feeds {feeds}/5 | "
                f"W/L {sum(1 for t in self.executor.closed_trades if t.won)}/{sum(1 for t in self.executor.closed_trades if not t.won)} | "
                f"P&L ${self.executor.total_pnl:+,.2f}"
            )

    def _print_summary(self):
        runtime = time.time() - self.start_time if self.start_time else 0

        print(f"\n{'=' * 70}")
        print("  SESSION SUMMARY")
        print(f"{'=' * 70}")
        print(f"  Runtime:        {runtime:.0f}s ({runtime/60:.1f} min)")
        print(f"  Ticks:          {self.feed.total_ticks:,}")
        print(f"  Signals:        {self.signals_generated}")
        print(f"  Feeds:          {self.feed.connected_count}/5")
        print()

        if self.executor.closed_trades:
            print(self.executor.summary_str())
        else:
            print("  No completed trades.")

        # Open trades
        n_open = sum(len(v) for v in self.executor.open_trades.values())
        if n_open > 0:
            print(f"\n  Open trades: {n_open} (will resolve at window end)")

        print(f"{'=' * 70}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

async def main():
    strategy = EmpiricalStrategy(bankroll=BANKROLL)

    loop = asyncio.get_event_loop()

    def shutdown(signum, frame):
        print("\n  Shutting down...")
        strategy.running = False
        for task in asyncio.all_tasks(loop):
            task.cancel()

    sig_module.signal(sig_module.SIGINT, shutdown)
    sig_module.signal(sig_module.SIGTERM, shutdown)

    await strategy.run()


if __name__ == "__main__":
    if "--benchmark" in sys.argv:
        # Quick benchmark of empirical lookups
        engine = EmpiricalEngine()
        times = []
        for _ in range(100_000):
            t0 = time.perf_counter_ns()
            engine.lookup(0.02, 120, 0.01)
            t1 = time.perf_counter_ns()
            times.append(t1 - t0)
        times = np.array(times)
        print(f"\n  Empirical lookup (100K runs):")
        print(f"    Mean:   {np.mean(times):.0f}ns ({np.mean(times)/1000:.2f}us)")
        print(f"    Median: {np.median(times):.0f}ns")
        print(f"    P99:    {np.percentile(times, 99):.0f}ns")
    else:
        asyncio.run(main())

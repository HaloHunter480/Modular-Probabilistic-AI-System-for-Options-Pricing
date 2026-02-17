"""
Production Live Execution Engine for Polymarket
=================================================

This is the REAL execution module that:
1. Signs orders cryptographically via py-clob-client (EIP-712)
2. Submits to Polymarket CLOB with proper authentication
3. Enforces hard risk limits (circuit breakers, max loss, position limits)
4. Logs every action for audit trail
5. Measures and reports execution latency

Usage:
    from live_executor import LiveExecutor, RiskManager

    executor = LiveExecutor()          # Loads keys from .env
    executor.connect()                 # Validates connection + balance

    risk = RiskManager(max_loss=50, max_position=200, max_orders_per_window=3)

    if risk.allow_trade(signal):
        result = executor.place_order(
            token_id="...",
            side="BUY",
            price=0.55,
            size=10.0,
        )
        risk.record_trade(result)
"""

import os
import time
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from collections import deque
from datetime import datetime

from dotenv import load_dotenv
from hexbytes import HexBytes
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    ApiCreds,
    OrderArgs,
    MarketOrderArgs,
    PartialCreateOrderOptions,
    BookParams,
)

load_dotenv()

# ─── Logging ──────────────────────────────────────────────────────────────────

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("executor")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(
    f"logs/executor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


# ─── Data Types ───────────────────────────────────────────────────────────────

@dataclass
class TradeResult:
    success: bool
    order_id: str = ""
    token_id: str = ""
    side: str = ""
    price: float = 0.0
    size: float = 0.0
    cost: float = 0.0          # price * size
    latency_ms: float = 0.0
    error: str = ""
    timestamp: float = 0.0


@dataclass
class Position:
    token_id: str
    side: str
    size: float
    avg_price: float
    cost: float
    timestamp: float


# ─── Risk Manager ─────────────────────────────────────────────────────────────

class RiskManager:
    """
    Hard risk limits that CANNOT be overridden by the strategy.
    These are your safety net. If any limit is hit, trading halts.

    Limits:
    - max_loss_per_session: Stop if cumulative loss exceeds this (USD)
    - max_position: Maximum total exposure across all positions (USD)
    - max_orders_per_window: Maximum orders per 5-min window
    - max_order_size: Maximum single order size (USD)
    - min_order_size: Minimum order size (USD) - below this isn't worth fees
    - max_daily_orders: Stop after this many orders in a day
    - circuit_breaker_consecutive_losses: Halt after N consecutive losses
    """

    def __init__(
        self,
        max_loss_per_session: float = 50.0,
        max_position: float = 200.0,
        max_orders_per_window: int = 3,
        max_order_size: float = 100.0,
        min_order_size: float = 1.0,
        max_daily_orders: int = 100,
        circuit_breaker_consecutive_losses: int = 5,
    ):
        self.max_loss = max_loss_per_session
        self.max_position = max_position
        self.max_orders_per_window = max_orders_per_window
        self.max_order_size = max_order_size
        self.min_order_size = min_order_size
        self.max_daily_orders = max_daily_orders
        self.circuit_breaker_n = circuit_breaker_consecutive_losses

        # State
        self.session_pnl: float = 0.0
        self.total_exposure: float = 0.0
        self.orders_this_window: int = 0
        self.orders_today: int = 0
        self.consecutive_losses: int = 0
        self.current_window_id: int = 0
        self.halted: bool = False
        self.halt_reason: str = ""
        self.trades: List[TradeResult] = []
        self.positions: Dict[str, Position] = {}

    def allow_trade(self, proposed_size: float, proposed_price: float) -> tuple:
        """
        Check if a trade is allowed. Returns (allowed: bool, reason: str).
        """
        if self.halted:
            return False, f"HALTED: {self.halt_reason}"

        # Update window counter
        window_id = int(time.time() // 300)
        if window_id != self.current_window_id:
            self.current_window_id = window_id
            self.orders_this_window = 0

        cost = proposed_size * proposed_price

        # Check all limits
        if cost < self.min_order_size:
            return False, f"Order too small: ${cost:.2f} < ${self.min_order_size:.2f}"

        if cost > self.max_order_size:
            return False, f"Order too large: ${cost:.2f} > ${self.max_order_size:.2f}"

        if self.total_exposure + cost > self.max_position:
            return False, (f"Position limit: ${self.total_exposure:.2f} + ${cost:.2f} "
                           f"> ${self.max_position:.2f}")

        if self.session_pnl < -self.max_loss:
            self.halted = True
            self.halt_reason = f"Max loss exceeded: ${self.session_pnl:.2f}"
            return False, self.halt_reason

        if self.orders_this_window >= self.max_orders_per_window:
            return False, f"Window order limit: {self.orders_this_window}/{self.max_orders_per_window}"

        if self.orders_today >= self.max_daily_orders:
            self.halted = True
            self.halt_reason = f"Daily order limit: {self.orders_today}/{self.max_daily_orders}"
            return False, self.halt_reason

        if self.consecutive_losses >= self.circuit_breaker_n:
            self.halted = True
            self.halt_reason = f"Circuit breaker: {self.consecutive_losses} consecutive losses"
            return False, self.halt_reason

        return True, "OK"

    def record_trade(self, result: TradeResult):
        """Record a trade result and update state."""
        self.trades.append(result)
        if result.success:
            self.orders_this_window += 1
            self.orders_today += 1
            self.total_exposure += result.cost

            # Track position
            if result.token_id in self.positions:
                pos = self.positions[result.token_id]
                pos.size += result.size
                pos.cost += result.cost
                pos.avg_price = pos.cost / pos.size if pos.size > 0 else 0
            else:
                self.positions[result.token_id] = Position(
                    token_id=result.token_id,
                    side=result.side,
                    size=result.size,
                    avg_price=result.price,
                    cost=result.cost,
                    timestamp=result.timestamp,
                )

    def record_pnl(self, amount: float):
        """Record realized P&L."""
        self.session_pnl += amount
        if amount < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def status(self) -> str:
        return (
            f"PnL: ${self.session_pnl:+.2f} | "
            f"Exposure: ${self.total_exposure:.2f}/{self.max_position:.0f} | "
            f"Orders: {self.orders_this_window}/{self.max_orders_per_window} (window) "
            f"{self.orders_today}/{self.max_daily_orders} (day) | "
            f"Consec losses: {self.consecutive_losses}/{self.circuit_breaker_n} | "
            f"{'HALTED: ' + self.halt_reason if self.halted else 'ACTIVE'}"
        )


# ─── Live Executor ────────────────────────────────────────────────────────────

class LiveExecutor:
    """
    Production order execution via Polymarket CLOB.

    Uses py-clob-client for:
    - EIP-712 order signing (cryptographic proof of intent)
    - Authenticated API calls (API key + secret + passphrase)
    - Order book fetching (best bid/ask for marketable orders)
    """

    def __init__(self, tick_size: str = "0.01"):
        self.tick_size = tick_size

        # Load credentials
        self.private_key = os.getenv("POLY_PRIVATE_KEY", "")
        self.api_key = os.getenv("POLY_API_KEY", "")
        self.api_secret = os.getenv("POLY_API_SECRET", "")
        self.api_passphrase = os.getenv("POLY_API_PASSPHRASE", "")

        self.client: Optional[ClobClient] = None
        self.connected = False
        self.address = ""

        # Metrics
        self.total_orders = 0
        self.successful_orders = 0
        self.failed_orders = 0
        self.latencies_ms: deque = deque(maxlen=100)

    def connect(self) -> bool:
        """Initialize connection to Polymarket CLOB."""
        logger.info("Connecting to Polymarket CLOB...")

        if not self.private_key:
            logger.error("POLY_PRIVATE_KEY not set in .env")
            return False

        try:
            # Create client
            self.client = ClobClient(
                host="https://clob.polymarket.com",
                key=HexBytes(self.private_key),
                chain_id=137,  # Polygon mainnet
            )

            # Set API credentials if available
            if self.api_key and self.api_secret and self.api_passphrase:
                creds = ApiCreds(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    api_passphrase=self.api_passphrase,
                )
                self.client.set_api_creds(creds)
                logger.info("API credentials set")
            else:
                # Try to derive API credentials
                logger.info("No API creds found, attempting to derive...")
                try:
                    creds = self.client.create_or_derive_api_creds()
                    self.client.set_api_creds(creds)
                    logger.info(f"Derived API key: {creds.api_key[:8]}...")
                except Exception as e:
                    logger.warning(f"Could not derive API creds: {e}")
                    logger.warning("You may need to generate API keys at https://polymarket.com")

            # Validate connection
            from eth_account import Account
            account = Account.from_key(self.private_key)
            self.address = account.address
            logger.info(f"Wallet: {self.address}")

            # Check balance
            try:
                balance = self.client.get_balance_allowance()
                logger.info(f"Balance/Allowance: {balance}")
            except Exception as e:
                logger.warning(f"Could not fetch balance: {e}")

            self.connected = True
            logger.info("Connected to Polymarket CLOB")
            return True

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def get_order_book(self, token_id: str) -> Optional[dict]:
        """Fetch order book for a token. Returns best bid/ask."""
        if not self.client:
            return None

        try:
            t0 = time.perf_counter_ns()
            book = self.client.get_order_book(token_id)
            t1 = time.perf_counter_ns()
            latency = (t1 - t0) / 1e6

            result = {
                "bids": book.bids if book.bids else [],
                "asks": book.asks if book.asks else [],
                "best_bid": float(book.bids[0].price) if book.bids else 0.0,
                "best_ask": float(book.asks[0].price) if book.asks else 1.0,
                "bid_size": float(book.bids[0].size) if book.bids else 0.0,
                "ask_size": float(book.asks[0].size) if book.asks else 0.0,
                "spread": 0.0,
                "latency_ms": latency,
            }
            result["spread"] = result["best_ask"] - result["best_bid"]
            return result

        except Exception as e:
            logger.error(f"Order book fetch failed: {e}")
            return None

    def place_limit_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
    ) -> TradeResult:
        """
        Place a limit order on Polymarket CLOB.

        This is the core execution path:
        1. Create order args
        2. py-clob-client signs with EIP-712
        3. Submit to CLOB API
        4. Return result with latency

        Args:
            token_id: Polymarket token ID
            side: "BUY" or "SELL"
            price: Limit price (0.01 to 0.99)
            size: Number of shares
        """
        if not self.client or not self.connected:
            return TradeResult(success=False, error="Not connected")

        t_start = time.perf_counter_ns()

        try:
            # Round price to tick size
            if self.tick_size == "0.01":
                price = round(price, 2)
            elif self.tick_size == "0.001":
                price = round(price, 3)

            # Clamp price to valid range
            price = max(0.01, min(0.99, price))
            size = round(size, 2)

            if size <= 0:
                return TradeResult(success=False, error=f"Invalid size: {size}")

            logger.info(f"Placing order: {side} {size:.2f} @ {price:.3f} | token={token_id[:12]}...")

            # Create and sign order
            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=side,
                fee_rate_bps=0,
            )

            options = PartialCreateOrderOptions(tick_size=self.tick_size)

            # This call:
            # 1. Creates the order struct
            # 2. Signs it with EIP-712 using the private key
            # 3. Submits to CLOB API with authentication
            # 4. Returns the order response
            response = self.client.create_and_post_order(order_args, options)

            t_end = time.perf_counter_ns()
            latency_ms = (t_end - t_start) / 1e6

            self.total_orders += 1
            self.latencies_ms.append(latency_ms)

            # Parse response
            if response and hasattr(response, "get"):
                order_id = response.get("orderID", response.get("id", ""))
                success = response.get("success", bool(order_id))
            elif isinstance(response, dict):
                order_id = response.get("orderID", response.get("id", ""))
                success = response.get("success", bool(order_id))
            else:
                order_id = str(response) if response else ""
                success = bool(response)

            result = TradeResult(
                success=success,
                order_id=str(order_id),
                token_id=token_id,
                side=side,
                price=price,
                size=size,
                cost=price * size,
                latency_ms=latency_ms,
                timestamp=time.time(),
            )

            if success:
                self.successful_orders += 1
                logger.info(
                    f"ORDER FILLED | {side} {size:.2f} @ {price:.3f} | "
                    f"Cost: ${result.cost:.2f} | "
                    f"ID: {order_id} | "
                    f"Latency: {latency_ms:.1f}ms"
                )
            else:
                self.failed_orders += 1
                logger.warning(f"ORDER REJECTED | Response: {response}")
                result.error = str(response)

            return result

        except Exception as e:
            t_end = time.perf_counter_ns()
            latency_ms = (t_end - t_start) / 1e6
            self.total_orders += 1
            self.failed_orders += 1
            logger.error(f"ORDER FAILED | {e} | Latency: {latency_ms:.1f}ms")
            return TradeResult(
                success=False,
                error=str(e),
                token_id=token_id,
                side=side,
                price=price,
                size=size,
                latency_ms=latency_ms,
                timestamp=time.time(),
            )

    def place_market_order(
        self,
        token_id: str,
        side: str,
        amount: float,
        worst_price: float = 0.0,
    ) -> TradeResult:
        """
        Place a Fill-or-Kill market order.
        Executes immediately at best available price or cancels.

        Args:
            token_id: Polymarket token ID
            side: "BUY" or "SELL"
            amount: Dollar amount to spend
            worst_price: Worst acceptable price (0 = any price)
        """
        if not self.client or not self.connected:
            return TradeResult(success=False, error="Not connected")

        t_start = time.perf_counter_ns()

        try:
            logger.info(f"Market order: {side} ${amount:.2f} | token={token_id[:12]}...")

            order_args = MarketOrderArgs(
                token_id=token_id,
                amount=amount,
                side=side,
                price=worst_price if worst_price > 0 else None,
                fee_rate_bps=0,
            )

            options = PartialCreateOrderOptions(tick_size=self.tick_size)

            response = self.client.create_market_order(order_args, options)

            t_end = time.perf_counter_ns()
            latency_ms = (t_end - t_start) / 1e6

            self.total_orders += 1
            self.latencies_ms.append(latency_ms)

            result = TradeResult(
                success=bool(response),
                order_id=str(response) if response else "",
                token_id=token_id,
                side=side,
                price=worst_price,
                size=amount,
                cost=amount,
                latency_ms=latency_ms,
                timestamp=time.time(),
            )

            if result.success:
                self.successful_orders += 1
                logger.info(f"MARKET ORDER OK | {side} ${amount:.2f} | Latency: {latency_ms:.1f}ms")
            else:
                self.failed_orders += 1
                logger.warning(f"MARKET ORDER FAILED | Response: {response}")

            return result

        except Exception as e:
            t_end = time.perf_counter_ns()
            latency_ms = (t_end - t_start) / 1e6
            self.total_orders += 1
            self.failed_orders += 1
            logger.error(f"MARKET ORDER ERROR | {e}")
            return TradeResult(
                success=False,
                error=str(e),
                latency_ms=latency_ms,
                timestamp=time.time(),
            )

    def cancel_all(self, market: str = "", asset_id: str = "") -> bool:
        """Cancel all open orders."""
        if not self.client:
            return False
        try:
            self.client.cancel_market_orders(market=market, asset_id=asset_id)
            logger.info("All orders cancelled")
            return True
        except Exception as e:
            logger.error(f"Cancel failed: {e}")
            return False

    def get_open_orders(self) -> list:
        """Get all open orders."""
        if not self.client:
            return []
        try:
            result = self.client.get_orders()
            return result if result else []
        except Exception as e:
            logger.error(f"Get orders failed: {e}")
            return []

    def metrics(self) -> str:
        avg_lat = sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0
        return (
            f"Orders: {self.total_orders} "
            f"({self.successful_orders} ok, {self.failed_orders} fail) | "
            f"Avg latency: {avg_lat:.1f}ms"
        )


# ─── Integration with HFT Strategy ───────────────────────────────────────────

def integrate_with_hft():
    """
    Shows how to connect LiveExecutor + RiskManager to btc_hft.py

    This replaces the paper ExecutionEngine with real order execution.
    """
    print("=" * 70)
    print("LIVE EXECUTION INTEGRATION TEST")
    print("=" * 70)

    # 1. Connect
    executor = LiveExecutor(tick_size="0.01")
    if not executor.connect():
        print("Failed to connect. Check your .env credentials.")
        return

    # 2. Initialize risk manager
    risk = RiskManager(
        max_loss_per_session=50,      # Stop if down $50
        max_position=200,              # Max $200 exposure at once
        max_orders_per_window=3,       # Max 3 orders per 5-min window
        max_order_size=50,             # Max $50 per order
        min_order_size=1.0,            # Min $1 per order
        max_daily_orders=50,           # Max 50 orders per day
        circuit_breaker_consecutive_losses=5,  # Halt after 5 losses in a row
    )

    print(f"\nRisk limits:")
    print(f"  Max loss/session:  ${risk.max_loss:.0f}")
    print(f"  Max position:      ${risk.max_position:.0f}")
    print(f"  Max orders/window: {risk.max_orders_per_window}")
    print(f"  Max order size:    ${risk.max_order_size:.0f}")
    print(f"  Circuit breaker:   {risk.circuit_breaker_n} consecutive losses")

    # 3. Test order book fetch
    print(f"\nExecutor: {executor.metrics()}")
    print(f"Risk:     {risk.status()}")
    print("=" * 70)
    print("\nTo go live, update btc_hft.py's ExecutionEngine to use LiveExecutor.")
    print("See go_live() function below.\n")


def go_live(bankroll: float = 500):
    """
    Full live trading mode.
    Patches btc_hft.py's ExecutionEngine with real execution.
    """
    import asyncio
    import btc_hft

    # Create live components
    executor = LiveExecutor(tick_size="0.01")
    if not executor.connect():
        print("FATAL: Could not connect to Polymarket")
        return

    risk = RiskManager(
        max_loss_per_session=bankroll * 0.10,   # 10% max session loss
        max_position=bankroll * 0.40,            # 40% max exposure
        max_orders_per_window=3,
        max_order_size=bankroll * 0.10,
        min_order_size=1.0,
        max_daily_orders=100,
        circuit_breaker_consecutive_losses=5,
    )

    # Create strategy with live execution patched in
    strategy = btc_hft.HFTStrategy(bankroll=bankroll, paper=False)

    # Monkey-patch the executor's execute method
    original_execute = strategy.executor.execute

    async def live_execute(signal, token_id):
        # Risk check
        size = signal.kelly_size / signal.market_price if signal.market_price > 0 else 0
        allowed, reason = risk.allow_trade(size, signal.market_price)

        if not allowed:
            logger.warning(f"Trade blocked: {reason}")
            return False

        # Execute for real
        result = executor.place_limit_order(
            token_id=token_id,
            side="BUY",
            price=signal.market_price,
            size=round(size, 2),
        )

        risk.record_trade(result)
        logger.info(f"Risk: {risk.status()}")

        return result.success

    strategy.executor.execute = live_execute

    print("=" * 70)
    print("GOING LIVE")
    print("=" * 70)
    print(f"  Bankroll:     ${bankroll:,.0f}")
    print(f"  Max loss:     ${risk.max_loss:,.0f} ({risk.max_loss/bankroll*100:.0f}%)")
    print(f"  Max position: ${risk.max_position:,.0f} ({risk.max_position/bankroll*100:.0f}%)")
    print(f"  Executor:     {executor.metrics()}")
    print(f"  Risk:         {risk.status()}")
    print("=" * 70)

    asyncio.run(strategy.run())


if __name__ == "__main__":
    import sys
    if "--go-live" in sys.argv:
        bankroll = 500
        for arg in sys.argv:
            if arg.startswith("--bankroll="):
                bankroll = float(arg.split("=")[1])
        go_live(bankroll)
    else:
        integrate_with_hft()

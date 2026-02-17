# 🚀 Polymarket HFT Arbitrage System

A professional-grade statistical arbitrage system for trading Bitcoin binary options on Polymarket prediction markets. Built with advanced mathematical models from quantitative finance, market microstructure theory, and high-frequency trading strategies.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper Trading](https://img.shields.io/badge/status-paper%20trading-green)](https://github.com)

## 📊 Project Overview

This system trades 5-minute Bitcoin binary options on Polymarket using:
- **Empirical probability surfaces** calibrated from 31 days of historical BTC data
- **Kelly Criterion** for optimal position sizing
- **Advanced toxic flow detection** (Hawkes process, VPIN, price jumps)
- **Asymmetric market making** with dynamic quote skewing
- **Multi-exchange data fusion** (Polymarket + Binance)

### Performance (Paper Trading - 75 minutes)
- **Return**: +148% ($500 → $1,238)
- **Win Rate**: 71% (20 wins / 8 losses)
- **Trades**: 28 (11 maker, 17 taker)
- **Sharpe Ratio**: ~2.5 (annualized)

> ⚠️ **Disclaimer**: Paper trading results are optimistic. Expected live performance: 0.5-2% daily return with 48-52% win rate due to adverse selection and realistic market conditions.

---

## 🎓 Academic Documentation

This project includes comprehensive academic documentation suitable for university submission:

📄 **[MATHEMATICAL_MODELS.md](MATHEMATICAL_MODELS.md)** - Complete mathematical and statistical framework (1,045 lines):
- Empirical probability model derivation
- Kelly Criterion for optimal betting
- Hawkes self-exciting processes for volume clustering
- VPIN (Volume-Synchronized Probability of Informed Trading)
- Kyle's Lambda for market impact estimation
- Order flow analysis and asymmetric quoting
- Market microstructure theory
- Toxic flow detection algorithms
- Full mathematical proofs with derivations
- Academic references to seminal papers
- Implementation details and complexity analysis

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Professional Strategy                       │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────────┐     │
│  │  Empirical │  │   Toxic     │  │  Order Flow      │     │
│  │  Engine    │  │   Flow      │  │  Pressure        │     │
│  │            │  │   Detector  │  │  Analysis        │     │
│  └────────────┘  └─────────────┘  └──────────────────┘     │
│         ↓               ↓                   ↓                │
│  ┌──────────────────────────────────────────────────┐       │
│  │         Signal Generator                          │       │
│  │  • Reality checks (15s delay, $50 min volume)   │       │
│  │  • Edge capping (15% max for Kelly sizing)      │       │
│  │  • Confidence weighting                          │       │
│  └──────────────────────────────────────────────────┘       │
│         ↓                                                    │
│  ┌──────────────────────────────────────────────────┐       │
│  │         Professional Executor                     │       │
│  │  • Maker/Taker decision (0.8% vs 4.5% edge)     │       │
│  │  • Fill simulation (40% maker, 1.5% taker slip) │       │
│  │  • Pre-emptive order cancellation                │       │
│  └──────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
         ↓                        ↓
┌──────────────┐        ┌──────────────────┐
│  Polymarket  │        │     Binance      │
│     CLOB     │        │   WebSocket      │
│  Order Book  │        │   Price Feed     │
└──────────────┘        └──────────────────┘
```

---

## 🔬 Mathematical Models

### 1. Empirical Probability Surface

Instead of Black-Scholes assumptions, we build probability surfaces from historical data:

```
P(UP | Δ, T) = historical_frequency(UP | price_diff=Δ, time_remaining=T)

where:
Δ = (BTC_current - Strike) / Strike (percentage difference)
T = Time remaining in 5-minute window (seconds)
```

**Training Data**: 45,000 1-minute observations → 9,000 simulated 5-minute windows

### 2. Kelly Criterion Position Sizing

Optimal bet sizing for maximizing logarithmic utility:

```python
f* = (p·b - q) / b  # Full Kelly
f_actual = 0.08 · f* · confidence  # Fractional Kelly (8%)

where:
p = Estimated win probability
b = Odds (payout/stake - 1)
q = 1 - p
confidence = weighted combination of sample size, edge magnitude, spread, toxic risk
```

### 3. Toxic Flow Detection

Multi-faceted detection system:

**Hawkes Process** (Volume Clustering):
```
λ(t) = μ + α·∑ exp(-β(t - t_i))
```

**VPIN** (Order Flow Toxicity):
```
VPIN = |Buy_Volume - Sell_Volume| / Total_Volume
Threshold: 0.80 (80% one-sided flow over 10s window)
```

**Action**: Cancel all resting orders immediately if toxic flow detected.

### 4. Asymmetric Market Making

Dynamic quote skewing based on order flow:

```python
If buying UP and flow is bullish (+0.8):
    Place bid LOWER (defensive - avoid momentum)
If buying UP and flow is bearish (-0.8):
    Place bid HIGHER (aggressive - flow in our favor)

skew = -sign(side) × pressure × 0.04  # Max 4% adjustment
```

---

## 📁 Project Structure

```
arbpoly/
├── README.md                          # This file
├── MATHEMATICAL_MODELS.md             # Academic documentation (1,045 lines)
├── requirements.txt                   # Python dependencies
│
├── professional_strategy.py           # Main trading system (1,563 lines)
│   ├── EmpiricalEngine               # Probability lookup
│   ├── HawkesProcess                 # Volume clustering
│   ├── VPIN                          # Toxic flow detection
│   ├── KyleLambda                    # Market impact
│   ├── OrderFlowPressure             # Directional flow
│   ├── ToxicFlowDetector             # Unified detection
│   ├── AdvancedSignalGenerator       # Signal evaluation
│   └── ProfessionalExecutor          # Order execution
│
├── btc_empirical_bot.py               # Empirical model bot (1,074 lines)
│   ├── EmpiricalEngine               # Core probability engine
│   ├── MultiExchangeFeed             # Binance + Polymarket feeds
│   └── EmpiricalStrategy             # Trading strategy
│
├── empirical_model.py                 # Model calibration script
│   └── Builds probability surfaces from btc_1m_candles.pkl
│
├── btc_1m_candles.pkl                 # Training data (31 days, 45k obs)
└── btc_hft.py                         # Original Black-Scholes bot (deprecated)
```

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.11+
pip install -r requirements.txt
```

### Installation

```bash
git clone https://github.com/yourusername/polymarket-arbitrage.git
cd polymarket-arbitrage
pip install -r requirements.txt
```

### Configuration

Edit the configuration constants in `professional_strategy.py`:

```python
# Position sizing
BANKROLL = 500.0              # Starting bankroll
KELLY_FRACTION = 0.08         # 8% of full Kelly
MIN_BET_SIZE = 4.0           
MAX_BET_SIZE = 30.0

# Execution thresholds
MIN_MAKER_EDGE = 0.008        # 0.8% for limit orders
MIN_TAKER_EDGE = 0.045        # 4.5% for market orders

# Risk management
MAX_TRADES_PER_WINDOW = 2
SIGNAL_COOLDOWN = 25.0        # Seconds between signals

# Reality checks
WINDOW_INIT_DELAY = 15.0      # Wait 15s after window opens
MIN_TOB_VOLUME = 50.0         # Minimum $50 at best bid/ask
MAX_REALISTIC_EDGE = 0.15     # Cap edge at 15%
```

### Run Paper Trading

```bash
# Professional strategy (recommended)
python3 professional_strategy.py

# Empirical strategy (simpler version)
python3 btc_empirical_bot.py
```

### Live Trading Setup

⚠️ **NOT IMPLEMENTED** - This is currently paper trading only.

For live trading, you would need:
1. Polymarket API key
2. Wallet setup with private key
3. EIP-712 order signing implementation
4. Real order submission via CLOB API

See `REALITY_CHECK.md` for details on live trading challenges.

---

## 📊 Features

### ✅ Core Features
- [x] Real-time multi-exchange data aggregation
- [x] Empirical probability calibration from 31 days of BTC data
- [x] Kelly Criterion position sizing with confidence adjustments
- [x] Hawkes process volume clustering detection
- [x] VPIN toxic flow measurement
- [x] Kyle's Lambda market impact estimation
- [x] Asymmetric quote skewing based on order flow
- [x] Pre-emptive order cancellation on toxic flow
- [x] Realistic fill simulation (40% maker, 1.5% taker slippage)
- [x] Order book reality checks (15s delay, $50 min volume, 15% edge cap)

### 🚧 Limitations
- [ ] Live order execution (paper trading only)
- [ ] Portfolio optimization across multiple markets
- [ ] Regime detection (trending vs. ranging)
- [ ] Deep learning for pattern recognition
- [ ] Cross-market arbitrage

---

## 📈 Performance Analysis

### Paper Trading Results (75 minutes, 15 windows)

| Metric | Value |
|--------|-------|
| **Starting Bankroll** | $500 |
| **Ending Bankroll** | $1,238 |
| **Total Return** | +148% |
| **Total Trades** | 28 |
| **Win Rate** | 71.4% (20W / 8L) |
| **Maker/Taker Split** | 11 maker / 17 taker |
| **Maker Fill Rate** | ~40% (as simulated) |
| **Avg Trade Size** | $4-30 (Kelly-weighted) |

### Reality Check Effectiveness

1. **Window Initialization Delay**: ✅ No trades in first 15s of new windows
2. **Ghost Town Filter**: ✅ Blocked thin order books
3. **Edge Capping**: ✅ Limited position sizing on anomalous edges (prevented exponential growth)
4. **Toxic Flow Detection**: ✅ 2,700+ toxic events detected, orders cancelled appropriately
5. **Fill Simulation**: ✅ Realistic maker fill rates applied

### Expected Live Performance

Based on market microstructure theory and adverse selection:

| Metric | Paper | Expected Live |
|--------|-------|---------------|
| **Daily Return** | 200%+ | 0.5-2% |
| **Win Rate** | 71% | 48-52% |
| **Maker Fill Rate** | 40% | 20-40% |
| **Max Drawdown** | 8% | 15-30% |
| **Sharpe Ratio** | 2.5 | 0.8-1.5 |

**Why the gap?**
- Adverse selection: Informed traders pick off maker orders
- Fill degradation: Best prices get hit by HFT before us
- Slippage: Real markets have wider spreads than paper
- Latency: 50-200ms latency vs. instantaneous paper execution

---

## 🔬 Research & References

### Academic Papers

1. **Kelly Criterion**
   - Kelly, J.L. (1956). "A New Interpretation of Information Rate"
   
2. **Market Microstructure**
   - Kyle, A.S. (1985). "Continuous Auctions and Insider Trading"
   - O'Hara, M. (1995). "Market Microstructure Theory"
   
3. **Point Processes**
   - Hawkes, A.G. (1971). "Spectra of some self-exciting point processes"
   
4. **High-Frequency Trading**
   - Easley, D., López de Prado, M.M., O'Hara, M. (2011). "The Microstructure of the Flash Crash: Flow Toxicity, Liquidity Crashes and the Probability of Informed Trading"
   - Aldridge, I. (2013). "High-Frequency Trading: A Practical Guide to Algorithmic Strategies and Trading Systems"

### Books

- "Advances in Financial Machine Learning" by Marcos López de Prado
- "Algorithmic Trading: Winning Strategies and Their Rationale" by Ernie Chan
- "Market Microstructure in Practice" by Lehalle & Laruelle

---

## ⚠️ Risk Disclaimer

**THIS SOFTWARE IS FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY.**

- **Not Financial Advice**: This project is a research implementation and should not be used for live trading without extensive testing and risk management.
- **No Guarantees**: Past performance (paper trading) does not guarantee future results.
- **High Risk**: Trading binary options and cryptocurrency involves substantial risk of loss.
- **Adverse Selection**: Real markets exhibit significant adverse selection that paper trading cannot capture.
- **Regulatory**: Ensure compliance with local regulations before trading.

**By using this software, you acknowledge that:**
- You understand the risks of algorithmic trading
- You will not hold the authors liable for any financial losses
- You will conduct thorough due diligence before any live trading
- You understand paper trading results are not indicative of live performance

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Live Trading Integration**
   - EIP-712 order signing
   - Real CLOB order submission
   - Position reconciliation

2. **Advanced Models**
   - Regime detection (Hidden Markov Models)
   - Deep learning for pattern recognition
   - Multi-asset portfolio optimization

3. **Infrastructure**
   - Lower latency execution (Rust/C++)
   - AWS deployment scripts
   - Monitoring & alerting

4. **Testing**
   - Unit tests for mathematical models
   - Backtesting framework
   - Monte Carlo simulation

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/polymarket-arbitrage.git
cd polymarket-arbitrage

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests (if implemented)
pytest tests/
```

---

## 📄 License

MIT License

Copyright (c) 2026 Harjot

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## 📧 Contact

- **GitHub**: [@yourusername](https://github.com/yourusername)
- **Email**: your.email@example.com

For academic inquiries or collaboration opportunities, please reach out via email.

---

## 🙏 Acknowledgments

- **Polymarket** for providing prediction market infrastructure
- **Binance** for real-time BTC price data
- **Academic Community** for foundational research in market microstructure
- **Open Source Community** for Python libraries (asyncio, numpy, scipy)

---

## 📚 Additional Resources

- [Polymarket API Documentation](https://docs.polymarket.com/)
- [Binance API Documentation](https://binance-docs.github.io/apidocs/)
- [Kelly Criterion Calculator](https://www.albionresearch.com/kelly/)
- [Market Microstructure Blog](https://mechanicalmarkets.wordpress.com/)

---

**Built with ❤️ for quantitative finance research**

*Last Updated: February 2026*

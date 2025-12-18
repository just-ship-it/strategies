# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a TradingView Pine Script strategies repository containing automated trading algorithms for Pine Editor. The repository focuses on LDPS (Liquidity Dependent Price Stability) and LDPM (Liquidity Dependent Price Movement) based trading strategies.

## Code Architecture

### Core Components

- **common.pine**: Shared library containing utility functions used across strategies
  - Position checking functions (`IsLong()`, `IsShort()`)
  - LDPM crossing detection (`LDPMCrossedInLastX()`)
  - Price level analysis (`PriceTouchedInLastX()`)
  - Session time helpers (`InSession()`)
  - Debugging tools (`debugLabel()`)

### Trading Strategies

1. **LDPS Trader.pine**: Bidirectional trading strategy
   - Uses LDPM levels (1-5) for support/resistance analysis
   - Implements LDPS bullish/bearish signals
   - Configurable minimum LDPM level requirements
   - Position type controls (long/short toggles)

2. **LS Scalper.pine**: Short-term scalping strategy
   - LDPS signal-based entry/exit
   - Fixed point-based take profit and stop loss
   - Optional trailing stop functionality
   - Webhook integration support

3. **AI Algo Trailing Stop.txt**: Advanced algorithmic strategy
   - AI-driven plot level analysis
   - LDPM trend analysis with configurable periods
   - Selective trailing stop with trigger points
   - Maximum run-up retracement protection

## Development Workflow

### File Structure
- Pine Script files use `.pine` extension
- Common utilities are centralized in `common.pine` library
- Strategy files import common library: `import Howitzah/common/8 as c`

### Strategy Development Pattern
1. Import common library for shared utilities
2. Define input parameters grouped by functionality
3. Process external signals (LDPS, LDPM)
4. Implement entry/exit logic
5. Add risk management (stop loss, take profit, trailing stops)

### Coding Standards
- **No Line Continuations**: Always write Pine Script code on single lines without line continuations for better readability and consistency

### External Dependencies
- Strategies expect external indicator inputs for LDPS and LDPM signals
- Common library uses TradingView's built-in `ta` library: `import TradingView/ta/10`

## Usage Notes
- All Pine Scripts are designed for TradingView Pine Editor
- Strategies use `//@version=6` Pine Script version
- Code is licensed under Mozilla Public License 2.0
- Files marked as untracked in git should be reviewed before committing
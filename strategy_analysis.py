"""
QuantEdge - Strategy / Finance Statistics Module
==================================================
Computes quantitative finance metrics on QuantEdge signals.

Metrics computed (per stock):
  1. Hit Rate         — % of signals where price moved in predicted direction
  2. Sharpe Ratio     — annualised return / volatility (risk-adjusted performance)
  3. Maximum Drawdown — worst peak-to-trough loss on the test period
  4. Cumulative Return — strategy vs buy-and-hold over 113 test days
  5. Win/Loss Ratio   — average gain on correct signals vs average loss on wrong ones
  6. Signal Frequency — how often the model fires BUY/SELL vs HOLD

Outputs:
  - Terminal print (formatted table)
  - strategy_results.csv
  - strategy_results.json
  - strategy_plot_1_cumulative_returns.png   (one chart, all validated stocks)
  - strategy_plot_2_sharpe_drawdown.png      (bar chart comparison)
  - strategy_plot_3_hit_rate.png             (hit rate vs buy-and-hold baseline)

Usage:
  python strategy_analysis.py

Place in AI_TRADING_CAPSTONE folder alongside pkl and json files.
"""

import pickle
import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

ALL_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "NVDA", "META",
    "RELIANCE.NS", "TCS.NS", "ICICIBANK.NS", "INFY.NS"
]

AUC_GATE       = 0.60
CONF_THRESHOLD = 0.60    # above this = BUY signal, below 0.40 = SELL signal
HOLD_LOW       = 0.40
TRADING_DAYS   = 252     # annualisation factor
RISK_FREE_RATE = 0.065   # 6.5% — Indian 10yr bond rate (conservative)
INITIAL_CAPITAL = 100.0  # ₹100 / $100 starting capital for simulation

# ─────────────────────────────────────────────────────────────
# LOAD PICKLES
# ─────────────────────────────────────────────────────────────

def load_pkl(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"  ✗ Not found: {path}")
        return None
    except Exception as e:
        print(f"  ✗ Error loading {path}: {e}")
        return None

print("\n" + "="*65)
print("  QuantEdge — Strategy & Finance Statistics")
print("="*65)
print("\n[1/7] Loading pickle files...")

xgb_probas    = load_pkl("xgb_probas.pkl")
rf_probas     = load_pkl("rf_probas.pkl")
lstm_probas   = load_pkl("lstm_probas.pkl")
model_results = load_pkl("model_results.pkl")
xgb_results   = load_pkl("xgb_results.pkl")
rf_results    = load_pkl("rf_results.pkl")
lstm_results  = load_pkl("lstm_results.pkl")
stacking_res  = load_pkl("stacking_ensemble_results.pkl")
training_data = load_pkl("training_data.pkl")

# ─────────────────────────────────────────────────────────────
# LOAD PRICE DATA FROM MOST RECENT JSON
# ─────────────────────────────────────────────────────────────

print("\n[2/7] Loading price data from JSON...")

def load_latest_json():
    """Finds and loads the most recent global_market_data JSON."""
    # Try dated files first (most recent)
    patterns = [
        "global_market_data_*.json",
        "complete_market_data.json",
        "collected_data.json",
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))

    if not candidates:
        print("  ✗ No market data JSON found.")
        return None

    # Sort by modification time — take newest
    candidates.sort(key=os.path.getmtime, reverse=True)
    chosen = candidates[0]
    print(f"  ✓ Using: {chosen}")

    try:
        with open(chosen, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"  ✗ Error reading {chosen}: {e}")
        return None

market_data = load_latest_json()

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def get_probas(stock):
    best_model = "xgb"
    for source in [model_results, stacking_res]:
        if source and isinstance(source, dict) and stock in source:
            entry = source[stock]
            if isinstance(entry, dict):
                bm = entry.get("best_model", "xgb")
                if isinstance(bm, str):
                    best_model = bm.lower()
                    break

    prob_map = {
        "xgb": xgb_probas, "xgboost": xgb_probas,
        "rf":  rf_probas,  "random_forest": rf_probas, "randomforest": rf_probas,
        "lstm": lstm_probas,
    }
    for key in [best_model, "xgb", "rf", "lstm"]:
        src = prob_map.get(key)
        if src and isinstance(src, dict) and stock in src:
            arr = np.array(src[stock]).flatten()
            if len(arr) > 0:
                return arr, key
    return None, best_model


def get_y_test(stock):
    for source in [xgb_results, rf_results, lstm_results, training_data]:
        if source is None:
            continue
        if isinstance(source, dict):
            if stock in source:
                entry = source[stock]
                if isinstance(entry, dict):
                    for k in ["y_test", "Y_test", "y_te", "test_labels"]:
                        if k in entry:
                            return np.array(entry[k]).flatten()
            if "y_test" in source:
                return np.array(source["y_test"]).flatten()
    return None


def get_prices(stock):
    """
    Extracts close prices from market_data JSON.
    Returns a pandas Series indexed by date.
    """
    if market_data is None:
        return None

    # Common JSON structures
    entry = None
    if stock in market_data:
        entry = market_data[stock]
    elif "stocks" in market_data and stock in market_data["stocks"]:
        entry = market_data["stocks"][stock]
    elif "data" in market_data and stock in market_data["data"]:
        entry = market_data["data"][stock]

    if entry is None:
        return None

    # Entry could be a list of {date, close} or a dict of {date: close}
    try:
        if isinstance(entry, list):
            df = pd.DataFrame(entry)
            # Find date and close columns (case-insensitive)
            date_col  = next((c for c in df.columns if c.lower() in ["date", "timestamp", "time"]), None)
            close_col = next((c for c in df.columns if c.lower() in ["close", "close_price", "price"]), None)
            if date_col and close_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.sort_values(date_col)
                return pd.Series(df[close_col].values, index=df[date_col].values)

        elif isinstance(entry, dict):
            # Could be {date_str: price} or {close: [...], date: [...]}
            if "close" in entry and isinstance(entry["close"], list):
                closes = entry["close"]
                dates  = entry.get("date", entry.get("dates", entry.get("index", list(range(len(closes))))))
                dates  = pd.to_datetime(dates)
                return pd.Series(closes, index=dates).sort_index()
            else:
                # Assume {date_str: price_value}
                series = pd.Series(entry)
                series.index = pd.to_datetime(series.index)
                return series.sort_index().astype(float)
    except Exception as e:
        pass

    return None


def get_stored_auc(stock):
    for source in [model_results, stacking_res]:
        if source and isinstance(source, dict) and stock in source:
            entry = source[stock]
            if isinstance(entry, dict):
                for k in ["best_auc", "auc", "test_auc"]:
                    if k in entry:
                        return float(entry[k])
    return None


# ─────────────────────────────────────────────────────────────
# FINANCE METRIC FUNCTIONS
# ─────────────────────────────────────────────────────────────

def compute_hit_rate(y_true, y_prob):
    """% of all predictions where direction was correct."""
    predicted = (y_prob >= 0.5).astype(int)
    return float((predicted == y_true).mean())


def compute_signal_returns(y_true, y_prob, prices=None):
    """
    Simulates a simple signal-based strategy:
      - prob >= CONF_THRESHOLD → BUY  → gain if price went UP
      - prob <= HOLD_LOW       → SELL → gain if price went DOWN
      - else                   → HOLD → no position

    If prices available: uses actual 5-day returns.
    If not: uses binary +1 / -1 per correct / incorrect signal.

    Returns: array of per-period returns for the strategy.
    """
    n = len(y_prob)
    strategy_returns = np.zeros(n)

    for i in range(n):
        p      = y_prob[i]
        actual = y_true[i]   # 1 = price went UP, 0 = price went DOWN

        if prices is not None and i + 5 < len(prices):
            # Actual 5-day return
            ret = (prices.iloc[i + 5] - prices.iloc[i]) / prices.iloc[i]
        else:
            # Binary: correct = +1%, wrong = -1%
            ret = None

        if p >= CONF_THRESHOLD:        # BUY signal
            if ret is not None:
                strategy_returns[i] = ret if actual == 1 else -ret
            else:
                strategy_returns[i] = 0.01 if actual == 1 else -0.01

        elif p <= HOLD_LOW:            # SELL signal
            if ret is not None:
                strategy_returns[i] = -ret if actual == 0 else ret
            else:
                strategy_returns[i] = 0.01 if actual == 0 else -0.01
        # else HOLD → 0 return

    return strategy_returns


def compute_sharpe(returns, rf_daily=RISK_FREE_RATE/TRADING_DAYS):
    """Annualised Sharpe Ratio."""
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    excess = returns - rf_daily
    return float(np.mean(excess) / np.std(excess) * np.sqrt(TRADING_DAYS))


def compute_max_drawdown(cumulative):
    """Maximum peak-to-trough drawdown as a percentage."""
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / np.where(peak == 0, 1, peak)
    return float(drawdown.min() * 100)   # as %


def compute_win_loss_ratio(returns):
    """Average win / average loss (absolute)."""
    wins   = returns[returns > 0]
    losses = returns[returns < 0]
    if len(losses) == 0 or len(wins) == 0:
        return None
    return float(np.mean(wins) / np.abs(np.mean(losses)))


def compute_cumulative(returns, initial=INITIAL_CAPITAL):
    """Cumulative portfolio value starting from initial capital."""
    cum = initial * np.cumprod(1 + returns)
    return cum


# ─────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────

print("\n[3/7] Computing finance metrics per stock...\n")
print(f"  {'Stock':<15} {'AUC':>6} {'HitRate':>8} {'Sharpe':>8} {'MaxDD%':>8} {'W/L':>6} {'FinalVal':>10} {'Status'}")
print("  " + "-"*75)

all_rows = []
cumulative_curves = {}   # for plot 1

for stock in ALL_STOCKS:
    y_prob, best_model = get_probas(stock)
    y_true             = get_y_test(stock)
    prices             = get_prices(stock)
    stored_auc         = get_stored_auc(stock)
    validated          = stored_auc is not None and stored_auc >= AUC_GATE

    row = {
        "stock":      stock,
        "best_model": best_model,
        "stored_auc": round(stored_auc, 4) if stored_auc else None,
        "validated":  validated,
    }

    if y_true is None or y_prob is None:
        print(f"  {stock:<15} — skipping (data not found)")
        row["note"] = "data missing"
        all_rows.append(row)
        continue

    # Align lengths
    n = min(len(y_true), len(y_prob))
    y_true = np.array(y_true[-n:]).flatten()
    y_prob = np.array(y_prob[-n:]).flatten()

    # Align prices to test period (last n+5 days)
    prices_test = None
    if prices is not None and len(prices) >= n:
        prices_test = prices.iloc[-(n + 10):]   # extra buffer for 5-day forward

    # ── Metrics ──
    hit_rate         = compute_hit_rate(y_true, y_prob)
    strat_returns    = compute_signal_returns(y_true, y_prob, prices_test)
    bh_returns       = np.diff(prices_test.values) / prices_test.values[:-1] if prices_test is not None and len(prices_test) > 1 else np.zeros(n)
    bh_returns       = bh_returns[-n:] if len(bh_returns) >= n else bh_returns

    sharpe           = compute_sharpe(strat_returns)
    bh_sharpe        = compute_sharpe(bh_returns)

    cum_strat        = compute_cumulative(strat_returns)
    cum_bh           = compute_cumulative(bh_returns[:n]) if len(bh_returns) >= n else None

    max_dd           = compute_max_drawdown(cum_strat)
    win_loss         = compute_win_loss_ratio(strat_returns)
    final_val        = float(cum_strat[-1]) if len(cum_strat) > 0 else INITIAL_CAPITAL

    total_signals    = int(np.sum((y_prob >= CONF_THRESHOLD) | (y_prob <= HOLD_LOW)))
    buy_signals      = int(np.sum(y_prob >= CONF_THRESHOLD))
    sell_signals     = int(np.sum(y_prob <= HOLD_LOW))

    # Annualised return %
    n_periods        = len(strat_returns)
    ann_return       = float((final_val / INITIAL_CAPITAL) ** (TRADING_DAYS / max(n_periods, 1)) - 1) * 100

    row.update({
        "hit_rate":       round(hit_rate,  4),
        "sharpe_ratio":   round(sharpe,    4),
        "bh_sharpe":      round(bh_sharpe, 4),
        "max_drawdown_pct": round(max_dd,  2),
        "win_loss_ratio": round(win_loss,  4) if win_loss else None,
        "final_value":    round(final_val, 2),
        "ann_return_pct": round(ann_return, 2),
        "total_signals":  total_signals,
        "buy_signals":    buy_signals,
        "sell_signals":   sell_signals,
        "n_test_days":    n,
    })

    # Store curves for validated stocks
    if validated:
        cumulative_curves[stock] = {
            "strategy": cum_strat,
            "bh":       cum_bh,
        }

    status  = "✓" if validated else "⚠"
    wl_str  = f"{win_loss:.2f}" if win_loss else "—"
    print(f"  {stock:<15} {stored_auc:>6.3f} {hit_rate:>7.1%} {sharpe:>8.3f} {max_dd:>7.1f}% {wl_str:>6} {final_val:>9.1f}  {status}")
    all_rows.append(row)

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────

print("\n" + "="*65)
print("  FINANCE STATISTICS SUMMARY")
print("="*65)

valid_rows = [r for r in all_rows if r.get("validated") and r.get("sharpe_ratio") is not None]

if valid_rows:
    avg_sharpe = np.mean([r["sharpe_ratio"] for r in valid_rows])
    avg_hit    = np.mean([r["hit_rate"]     for r in valid_rows])
    avg_dd     = np.mean([r["max_drawdown_pct"] for r in valid_rows])
    best_s     = max(valid_rows, key=lambda r: r["sharpe_ratio"])
    best_h     = max(valid_rows, key=lambda r: r["hit_rate"])

    print(f"\n  Validated stocks analysed  : {len(valid_rows)}")
    print(f"  Average Sharpe Ratio       : {avg_sharpe:.3f}  (>1.0 = strong, >0.5 = acceptable)")
    print(f"  Average Hit Rate           : {avg_hit:.1%}")
    print(f"  Average Max Drawdown       : {avg_dd:.1f}%")
    print(f"  Best Sharpe                : {best_s['stock']} ({best_s['sharpe_ratio']:.3f})")
    print(f"  Best Hit Rate              : {best_h['stock']} ({best_h['hit_rate']:.1%})")

    print(f"\n  Per-stock breakdown:")
    for r in valid_rows:
        sharpe_label = "strong" if r["sharpe_ratio"] > 1.0 else "acceptable" if r["sharpe_ratio"] > 0.5 else "weak"
        print(f"    {r['stock']:<15}  Sharpe={r['sharpe_ratio']:>6.3f} ({sharpe_label})  "
              f"Hit={r['hit_rate']:.1%}  MaxDD={r['max_drawdown_pct']:.1f}%  "
              f"FinalVal={r['final_value']:.1f}")

# ─────────────────────────────────────────────────────────────
# SAVE CSV + JSON
# ─────────────────────────────────────────────────────────────

print("\n[4/7] Saving outputs...")

df = pd.DataFrame(all_rows)
df.to_csv("strategy_results.csv", index=False)
print("  ✓ strategy_results.csv")

with open("strategy_results.json", "w") as f:
    json.dump(all_rows, f, indent=2, default=str)
print("  ✓ strategy_results.json")

# ─────────────────────────────────────────────────────────────
# PLOT 1: Cumulative Returns — Strategy vs Buy-and-Hold
# ─────────────────────────────────────────────────────────────

print("\n[5/7] Generating Plot 1 — Cumulative Returns...")

if cumulative_curves:
    n_stocks = len(cumulative_curves)
    cols     = min(3, n_stocks)
    rows_n   = (n_stocks + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(6 * cols, 4 * rows_n))
    axes = np.array(axes).flatten() if n_stocks > 1 else [axes]

    colors_strat = "#2ecc71"
    colors_bh    = "#3498db"

    for idx, (stock, curves) in enumerate(cumulative_curves.items()):
        ax  = axes[idx]
        cum = curves["strategy"]
        bh  = curves["bh"]
        x   = np.arange(len(cum))

        ax.plot(x, cum, color=colors_strat, linewidth=2.0, label="QuantEdge Strategy")
        if bh is not None and len(bh) > 0:
            bh_x = np.arange(len(bh))
            ax.plot(bh_x, bh, color=colors_bh, linewidth=1.5,
                    linestyle="--", label="Buy & Hold", alpha=0.8)
        ax.axhline(INITIAL_CAPITAL, color="#bdc3c7", linestyle=":", linewidth=1)

        final     = cum[-1]
        ret_pct   = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        ret_label = f"+{ret_pct:.1f}%" if ret_pct >= 0 else f"{ret_pct:.1f}%"
        color     = "#27ae60" if ret_pct >= 0 else "#e74c3c"

        ax.set_title(f"{stock.replace('.NS','')}  →  {ret_label}",
                     fontsize=11, fontweight="bold", color=color)
        ax.set_xlabel("Test Day", fontsize=9)
        ax.set_ylabel(f"Portfolio Value (start={INITIAL_CAPITAL:.0f})", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for idx in range(len(cumulative_curves), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("QuantEdge — Cumulative Strategy Returns vs Buy-and-Hold\n"
                 "(Validated stocks only | 113 test days | Signal threshold = 60%)",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("strategy_plot_1_cumulative_returns.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ strategy_plot_1_cumulative_returns.png")

# ─────────────────────────────────────────────────────────────
# PLOT 2: Sharpe Ratio + Max Drawdown side by side
# ─────────────────────────────────────────────────────────────

print("\n[6/7] Generating Plot 2 — Sharpe Ratio & Max Drawdown...")

plot_rows = [r for r in all_rows if r.get("sharpe_ratio") is not None]

if plot_rows:
    p_stocks = [r["stock"].replace(".NS", "") for r in plot_rows]
    p_sharpe = [r["sharpe_ratio"]     for r in plot_rows]
    p_dd     = [abs(r["max_drawdown_pct"]) for r in plot_rows]
    p_colors = ["#2ecc71" if r["validated"] else "#e74c3c" for r in plot_rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(p_stocks))

    # Sharpe
    bars1 = ax1.bar(x, p_sharpe, color=p_colors, alpha=0.85, width=0.55, zorder=3)
    ax1.axhline(1.0, color="#e67e22", linestyle="--", linewidth=1.8, label="Strong threshold (1.0)")
    ax1.axhline(0.5, color="#f39c12", linestyle=":",  linewidth=1.5, label="Acceptable threshold (0.5)")
    ax1.axhline(0.0, color="#bdc3c7", linestyle="-",  linewidth=1.0)
    for bar, val in zip(bars1, p_sharpe):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(p_stocks, rotation=25, ha="right", fontsize=9)
    ax1.set_ylabel("Annualised Sharpe Ratio", fontsize=11)
    ax1.set_title("Sharpe Ratio per Stock\n(higher = better risk-adjusted return)",
                  fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3, zorder=0)

    # Max Drawdown
    bars2 = ax2.bar(x, p_dd, color=p_colors, alpha=0.85, width=0.55, zorder=3)
    ax2.axhline(10, color="#e67e22", linestyle="--", linewidth=1.8, label="10% caution line")
    for bar, val in zip(bars2, p_dd):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(p_stocks, rotation=25, ha="right", fontsize=9)
    ax2.set_ylabel("Max Drawdown (%)", fontsize=11)
    ax2.set_title("Maximum Drawdown per Stock\n(lower = less risk | strategy simulation)",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3, zorder=0)

    legend_patches = [
        mpatches.Patch(color="#2ecc71", label="Validated (AUC ≥ 0.60)"),
        mpatches.Patch(color="#e74c3c", label="Suppressed (AUC < 0.60)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    plt.savefig("strategy_plot_2_sharpe_drawdown.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ strategy_plot_2_sharpe_drawdown.png")

# ─────────────────────────────────────────────────────────────
# PLOT 3: Hit Rate comparison
# ─────────────────────────────────────────────────────────────

print("\n[7/7] Generating Plot 3 — Hit Rate Analysis...")

hr_rows = [r for r in all_rows if r.get("hit_rate") is not None]

if hr_rows:
    h_stocks   = [r["stock"].replace(".NS", "") for r in hr_rows]
    h_hitrates = [r["hit_rate"] * 100           for r in hr_rows]
    h_colors   = ["#2ecc71" if r["validated"] else "#e74c3c" for r in hr_rows]

    fig, ax = plt.subplots(figsize=(12, 6))
    x    = np.arange(len(h_stocks))
    bars = ax.bar(x, h_hitrates, color=h_colors, alpha=0.85, width=0.55, zorder=3)

    ax.axhline(50, color="#e74c3c", linestyle="--", linewidth=1.8,
               label="Random baseline = 50%", zorder=5)
    ax.axhline(60, color="#e67e22", linestyle=":",  linewidth=1.5,
               label="Good threshold = 60%", zorder=5)

    for bar, val in zip(bars, h_hitrates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(h_stocks, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Hit Rate (%)", fontsize=12)
    ax.set_ylim(30, 90)
    ax.set_title("QuantEdge — Prediction Hit Rate per Stock\n"
                 "(% of all predictions where price moved in predicted direction)",
                 fontsize=12, fontweight="bold", pad=15)

    legend_patches = [
        mpatches.Patch(color="#2ecc71", label="Validated (AUC ≥ 0.60)"),
        mpatches.Patch(color="#e74c3c", label="Suppressed (AUC < 0.60)"),
    ]
    ax.legend(handles=legend_patches + ax.get_legend_handles_labels()[0],
              loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    plt.tight_layout()
    plt.savefig("strategy_plot_3_hit_rate.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ strategy_plot_3_hit_rate.png")

# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────

print("\n" + "="*65)
print("  OUTPUT FILES GENERATED")
print("="*65)
print("  strategy_results.csv                   — full metrics table")
print("  strategy_results.json                  — for technical report")
print("  strategy_plot_1_cumulative_returns.png — strategy vs buy-and-hold")
print("  strategy_plot_2_sharpe_drawdown.png    — risk-adjusted metrics")
print("  strategy_plot_3_hit_rate.png           — prediction hit rate")

print("\n" + "="*65)
print("  REPORT-READY STATEMENT")
print("="*65)
print("""
  'Strategy performance was evaluated using quantitative finance
  metrics on 113 unseen test days. Sharpe Ratio was computed
  using annualised returns relative to a 6.5% risk-free rate
  (Indian 10-year bond). Hit Rate measures directional accuracy
  across all predictions. Maximum Drawdown quantifies the worst
  simulated loss from peak capital during the test period.
  Cumulative return curves compare the QuantEdge signal strategy
  against a passive buy-and-hold benchmark for each validated stock.'
""")
print("="*65 + "\n")
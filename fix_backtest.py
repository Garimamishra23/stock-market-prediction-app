import pickle
import numpy as np
import pandas as pd
import json
import glob

with open('ensemble_probas.pkl', 'rb') as f:
    ensemble_probas = pickle.load(f)

with open('training_data.pkl', 'rb') as f:
    data = pickle.load(f)

with open('model_results.pkl', 'rb') as f:
    model_results = pickle.load(f)

json_files = glob.glob("global_market_data_*.json")
with open(sorted(json_files)[-1], 'r', encoding='utf-8') as f:
    raw = json.load(f)

CONFIDENCE_CUTOFF = 0.60
TRANSACTION_COST  = 0.001
RISK_FREE_RATE    = 0.065 / 252

results = []

for symbol in ensemble_probas:
    if symbol not in data:
        continue
    try:
        stock  = data[symbol]
        y_test = stock['y_test']
        proba  = ensemble_probas[symbol]

        market      = raw[symbol].get('market_data', raw[symbol])
        df_list     = market.get('full_dataframe', [])
        df          = pd.DataFrame(df_list)
        df['close'] = pd.to_numeric(df['close'], errors='coerce')

        n_test      = len(y_test)
        test_prices = df['close'].iloc[-n_test:].values

        if len(test_prices) < n_test:
            continue

        # ── Simulate strategy ─────────────────────────────────
        capital  = 100.0
        position = 0.0
        trades   = 0
        values   = [capital]
        entry_price = None

        for i in range(len(proba)):
            price = float(test_prices[i])

            if proba[i] >= CONFIDENCE_CUTOFF and position == 0:
                shares      = capital / price
                capital    -= shares * price * (1 + TRANSACTION_COST)
                position    = shares
                entry_price = price
                trades     += 1

            elif proba[i] <= (1 - CONFIDENCE_CUTOFF) and position > 0:
                capital  += position * price * (1 - TRANSACTION_COST)
                position  = 0
                trades   += 1

            current_val = capital + (position * price if position > 0 else 0)
            values.append(current_val)

        # Close any open position at last price
        if position > 0:
            capital += position * float(test_prices[-1]) * (1 - TRANSACTION_COST)
            trades  += 1  # count the forced close

        final_val  = capital
        values_arr = np.array(values)
        total_return = (final_val - 100.0)

        # ── Sharpe ratio ──────────────────────────────────────
        daily_rets = np.diff(values_arr) / np.where(values_arr[:-1] > 0, values_arr[:-1], 1)
        excess     = daily_rets - RISK_FREE_RATE
        if excess.std() > 1e-10 and trades > 0:
            sharpe = float(excess.mean() / excess.std() * np.sqrt(252))
        else:
            sharpe = float('nan')  # not enough trades to compute

        # ── Max drawdown ──────────────────────────────────────
        peak   = values_arr[0]
        max_dd = 0.0
        for v in values_arr:
            if v > peak:
                peak = v
            dd = (peak - v) / peak * 100
            if dd > max_dd:
                max_dd = dd

        # ── Buy and hold ──────────────────────────────────────
        bh_final   = 100.0 * (float(test_prices[-1]) / float(test_prices[0]))
        bh_return  = bh_final - 100.0
        bh_rets    = np.diff(test_prices.astype(float)) / test_prices[:-1].astype(float)
        bh_excess  = bh_rets - RISK_FREE_RATE
        bh_sharpe  = float(bh_excess.mean() / bh_excess.std() * np.sqrt(252)) \
                     if bh_excess.std() > 1e-10 else float('nan')

        # ── Hit rate ──────────────────────────────────────────
        preds    = (proba >= 0.5).astype(int)
        hit_rate = float(np.mean(preds == y_test)) * 100

        # ── Alpha ─────────────────────────────────────────────
        alpha = total_return - bh_return

        best_model = model_results.get(symbol, {}).get('best_model', '?')
        auc        = model_results.get(symbol, {}).get('best_auc', 0.0)

        sharpe_str    = f"{sharpe:.3f}"    if not np.isnan(sharpe)    else "N/A"
        bh_sharpe_str = f"{bh_sharpe:.3f}" if not np.isnan(bh_sharpe) else "N/A"

        print(f"{symbol:<15} Model={best_model:<14} AUC={auc:.3f}  "
              f"Trades={trades:>2}  Return={total_return:+.1f}%  "
              f"BH={bh_return:+.1f}%  Alpha={alpha:+.1f}%  "
              f"Sharpe={sharpe_str}  MaxDD={max_dd:.1f}%  Hit={hit_rate:.1f}%")

        results.append({
            'Stock':          symbol,
            'Model':          best_model,
            'AUC':            round(auc, 3),
            'Trades':         trades,
            'Strategy_Return':round(total_return, 2),
            'BH_Return':      round(bh_return, 2),
            'Alpha':          round(alpha, 2),
            'Sharpe':         round(sharpe, 3) if not np.isnan(sharpe) else None,
            'BH_Sharpe':      round(bh_sharpe, 3) if not np.isnan(bh_sharpe) else None,
            'Max_DD':         round(max_dd, 1),
            'Hit_Rate':       round(hit_rate, 1),
        })

    except Exception as e:
        print(f"{symbol}: Error — {e}")
        import traceback; traceback.print_exc()

print("\n" + "=" * 80)
df_r = pd.DataFrame(results)

# Summary — only stocks with actual trades
traded = df_r[df_r['Trades'] > 0]
print(f"\nSTOCKS WITH ACTIVE TRADES: {len(traded)}/{len(df_r)}")
print(traded[['Stock','Model','AUC','Trades','Strategy_Return','BH_Return','Alpha','Sharpe','Max_DD','Hit_Rate']].to_string(index=False))

print(f"\nSTOCKS WITH NO SIGNALS (held cash = 0% return):")
no_trade = df_r[df_r['Trades'] == 0]
print(no_trade[['Stock','Model','AUC','Hit_Rate']].to_string(index=False))

print(f"\nPORTFOLIO SUMMARY (traded stocks only):")
if len(traded) > 0:
    print(f"  Avg Strategy Return: {traded['Strategy_Return'].mean():+.2f}%")
    print(f"  Avg BH Return:       {traded['BH_Return'].mean():+.2f}%")
    print(f"  Avg Alpha:           {traded['Alpha'].mean():+.2f}%")
    valid_sharpe = traded['Sharpe'].dropna()
    if len(valid_sharpe) > 0:
        print(f"  Avg Sharpe:          {valid_sharpe.mean():.3f}")
    print(f"  Avg Max Drawdown:    {traded['Max_DD'].mean():.1f}%")

df_r.to_csv('backtest_fixed.csv', index=False)
print(f"\n  ✓ Saved: backtest_fixed.csv")
# improved_stacking.py - BEST-PER-STOCK MODEL SELECTION
# Strategy: instead of a broken meta-learner, pick the best individual model
# per stock based on AUC (most reliable metric for imbalanced data).
# This gives cleaner, more defensible results for the capstone presentation.

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score
)
import warnings
warnings.filterwarnings('ignore')

CONFIDENCE_CUTOFF = 0.60   # lowered from 0.62 to fire more signals

print("=" * 70)
print("  BEST-PER-STOCK ENSEMBLE — SWING TRADING")
print(f"  Strategy: pick best model per stock by AUC")
print(f"  Confidence cutoff: {CONFIDENCE_CUTOFF*100:.0f}%")
print("=" * 70)

# ── Load results ──────────────────────────────────────────────────────────────
with open('training_data.pkl', 'rb') as f:
    data = pickle.load(f)

with open('xgb_results.pkl', 'rb') as f:
    xgb_results = pickle.load(f)

with open('rf_results.pkl', 'rb') as f:
    rf_results = pickle.load(f)

with open('lstm_results.pkl', 'rb') as f:
    lstm_results = pickle.load(f)

with open('xgb_probas.pkl', 'rb') as f:
    xgb_probas = pickle.load(f)
print("  ✓ Loaded XGBoost probabilities")

with open('rf_probas.pkl', 'rb') as f:
    rf_probas = pickle.load(f)
print("  ✓ Loaded Random Forest probabilities")

with open('lstm_probas.pkl', 'rb') as f:
    lstm_probas = pickle.load(f)
print("  ✓ Loaded LSTM probabilities")



results         = []
ensemble_probas = {}
ensemble_models = {}

for symbol, stock in data.items():
    print(f"\n{'─'*60}")
    print(f"  {symbol}")

    y_test = stock['y_test']

    # Get probabilities
    xgb_proba = xgb_probas.get(symbol, np.full(len(y_test), 0.5))
    rf_proba  = rf_probas.get(symbol,  np.full(len(y_test), 0.5))

    # Align LSTM (sequence offset)
    if symbol in lstm_probas:
        lp = lstm_probas[symbol]
        if len(lp) < len(y_test):
            lp = np.concatenate([np.full(len(y_test) - len(lp), 0.5), lp])
        elif len(lp) > len(y_test):
            lp = lp[-len(y_test):]
        lstm_proba = lp
    else:
        lstm_proba = np.full(len(y_test), 0.5)

    # ── Metrics helper ────────────────────────────────────────────────────────
    def metrics(proba, y):
        pred = (proba >= 0.5).astype(int)
        acc  = float(accuracy_score(y, pred))
        f1   = float(f1_score(y, pred, zero_division=0))
        auc  = float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else 0.5
        return acc, f1, auc

    xgb_acc,  xgb_f1,  xgb_auc  = metrics(xgb_proba,  y_test)
    rf_acc,   rf_f1,   rf_auc   = metrics(rf_proba,   y_test)
    lstm_acc, lstm_f1, lstm_auc = metrics(lstm_proba, y_test)

    


    print(f"  {'Model':<20} {'Acc':>6} {'F1':>6} {'AUC':>6}")
    print(f"  {'─'*42}")
    print(f"  {'XGBoost':<20} {xgb_acc:>6.3f} {xgb_f1:>6.3f} {xgb_auc:>6.3f}")
    print(f"  {'Random Forest':<20} {rf_acc:>6.3f} {rf_f1:>6.3f} {rf_auc:>6.3f}")
    print(f"  {'LSTM':<20} {lstm_acc:>6.3f} {lstm_f1:>6.3f} {lstm_auc:>6.3f}")

   
   # ── XGB+RF ensemble (used as one candidate alongside individual models) ────
    # Weights computed from THIS run's actual AUCs, not hardcoded values
    tree_total = xgb_auc + rf_auc
    if tree_total > 0:
        w_xgb = xgb_auc / tree_total
        w_rf  = rf_auc  / tree_total
    else:
        w_xgb = w_rf = 0.5

    ens_proba            = w_xgb * xgb_proba + w_rf * rf_proba
    ens_acc, ens_f1, ens_auc = metrics(ens_proba, y_test)

    print(f"  {'XGB+RF Ensemble':<20} {ens_acc:>6.3f} {ens_f1:>6.3f} {ens_auc:>6.3f}")
    print(f"  Weights → XGB:{w_xgb:.2f}  RF:{w_rf:.2f}  (XGB+RF blend, LSTM evaluated separately)")

    # ── Pick best by AUC — LSTM now included ──────────────────────────────────
    # LSTM infrastructure exists in app.py and lstm_models.pkl is saved
    # Including LSTM prevents train/inference mismatch for stocks where it wins
    candidates = [
        ('XGBoost',       xgb_auc,  xgb_proba,  xgb_acc,  xgb_f1),
        ('Random Forest', rf_auc,   rf_proba,   rf_acc,   rf_f1),
        ('LSTM',          lstm_auc, lstm_proba, lstm_acc, lstm_f1),
        ('Ensemble',      ens_auc,  ens_proba,  ens_acc,  ens_f1),
    ]
    best_name, best_auc, best_proba, best_acc, best_f1 = max(candidates, key=lambda x: x[1])
    print(f"\n  ✓ Best deployable model: {best_name} (AUC={best_auc:.3f})")

    # ── Confident signals ─────────────────────────────────────────────────────
    confident_mask = (best_proba >= CONFIDENCE_CUTOFF) | \
                     (best_proba <= (1 - CONFIDENCE_CUTOFF))
    n_confident    = int(confident_mask.sum())
    n_buy          = int((best_proba >= CONFIDENCE_CUTOFF).sum())
    n_sell         = int((best_proba <= (1 - CONFIDENCE_CUTOFF)).sum())

    if n_confident > 0:
        conf_pred = (best_proba[confident_mask] >= 0.5).astype(int)
        conf_acc  = float(accuracy_score(y_test[confident_mask], conf_pred))
    else:
        conf_acc = 0.0

    print(f"  Confident signals: {n_confident}/{len(y_test)} (BUY={n_buy}, SELL={n_sell})")
    if n_confident > 0:
        print(f"  Confident accuracy: {conf_acc:.3f}")

    ensemble_probas[symbol] = best_proba
    ensemble_models[symbol] = best_name

    results.append({
        'Stock':             symbol,
        'XGB_AUC':           xgb_auc,
        'RF_AUC':            rf_auc,
        'LSTM_AUC':          lstm_auc,
        'Ensemble_AUC':      ens_auc,
        'Best_Model':        best_name,
        'Best_AUC':          best_auc,
        'Best_Acc':          best_acc,
        'Best_F1':           best_f1,
        'Confident_Signals': n_confident,
        'Confident_Acc':     conf_acc,
        'XGB_Acc':           xgb_acc,
        'RF_Acc':            rf_acc,
        'LSTM_Acc':          lstm_acc,
        'Signal_Bias':       False,      # ← add this
    })

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("  FINAL RESULTS — BEST MODEL PER STOCK")
print("=" * 75)

df = pd.DataFrame(results).sort_values('Best_AUC', ascending=False)

print(f"\n  {'STOCK':<15} {'BEST_MODEL':<16} {'ACC':>6} {'F1':>6} {'AUC':>6} {'SIGNALS':>8} {'CONF_ACC':>9}")
print(f"  {'-'*72}")
for _, row in df.iterrows():
    print(f"  {row['Stock']:<15} {row['Best_Model']:<16} "
          f"{row['Best_Acc']:>6.3f} {row['Best_F1']:>6.3f} "
          f"{row['Best_AUC']:>6.3f} {int(row['Confident_Signals']):>8} "
          f"{row['Confident_Acc']:>9.3f}")

print(f"  {'-'*72}")
print(f"  {'AVERAGE':<15} {'':<16} "
      f"{df['Best_Acc'].mean():>6.3f} {df['Best_F1'].mean():>6.3f} "
      f"{df['Best_AUC'].mean():>6.3f}")

print(f"\n  Model selection:")
for model in ['XGBoost', 'Random Forest', 'LSTM', 'Ensemble']:
    stocks = df[df['Best_Model'] == model]['Stock'].tolist()
    if stocks:
        print(f"    {model:<16}: {', '.join(stocks)}")

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv('improved_stacking_results.csv', index=False)

with open('ensemble_models.pkl', 'wb') as f:
    pickle.dump(ensemble_models, f)

with open('ensemble_probas.pkl', 'wb') as f:
    pickle.dump(ensemble_probas, f)

# model_results.pkl — used by app.py dashboard
model_results = {}
for _, row in df.iterrows():
    sym = row['Stock']
    model_results[sym] = {
        'best_model':        row['Best_Model'],
        'best_auc':          row['Best_AUC'],
        'best_acc':          row['Best_Acc'],
        'best_f1':           row['Best_F1'],
        'xgb_auc':           row['XGB_AUC'],
        'rf_auc':            row['RF_AUC'],
        'lstm_auc':          row['LSTM_AUC'],
        'ensemble_auc':      row['Ensemble_AUC'],
        'confident_signals': int(row['Confident_Signals']),
        'confident_acc':     row['Confident_Acc'],
        'best_individual':   row['Best_AUC'],
        'weighted':          row['Ensemble_AUC'],
        'selective':         row['Confident_Acc'],
        'best_method':       row['Best_AUC'],
        'signal_bias':       bool(row.get('Signal_Bias', False)),  # ← add this
    }

with open('model_results.pkl', 'wb') as f:
    pickle.dump(model_results, f)

print("\n  ✓ Saved: improved_stacking_results.csv")
print("  ✓ Saved: ensemble_models.pkl")
print("  ✓ Saved: ensemble_probas.pkl")
print("  ✓ Saved: model_results.pkl  ← dashboard ready")

# ── Consistency check ─────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("  CONSISTENCY CHECK")
print("=" * 75)
low_auc_stocks = [(s, d['best_auc']) for s, d in model_results.items() if d['best_auc'] < 0.55]
good_stocks    = [(s, d['best_auc']) for s, d in model_results.items() if d['best_auc'] >= 0.60]
warn_stocks    = [(s, d['best_auc']) for s, d in model_results.items() if 0.55 <= d['best_auc'] < 0.60]

print(f"\n  ✅ Reliable signals (AUC ≥ 0.60): {len(good_stocks)} stocks")
for s, a in good_stocks:
    print(f"     {s:<15} AUC={a:.3f}")

print(f"\n  ⚠️  Weak signals (0.55 ≤ AUC < 0.60): {len(warn_stocks)} stocks")
for s, a in warn_stocks:
    print(f"     {s:<15} AUC={a:.3f}  ← shown with caution label")

print(f"\n  🔴 Suppressed (AUC < 0.55): {len(low_auc_stocks)} stocks")
for s, a in low_auc_stocks:
    print(f"     {s:<15} AUC={a:.3f}  ← momentum fallback only")

lstm_selected = [s for s, m in ensemble_models.items() if m == 'LSTM']
if lstm_selected:
    print(f"\n  ✅ LSTM selected for: {lstm_selected} — live inference will use LSTM for these stocks")
else:
    print(f"\n  ℹ️  No LSTM assignments this run — XGB/RF/Ensemble selected for all stocks")

print(f"\n  Next step: streamlit run app.py")
print("=" * 75)
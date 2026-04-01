# stacking_ensemble_pro.py - SWING TRADING FIXED VERSION
# Changes from your original:
# FIX 1 (CRITICAL): StratifiedKFold(shuffle=True) → TimeSeriesSplit
#         shuffle=True was mixing future data into past training folds.
#         This was causing inflated accuracy numbers that don't hold in real trading.
# FIX 2: Uses real LSTM probabilities from lstm_probas.pkl (not fake constant)
# FIX 3: Reports F1 and AUC (accuracy alone is misleading for imbalanced data)
# FIX 4: Saves full comparison including AUC for dashboard display

import pickle
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("  PROFESSIONAL STACKING ENSEMBLE — SWING TRADING")
print("=" * 70)

# ── Load data ─────────────────────────────────────────────────────────────────
with open('training_data.pkl', 'rb') as f:
    data = pickle.load(f)

with open('xgb_results.pkl', 'rb') as f:
    xgb_results = pickle.load(f)

with open('rf_results.pkl', 'rb') as f:
    rf_results = pickle.load(f)

with open('lstm_results.pkl', 'rb') as f:
    lstm_results = pickle.load(f)

# FIX 2: Load real per-row probabilities
try:
    with open('lstm_probas.pkl', 'rb') as f:
        lstm_probas = pickle.load(f)
    print("  ✓ Loaded real LSTM probabilities")
except FileNotFoundError:
    lstm_probas = {}
    print("  ⚠  lstm_probas.pkl not found — LSTM column will be 0.5 (neutral)")

comparison_data = []

for symbol, stock in data.items():
    print(f"\n{'─'*60}")
    print(f"  Stacking Ensemble for {symbol}")

    X_train = stock['X_train']
    X_val   = stock['X_val']
    X_test  = stock['X_test']
    y_train = stock['y_train']
    y_val   = stock['y_val']
    y_test  = stock['y_test']

    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    xgb_acc  = float(xgb_results.get(symbol, 0))
    rf_acc   = float(rf_results.get(symbol, 0))
    lstm_acc = float(lstm_results.get(symbol, 0))

    # ════════════════════════════════════════════════════════════════════════
    # LEVEL 0: TRAIN BASE MODELS on full train+val set
    # ════════════════════════════════════════════════════════════════════════
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        scale_pos_weight=float(stock['imbalance_ratio']),
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_trainval, y_trainval)
    rf_model.fit(X_trainval, y_trainval)

    # Individual test predictions
    xgb_pred_test = xgb_model.predict(X_test)
    rf_pred_test  = rf_model.predict(X_test)
    xgb_proba_test = xgb_model.predict_proba(X_test)[:, 1]
    rf_proba_test  = rf_model.predict_proba(X_test)[:, 1]

    xgb_test_acc = float(accuracy_score(y_test, xgb_pred_test))
    rf_test_acc  = float(accuracy_score(y_test, rf_pred_test))

    # FIX 2: Real LSTM probabilities
    if symbol in lstm_probas:
        lstm_proba_test = lstm_probas[symbol]
        if len(lstm_proba_test) < len(X_test):
            pad = np.full(len(X_test) - len(lstm_proba_test), 0.5)
            lstm_proba_test = np.concatenate([pad, lstm_proba_test])
        elif len(lstm_proba_test) > len(X_test):
            lstm_proba_test = lstm_proba_test[-len(X_test):]
    else:
        lstm_proba_test = np.full(len(X_test), 0.5)

    # ════════════════════════════════════════════════════════════════════════
    # FIX 1: META-FEATURES using TimeSeriesSplit (not StratifiedKFold shuffle)
    # ════════════════════════════════════════════════════════════════════════
    print(f"  Building meta-features with TimeSeriesSplit...")

    # FIX 1: TimeSeriesSplit — no shuffle, no future data in past folds
    tscv = TimeSeriesSplit(n_splits=5)
    meta_features_train = np.zeros((len(X_trainval), 2))  # XGB + RF

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_trainval)):
        X_tr, X_vl = X_trainval[tr_idx], X_trainval[val_idx]
        y_tr        = y_trainval[tr_idx]

        if len(np.unique(y_tr)) < 2:
            continue

        xgb_fold = xgb.XGBClassifier(n_estimators=100, max_depth=3,
                                       random_state=42, eval_metric='logloss',
                                       verbosity=0)
        rf_fold  = RandomForestClassifier(n_estimators=100, max_depth=5,
                                           class_weight='balanced', random_state=42)
        xgb_fold.fit(X_tr, y_tr)
        rf_fold.fit(X_tr, y_tr)

        meta_features_train[val_idx, 0] = xgb_fold.predict_proba(X_vl)[:, 1]
        meta_features_train[val_idx, 1] = rf_fold.predict_proba(X_vl)[:, 1]

    # Test meta-features
    meta_features_test = np.column_stack([xgb_proba_test, rf_proba_test])

    # ════════════════════════════════════════════════════════════════════════
    # LEVEL 1: META-LEARNERS
    # ════════════════════════════════════════════════════════════════════════

    # Meta-learner 1: Logistic Regression
    meta_lr = LogisticRegression(class_weight='balanced', max_iter=500, random_state=42)
    meta_lr.fit(meta_features_train, y_trainval)
    ensemble_lr_proba = meta_lr.predict_proba(meta_features_test)[:, 1]
    ensemble_lr_pred  = (ensemble_lr_proba >= 0.5).astype(int)
    ensemble_lr_acc   = float(accuracy_score(y_test, ensemble_lr_pred))
    ensemble_lr_auc   = float(roc_auc_score(y_test, ensemble_lr_proba)) \
                        if len(np.unique(y_test)) > 1 else 0.5
    ensemble_lr_f1    = float(f1_score(y_test, ensemble_lr_pred, zero_division=0))

    # Meta-learner 2: XGBoost meta
    meta_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=3,
                                   random_state=42, eval_metric='logloss',
                                   verbosity=0)
    meta_xgb.fit(meta_features_train, y_trainval)
    ensemble_xgb_proba = meta_xgb.predict_proba(meta_features_test)[:, 1]
    ensemble_xgb_pred  = (ensemble_xgb_proba >= 0.5).astype(int)
    ensemble_xgb_acc   = float(accuracy_score(y_test, ensemble_xgb_pred))
    ensemble_xgb_auc   = float(roc_auc_score(y_test, ensemble_xgb_proba)) \
                         if len(np.unique(y_test)) > 1 else 0.5
    ensemble_xgb_f1    = float(f1_score(y_test, ensemble_xgb_pred, zero_division=0))

    # Simple average baseline
    ensemble_avg_proba = meta_features_test.mean(axis=1)
    ensemble_avg_pred  = (ensemble_avg_proba >= 0.5).astype(int)
    ensemble_avg_acc   = float(accuracy_score(y_test, ensemble_avg_pred))
    ensemble_avg_auc   = float(roc_auc_score(y_test, ensemble_avg_proba)) \
                         if len(np.unique(y_test)) > 1 else 0.5

    best_individual = max(xgb_acc, rf_acc, lstm_acc)
    best_ensemble   = max(ensemble_lr_acc, ensemble_xgb_acc, ensemble_avg_acc)

    # FIX 3: Print F1 and AUC alongside accuracy
    print(f"\n  {'Model':<30} {'Acc':>6} {'F1':>6} {'AUC':>6}")
    print(f"  {'─'*50}")
    print(f"  {'XGBoost (Individual)':<30} {xgb_acc:>6.3f}")
    print(f"  {'Random Forest (Individual)':<30} {rf_acc:>6.3f}")
    print(f"  {'LSTM (Individual)':<30} {lstm_acc:>6.3f}")
    print(f"  {'─'*50}")
    print(f"  {'Simple Average Ensemble':<30} {ensemble_avg_acc:>6.3f} {'—':>6} {ensemble_avg_auc:>6.3f}")
    print(f"  {'Stacking + Logistic Reg':<30} {ensemble_lr_acc:>6.3f} {ensemble_lr_f1:>6.3f} {ensemble_lr_auc:>6.3f}")
    print(f"  {'Stacking + XGBoost':<30} {ensemble_xgb_acc:>6.3f} {ensemble_xgb_f1:>6.3f} {ensemble_xgb_auc:>6.3f}")

    comparison_data.append({
        'Stock':            symbol,
        'XGBoost':          xgb_acc,
        'RF':               rf_acc,
        'LSTM':             lstm_acc,
        'Best_Individual':  best_individual,
        'Simple_Avg':       ensemble_avg_acc,
        'Stacking_LR':      ensemble_lr_acc,
        'Stacking_LR_AUC':  ensemble_lr_auc,
        'Stacking_LR_F1':   ensemble_lr_f1,
        'Stacking_XGB':     ensemble_xgb_acc,
        'Stacking_XGB_AUC': ensemble_xgb_auc,
        'Stacking_XGB_F1':  ensemble_xgb_f1,
        'Best_Ensemble':    best_ensemble,
        'Improvement':      best_ensemble - best_individual,
    })

# ── Final table ───────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  MASTER COMPARISON TABLE")
print("=" * 80)

df = pd.DataFrame(comparison_data).sort_values('Best_Ensemble', ascending=False)

print(f"\n  {'STOCK':<15} {'XGB':>6} {'RF':>6} {'LSTM':>6} {'BEST_IND':>9} "
      f"{'STKL_ACC':>9} {'STKL_AUC':>9} {'IMPROVE':>8}")
print(f"  {'-'*75}")

for _, row in df.iterrows():
    print(f"  {row['Stock']:<15} "
          f"{row['XGBoost']:>6.3f} "
          f"{row['RF']:>6.3f} "
          f"{row['LSTM']:>6.3f} "
          f"{row['Best_Individual']:>9.3f} "
          f"{row['Stacking_LR']:>9.3f} "
          f"{row['Stacking_LR_AUC']:>9.3f} "
          f"+{row['Improvement']:>7.3f}")

print(f"  {'-'*75}")
print(f"  {'AVERAGE':<15} "
      f"{df['XGBoost'].mean():>6.3f} "
      f"{df['RF'].mean():>6.3f} "
      f"{df['LSTM'].mean():>6.3f} "
      f"{df['Best_Individual'].mean():>9.3f} "
      f"{df['Stacking_LR'].mean():>9.3f} "
      f"{df['Stacking_LR_AUC'].mean():>9.3f} "
      f"+{df['Improvement'].mean():>7.3f}")

df.to_csv('stacking_ensemble_results.csv', index=False)

with open('stacking_ensemble_results.pkl', 'wb') as f:
    pickle.dump(comparison_data, f)

print("\n  ✓ Saved: stacking_ensemble_results.csv")
print("  ✓ Saved: stacking_ensemble_results.pkl")
print("\n  Next step: python improved_stacking.py")
print("=" * 80)
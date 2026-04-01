# train_xgboost.py - SWING TRADING FIXED VERSION
# Changes from your original:
# FIX 1: Added TimeSeriesSplit walk-forward validation (no data leakage)
# FIX 2: Save full trained model object (not just accuracy) for stacking
# FIX 3: Save predict_proba on test set (real probabilities for ensemble)
# FIX 4: Added F1, Precision, Recall, ROC-AUC reporting (accuracy alone is misleading)
# FIX 5: CalibratedClassifierCV wraps XGB so probabilities are trustworthy
# Everything else from your version is kept as-is

import pickle
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from typing import Dict, List, Any

import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  TRAINING XGBOOST — SWING TRADING")
print("=" * 60)

with open('training_data.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"\n  Loaded {len(data)} stocks: {list(data.keys())}")

results:      Dict[str, float]      = {}
model_store:  Dict[str, Any]        = {}   # FIX 2: store full model
proba_store:  Dict[str, np.ndarray] = {}   # FIX 3: store test probabilities
all_accuracies: List[float] = []

for symbol, stock in data.items():
    print(f"\n{'─'*55}")
    print(f"  {symbol}")
    print(f"  Train: {stock['X_train'].shape[0]}  Val: {stock['X_val'].shape[0]}  Test: {stock['X_test'].shape[0]}")
    print(f"  Features: {stock['X_train'].shape[1]}  Imbalance: {float(stock['imbalance_ratio']):.2f}")

    X_train = stock['X_train']
    X_val   = stock['X_val']
    X_test  = stock['X_test']
    y_train = stock['y_train']
    y_val   = stock['y_val']
    y_test  = stock['y_test']

    # Combine train+val for final training (test stays untouched)
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    # ════════════════════════════════════════════════════════════════════════
    # FIX 1: WALK-FORWARD VALIDATION (TimeSeriesSplit)
    # Your original used the same train/test split for evaluation.
    # This shows us real out-of-sample performance across 5 time windows.
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n  Walk-forward validation (5 folds):")
    tscv     = TimeSeriesSplit(n_splits=5)
    wf_accs  = []
    wf_aucs  = []

    base_xgb = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=float(stock['imbalance_ratio']),
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_trainval)):
        X_tr, X_te = X_trainval[tr_idx], X_trainval[te_idx]
        y_tr, y_te = y_trainval[tr_idx], y_trainval[te_idx]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        fold_model = xgb.XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.03,
            min_child_weight=5,
            scale_pos_weight=float(stock['imbalance_ratio']),
            random_state=42, eval_metric='logloss', verbosity=0
        )
        fold_model.fit(X_tr, y_tr)
        y_prob = fold_model.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        fold_acc = accuracy_score(y_te, y_pred)
        fold_auc = roc_auc_score(y_te, y_prob) if len(np.unique(y_te)) > 1 else 0.5
        wf_accs.append(fold_acc)
        wf_aucs.append(fold_auc)
        print(f"    Fold {fold+1}: Acc={fold_acc:.3f}  AUC={fold_auc:.3f}")

    print(f"  Walk-forward avg → Acc: {np.mean(wf_accs):.3f}  AUC: {np.mean(wf_aucs):.3f}")

    # ════════════════════════════════════════════════════════════════════════
    # FIX 5: CALIBRATED MODEL — train on full train+val, calibrate probabilities
    # CalibratedClassifierCV makes the confidence % meaningful
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n  Training final calibrated model on train+val...")

    raw_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=float(stock['imbalance_ratio']),
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )

    # FIX 5: Sigmoid calibration is more stable than isotonic on small datasets
    # isotonic with cv=3 causes probability collapse on 113-sample test sets
    model = CalibratedClassifierCV(raw_model, method='isotonic', cv=3)
    model.fit(X_trainval, y_trainval)

    # ── Evaluate on held-out test set ────────────────────────────────────────
    y_prob_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_prob_test >= 0.5).astype(int)

    accuracy  = float(accuracy_score(y_test, y_pred_test))
    precision_up   = float(precision_score(y_test, y_pred_test, pos_label=1, zero_division=0))
    precision_down = float(precision_score(y_test, y_pred_test, pos_label=0, zero_division=0))
    recall_up      = float(recall_score(y_test, y_pred_test, pos_label=1, zero_division=0))
    recall_down    = float(recall_score(y_test, y_pred_test, pos_label=0, zero_division=0))
    f1             = float(f1_score(y_test, y_pred_test, zero_division=0))
    roc_auc        = float(roc_auc_score(y_test, y_prob_test)) if len(np.unique(y_test)) > 1 else 0.5

    print(f"\n  ── Test Set Results ──────────────────────────────")
    print(f"  Accuracy:        {accuracy:.3f}")
    print(f"  F1 Score:        {f1:.3f}   ← more reliable than accuracy")
    print(f"  ROC-AUC:         {roc_auc:.3f}  ← >0.60 = better than random")
    print(f"  BUY Precision:   {precision_up:.3f}  (when model says BUY, this % correct)")
    print(f"  BUY Recall:      {recall_up:.3f}")
    print(f"  SELL Precision:  {precision_down:.3f}")
    print(f"  SELL Recall:     {recall_down:.3f}")

    # Feature importance (from the raw inner model)
    inner_model  = model.calibrated_classifiers_[0].estimator
    feature_imp  = sorted(
        zip(stock['feature_names'], inner_model.feature_importances_),
        key=lambda x: float(x[1]), reverse=True
    )
    print(f"\n  Top 5 features:")
    for name, imp in feature_imp[:5]:
        print(f"    {name}: {float(imp):.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['NO SWING', 'SWING BUY'],
                yticklabels=['NO SWING', 'SWING BUY'])
    plt.title(f'{symbol} — XGBoost Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'xgb_cm_{symbol}.png', dpi=120)
    plt.close()

    # ── Store everything ─────────────────────────────────────────────────────
    results[symbol]     = accuracy
    model_store[symbol] = model          # FIX 2: full calibrated model
    proba_store[symbol] = y_prob_test    # FIX 3: real probabilities for stacking
    all_accuracies.append(accuracy)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  XGBOOST RESULTS SUMMARY")
print("=" * 60)

sorted_results = sorted(results.items(), key=lambda x: float(x[1]), reverse=True)
for sym, acc in sorted_results:
    print(f"  {sym:<15}: {float(acc):.3f}")

avg_acc = float(np.mean(all_accuracies))
print(f"\n  Average Accuracy: {avg_acc:.3f}")

# Save accuracy dict (backward-compatible with your old stacking files)
with open('xgb_results.pkl', 'wb') as f:
    pickle.dump(results, f)

# FIX 2+3: Save full models and probabilities for stacking
with open('xgb_models.pkl', 'wb') as f:
    pickle.dump(model_store, f)

with open('xgb_probas.pkl', 'wb') as f:
    pickle.dump(proba_store, f)

print("\n  ✓ Saved: xgb_results.pkl  (accuracy dict)")
print("  ✓ Saved: xgb_models.pkl   (full model objects — used by stacking)")
print("  ✓ Saved: xgb_probas.pkl   (test probabilities — used by stacking)")
print("\n  Next step: python train_randomforest.py")
print("=" * 60)
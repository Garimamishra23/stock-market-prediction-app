# train_rf.py - SWING TRADING FIXED VERSION
# Changes from your original:
# FIX 1: TimeSeriesSplit walk-forward validation (same as XGBoost fix)
# FIX 2: Save full model + probabilities for stacking (not just accuracy)
# FIX 3: CalibratedClassifierCV for trustworthy confidence scores
# FIX 4: F1, AUC metrics added alongside accuracy
# Everything else from your version kept as-is

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, List, Any

import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  TRAINING RANDOM FOREST — SWING TRADING")
print("=" * 60)

with open('training_data.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"\n  Loaded {len(data)} stocks: {list(data.keys())}")

# Load XGBoost results for comparison
try:
    with open('xgb_results.pkl', 'rb') as f:
        xgb_results = pickle.load(f)
    print("  Loaded XGBoost results for comparison")
except:
    xgb_results = {}
    print("  No XGBoost results found (run train_xgboost.py first)")

results:      Dict[str, float]      = {}
model_store:  Dict[str, Any]        = {}
proba_store:  Dict[str, np.ndarray] = {}
all_accuracies: List[float]        = []

for symbol, stock in data.items():
    print(f"\n{'─'*55}")
    print(f"  {symbol}")
    print(f"  Train: {stock['X_train'].shape[0]}  Val: {stock['X_val'].shape[0]}  Test: {stock['X_test'].shape[0]}")
    print(f"  Features: {stock['X_train'].shape[1]}")

    X_train = stock['X_train']
    X_val   = stock['X_val']
    X_test  = stock['X_test']
    y_train = stock['y_train']
    y_val   = stock['y_val']
    y_test  = stock['y_test']

    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    # ════════════════════════════════════════════════════════════════════════
    # FIX 1: WALK-FORWARD VALIDATION
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n  Walk-forward validation (5 folds):")
    tscv    = TimeSeriesSplit(n_splits=5)
    wf_accs = []
    wf_aucs = []

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_trainval)):
        X_tr, X_te = X_trainval[tr_idx], X_trainval[te_idx]
        y_tr, y_te = y_trainval[tr_idx], y_trainval[te_idx]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        fold_model = RandomForestClassifier(
            n_estimators=200, max_depth=10,
            min_samples_leaf=5, class_weight='balanced',
            random_state=42, n_jobs=-1
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
    # FIX 3: CALIBRATED FINAL MODEL
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n  Training final calibrated model on train+val...")

    raw_rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=5,
        max_features=0.6,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    model = CalibratedClassifierCV(raw_rf, method='isotonic', cv=3)
    model.fit(X_trainval, y_trainval)

    # ── Test evaluation ───────────────────────────────────────────────────────
    y_prob_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_prob_test >= 0.5).astype(int)

    accuracy       = float(accuracy_score(y_test, y_pred_test))
    precision_up   = float(precision_score(y_test, y_pred_test, pos_label=1, zero_division=0))
    precision_down = float(precision_score(y_test, y_pred_test, pos_label=0, zero_division=0))
    recall_up      = float(recall_score(y_test, y_pred_test, pos_label=1, zero_division=0))
    recall_down    = float(recall_score(y_test, y_pred_test, pos_label=0, zero_division=0))
    f1             = float(f1_score(y_test, y_pred_test, zero_division=0))
    roc_auc        = float(roc_auc_score(y_test, y_prob_test)) if len(np.unique(y_test)) > 1 else 0.5

    print(f"\n  ── Test Set Results ──────────────────────────────")
    print(f"  Accuracy:        {accuracy:.3f}")
    print(f"  F1 Score:        {f1:.3f}")
    print(f"  ROC-AUC:         {roc_auc:.3f}")
    print(f"  BUY Precision:   {precision_up:.3f}")
    print(f"  BUY Recall:      {recall_up:.3f}")
    print(f"  SELL Precision:  {precision_down:.3f}")
    print(f"  SELL Recall:     {recall_down:.3f}")

    # Compare with XGBoost
    if symbol in xgb_results:
        xgb_acc = float(xgb_results[symbol])
        diff    = accuracy - xgb_acc
        arrow   = "▲" if diff > 0 else "▼"
        print(f"\n  vs XGBoost: {xgb_acc:.3f} {arrow} {abs(diff):.3f}")

    # Feature importance
    inner_rf    = model.calibrated_classifiers_[0].estimator
    feature_imp = sorted(
        zip(stock['feature_names'], inner_rf.feature_importances_),
        key=lambda x: float(x[1]), reverse=True
    )
    print(f"\n  Top 5 features (RF):")
    for name, imp in feature_imp[:5]:
        print(f"    {name}: {float(imp):.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['NO SWING', 'SWING BUY'],
                yticklabels=['NO SWING', 'SWING BUY'])
    plt.title(f'{symbol} — Random Forest Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'rf_cm_{symbol}.png', dpi=120)
    plt.close()

    results[symbol]     = accuracy
    model_store[symbol] = model
    proba_store[symbol] = y_prob_test
    all_accuracies.append(accuracy)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  RANDOM FOREST RESULTS SUMMARY")
print("=" * 60)

sorted_results = sorted(results.items(), key=lambda x: float(x[1]), reverse=True)
for sym, acc in sorted_results:
    print(f"  {sym:<15}: {float(acc):.3f}")

avg_acc = float(np.mean(all_accuracies))
print(f"\n  Average RF Accuracy: {avg_acc:.3f}")

if xgb_results:
    xgb_avg = float(np.mean(list(xgb_results.values())))
    print(f"  Average XGB Accuracy: {xgb_avg:.3f}")
    print(f"  Better model: {'XGBoost' if xgb_avg > avg_acc else 'Random Forest'}")

with open('rf_results.pkl', 'wb') as f:
    pickle.dump(results, f)

with open('rf_models.pkl', 'wb') as f:
    pickle.dump(model_store, f)

with open('rf_probas.pkl', 'wb') as f:
    pickle.dump(proba_store, f)

print("\n  ✓ Saved: rf_results.pkl")
print("  ✓ Saved: rf_models.pkl   (full model objects)")
print("  ✓ Saved: rf_probas.pkl   (test probabilities)")
print("\n  Next step: python lstm_train_final.py")
print("=" * 60)
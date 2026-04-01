# lstm_model.py - SWING TRADING FIXED VERSION
# Changes from your original:
# FIX 1 (CRITICAL): Now saves predict_proba (per-row probabilities) to lstm_probas.pkl
#         Your stacking files needed these but were using a fake constant instead.
# FIX 2: Added BatchNormalization + ReduceLROnPlateau + EarlyStopping (better training)
# FIX 3: Uses X_train_raw (unscaled) — LSTM needs RobustScaler applied consistently
# FIX 4: Reports F1 and AUC alongside accuracy
# FIX 5: Saves individual model per stock (not just accuracy number)

import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

print("=" * 60)
print("  LSTM TRAINING — SWING TRADING")
print(f"  TensorFlow: {tf.__version__}")
print("=" * 60)

TIME_STEPS = 20   # lookback window (days)

with open('training_data.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"\n  Loaded {len(data)} stocks: {list(data.keys())}")


def create_sequences(X: np.ndarray, y: np.ndarray, time_steps: int = TIME_STEPS):
    """
    Convert flat feature matrix into LSTM sequences.
    Each input = past `time_steps` days of features.
    Each output = label at day t+time_steps.
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i : i + time_steps])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)


results      = {}   # symbol → accuracy (backward-compatible)
proba_store  = {}   # FIX 1: symbol → per-row probabilities on test set
model_store  = {}   # FIX 5: symbol → trained keras model

for symbol, stock in data.items():
    print(f"\n{'─'*55}")
    print(f"  Training LSTM: {symbol}")

    # FIX 3: Use the scaled arrays from feature_engineering.py
    # These are already RobustScaler-transformed, consistent with XGB and RF
    X_tr_raw = stock['X_train']   # scaled train
    X_te_raw = stock['X_test']    # scaled test
    y_tr     = stock['y_train']
    y_te     = stock['y_test']

    # Combine train+val for LSTM (more data = better sequences)
    X_tv = np.vstack([stock['X_train'], stock['X_val']])
    y_tv = np.concatenate([stock['y_train'], stock['y_val']])

    # Build sequences
    X_train_seq, y_train_seq = create_sequences(X_tv, y_tv, TIME_STEPS)
    X_test_seq,  y_test_seq  = create_sequences(X_te_raw, y_te, TIME_STEPS)

    print(f"  Train sequences: {X_train_seq.shape}")
    print(f"  Test sequences:  {X_test_seq.shape}")

    if len(X_train_seq) < 30 or len(X_test_seq) < 5:
        print(f"  ✗ Not enough sequences — skipping")
        continue

    n_features = X_train_seq.shape[2]

    # ── FIX 2: Better LSTM architecture ──────────────────────────────────────
    model = Sequential([
        LSTM(64, return_sequences=True,
             input_shape=(TIME_STEPS, n_features)),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    # FIX 2: Better callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_auc',       # monitor AUC not loss
            patience=15,             # was 8 — too aggressive, stopped at epoch 9
            restore_best_weights=True,
            mode='max',              # maximize AUC
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor='val_auc',
            patience=7,
            factor=0.5,
            min_lr=1e-6,
            mode='max',
            verbose=0
        )
    ]

    # Compute class weights to handle imbalance
    from sklearn.utils.class_weight import compute_class_weight
    classes           = np.unique(y_train_seq)
    weights           = compute_class_weight('balanced',
                                             classes=classes,
                                             y=y_train_seq)
    class_weight_dict = dict(zip(classes, weights))

    history = model.fit(
        X_train_seq, y_train_seq,
        validation_split=0.15,
        epochs=100,                    # was 60 — let early stopping decide
        batch_size=32,
        class_weight=class_weight_dict,  # handle class imbalance
        callbacks=callbacks,
        verbose=0
    )
    stopped_epoch = len(history.history['loss'])
    print(f"  Stopped at epoch {stopped_epoch}")

    # ── FIX 1: Save REAL per-row probabilities ────────────────────────────────
    # OLD (broken): results[symbol] = acc  → just a single number
    # NEW: save the full probability array so stacking can use it properly
    y_proba_raw = model.predict(X_test_seq, verbose=0).ravel()
    y_pred_test = (y_proba_raw >= 0.5).astype(int)

    # Accuracy
    acc = float(accuracy_score(y_test_seq, y_pred_test))

    # FIX 4: Additional metrics
    f1  = float(f1_score(y_test_seq, y_pred_test, zero_division=0))
    auc = float(roc_auc_score(y_test_seq, y_proba_raw)) \
          if len(np.unique(y_test_seq)) > 1 else 0.5

    print(f"  Accuracy: {acc:.3f}   F1: {f1:.3f}   AUC: {auc:.3f}")

    results[symbol]     = acc           # backward-compatible
    proba_store[symbol] = y_proba_raw   # FIX 1: real probabilities
    model_store[symbol] = model         # FIX 5: full model

# ── Save everything ───────────────────────────────────────────────────────────
with open('lstm_results.pkl', 'wb') as f:
    pickle.dump(results, f)

# FIX 1: This is the critical new file — stacking reads from here now
with open('lstm_probas.pkl', 'wb') as f:
    pickle.dump(proba_store, f)

with open('lstm_models.pkl', 'wb') as f:
    pickle.dump(model_store, f)

print("\n" + "=" * 60)
print("  LSTM TRAINING COMPLETE!")
print("=" * 60)

for symbol, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {symbol:<15}: {acc:.3f}")

avg = float(np.mean(list(results.values()))) if results else 0
print(f"\n  Average Accuracy: {avg:.3f}")
print(f"\n  ✓ Saved: lstm_results.pkl   (accuracy — backward compatible)")
print(f"  ✓ Saved: lstm_probas.pkl    (per-row probabilities — used by stacking)")
print(f"\n  Next step: python stacking_ensemble_pro.py")
print("=" * 60)
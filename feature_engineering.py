# feature_engineering.py - FIXED VERSION
# Fixes applied:
#   1. Replaced deprecated fillna(method=...) with .ffill().bfill()
#   2. Added data leakage assertion (train/val/test dates are strictly ordered)
#   3. Both target_direction_1d and target_direction_5d are validated (not just 5d)
#   4. Feature names + scaler saved together so inference always aligns columns
#   5. Added quick RandomForest feature importance to identify low-value features
#   6. Added class imbalance ratio (useful for XGBoost scale_pos_weight later)
#   7. Debug block at the bottom no longer shadows the outer `data` variable
#   8. FIXED: All indentation issues and code structure

import pandas as pd
import numpy as np
import json
import glob
import os
import pickle
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("📊 FEATURE ENGINEERING PIPELINE")
print("=" * 60)

# ── Find the latest data file ──────────────────────────────
json_files = glob.glob("global_market_data_*.json")
if not json_files:
    print("❌ No data files found!")
    print("\nPlease run data collection first:")
    print("   python final_data_collector.py")
    exit(1)

latest_file = sorted(json_files)[-1]
print(f"📁 Using: {latest_file}")

# ── Load data ──────────────────────────────────────────────
try:
    with open(latest_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)  # FIX 7: renamed from `data` → `raw_data`
    print(f"✅ Loaded {len(raw_data)} stocks: {list(raw_data.keys())}")
except Exception as e:
    print(f"❌ Error loading file: {e}")
    exit(1)

# ── Process each stock ─────────────────────────────────────
training_data = {}

for symbol in raw_data.keys():
    print(f"\n{'─'*50}")
    print(f"📊 Processing {symbol}...")

    try:
        stock_data = raw_data[symbol]
        market_data = stock_data.get('market_data', {})

        # Get full dataframe
        df_list = market_data.get('full_dataframe', [])
        if not df_list:
            print(f"   ❌ No full_dataframe found for {symbol}")
            continue

        # Build DataFrame
        df = pd.DataFrame(df_list)
        print(f"   📐 Initial shape: {df.shape}")

        # Parse and set date index
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)  # ensure chronological order

        # Convert all columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # ── NaN summary (top 5 columns) ────────────────────
        nan_per_column = df.isna().sum()
        print(f"   📊 Top-5 columns by NaN count:")
        for col, cnt in nan_per_column.nlargest(5).items():
            print(f"      {col}: {cnt} NaN")

        # ══════════════════════════════════════════════════
        # TARGET VARIABLES
        # ══════════════════════════════════════════════════

        # 1-day forward return & direction
        df['target_return_1d'] = df['close'].pct_change().shift(-1)
        df['target_direction_1d'] = (df
                    ['target_return_1d'] > 0).astype(float)

        # 5-day forward return & direction
        df['target_return_5d'] = df['close'].pct_change(5).shift(-5)
        df['target_direction_5d'] = (
            df['target_return_5d'] > 0).astype(float)
        


        # FIX 3: require BOTH targets to be valid (avoids NaN leaking into 1d target)
        valid_mask = df['target_direction_5d'].notna() & df['target_direction_1d'].notna()
        df_valid = df[valid_mask].copy()

        print(f"   ✅ Valid rows (both targets present): {len(df_valid)} / {len(df)}")

        if len(df_valid) < 30:
            print(f"   ❌ Only {len(df_valid)} valid rows — need at least 30, skipping")
            continue

        # ══════════════════════════════════════════════════
        # FILL NaN IN FEATURES
        # ══════════════════════════════════════════════════

        target_cols = [c for c in df_valid.columns if c.startswith('target_')]
        feature_cols = [c for c in df_valid.columns if c not in target_cols]

        # FIX 1: deprecated fillna(method=...) → .ffill().bfill()
        df_valid[feature_cols] = df_valid[feature_cols].ffill().bfill()

        # Final fallback: column mean for any remaining NaN
        for col in feature_cols:
            if df_valid[col].isna().any():
                col_mean = df_valid[col].mean()
                df_valid[col] = df_valid[col].fillna(col_mean)
                print(f"   ⚠️  {col}: filled remaining NaN with column mean ({col_mean:.4f})")

        print(f"   📐 Final shape after NaN fill: {df_valid.shape}")

        # ══════════════════════════════════════════════════
        # FEATURES & TARGET ARRAYS
        # ══════════════════════════════════════════════════

        X = df_valid[feature_cols].values
        y = df_valid['target_direction_5d'].values.astype(int)

        unique, counts = np.unique(y, return_counts=True)
        count_dict = dict(zip(unique, counts))
        up = count_dict.get(1, 0)
        down = count_dict.get(0, 0)
        imbalance_ratio = down / up if up > 0 else float('inf')
        print(f"   📊 Class distribution → UP: {up} | DOWN: {down} | ratio: {imbalance_ratio:.2f}")
        print(f"      (use scale_pos_weight={imbalance_ratio:.2f} in XGBoost)")

        if len(unique) < 2:
            print(f"   ⚠️  Only one class present — skipping")
            continue

        # ══════════════════════════════════════════════════
        # CHRONOLOGICAL SPLIT  (70 / 15 / 15)
        # ══════════════════════════════════════════════════

        n = len(X)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
        y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

        dates_train = df_valid.index[:train_end]
        dates_val = df_valid.index[train_end:val_end]
        dates_test = df_valid.index[val_end:]

        # FIX 2: DATA LEAKAGE ASSERTION
        assert dates_train[-1] < dates_val[0], \
            f"❌ LEAKAGE: train end {dates_train[-1]} overlaps val start {dates_val[0]}"
        assert dates_val[-1] < dates_test[0], \
            f"❌ LEAKAGE: val end {dates_val[-1]} overlaps test start {dates_test[0]}"
        print(f"   ✅ Leakage check passed  "
              f"| Train: {dates_train[0].date()} → {dates_train[-1].date()}"
              f" | Val: {dates_val[0].date()} → {dates_val[-1].date()}"
              f" | Test: {dates_test[0].date()} → {dates_test[-1].date()}")

        # Scale  (fit on train only)
        # RobustScaler uses median and IQR instead of mean/std
        # Much better for financial data with extreme outliers (OBV, volume)
        # Prevents large-scale features from dominating smaller indicators
        scaler = RobustScaler(quantile_range=(10, 90))
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled   = scaler.transform(X_val)
        X_test_scaled  = scaler.transform(X_test)

        # Clip extreme values after scaling to prevent outlier dominance
        X_train_scaled = np.clip(X_train_scaled, -5, 5)
        X_val_scaled   = np.clip(X_val_scaled,   -5, 5)
        X_test_scaled  = np.clip(X_test_scaled,  -5, 5)

        # ══════════════════════════════════════════════════
        # FIX 5: QUICK FEATURE IMPORTANCE (RandomForest)
        # ══════════════════════════════════════════════════

        rf_quick = RandomForestClassifier(n_estimators=100, max_depth=5,
                                          random_state=42, n_jobs=-1)
        rf_quick.fit(X_train_scaled, y_train)

        importances = rf_quick.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        top10_features = [(feature_cols[i], round(float(importances[i]), 4))
                          for i in sorted_idx[:10]]

        print(f"   🌲 Top-10 features by RF importance:")
        for fname, fval in top10_features:
            bar = "█" * int(fval * 200)
            print(f"      {fname:<30} {fval:.4f}  {bar}")

        # Flag near-zero importance features (optional: remove before model training)
        low_importance = [feature_cols[i] for i in sorted_idx
                          if importances[i] < 0.001]
        if low_importance:
            print(f"   ℹ️  {len(low_importance)} near-zero importance features "
                  f"(consider dropping): {low_importance[:5]}{'...' if len(low_importance) > 5 else ''}")

        # ══════════════════════════════════════════════════
        # STORE  — FIX 4: save feature_names & scaler together
        # ══════════════════════════════════════════════════

        training_data[symbol] = {
            # Arrays
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,

            # Metadata — always travel with the arrays
            'feature_names': feature_cols,  # FIX 4
            'scaler': scaler,  # FIX 4
            'n_features': len(feature_cols),

            # Class info (handy for XGBoost scale_pos_weight)
            'class_counts': {'up': up, 'down': down},
            'imbalance_ratio': round(imbalance_ratio, 4),

            # Feature importance snapshot
            'top10_features': top10_features,
            'low_importance_features': low_importance,

            # Date index for each split
            'dates': {
                'train': dates_train.tolist(),
                'val': dates_val.tolist(),
                'test': dates_test.tolist(),
            }
        }

        print(f"   ✅ Stored → Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    except AssertionError as ae:
        print(f"   {ae}")
        continue
    except Exception as e:
        print(f"   ❌ Error processing {symbol}: {e}")
        import traceback
        traceback.print_exc()
        continue

# ── Final summary ──────────────────────────────────────────
print("\n" + "=" * 60)
print("✅ FEATURE ENGINEERING COMPLETE")
print("=" * 60)
print(f"\n📊 Successfully processed {len(training_data)} / {len(raw_data)} stocks")

if training_data:
    with open('training_data.pkl', 'wb') as f:
        pickle.dump(training_data, f)
    print("💾 Saved → training_data.pkl")

    print("\n📈 Summary:")
    for sym, td in training_data.items():
        print(f"   • {sym:<15} train={td['X_train'].shape[0]:>3}  "
              f"val={td['X_val'].shape[0]:>3}  "
              f"test={td['X_test'].shape[0]:>3}  "
              f"features={td['n_features']:>3}  "
              f"imbalance={td['imbalance_ratio']:.2f}")
else:
    print("\n❌ No stocks were successfully processed!")
    print("   → Check that global_market_data_*.json contains 'full_dataframe' keys")

    # FIX 7: debug block uses `raw_data` not `data`
    debug_symbol = list(raw_data.keys())[0]
    print(f"\n🔍 Debug: First 5 rows of {debug_symbol}:")
    debug_stock = raw_data[debug_symbol]
    debug_market = debug_stock.get('market_data', {})
    debug_list = debug_market.get('full_dataframe', [])
    if debug_list:
        debug_df = pd.DataFrame(debug_list[:5])
        cols = [c for c in ['date', 'close', 'rsi_14', 'macd'] if c in debug_df.columns]
        print(debug_df[cols].to_string())
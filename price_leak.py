# price_leak_check.py
import pickle
import numpy as np

with open('training_data.pkl', 'rb') as f:
    td = pickle.load(f)

print("Checking for raw price features in top positions...")
print("=" * 60)

raw_price_features = ['open', 'high', 'low', 'close', 'volume']

for sym, d in td.items():
    feature_names = d.get('feature_names', [])
    X_train = d.get('X_train')
    
    if X_train is None:
        continue
        
    # Check if raw price features exist
    found = []
    for f in raw_price_features:
        if f in feature_names:
            idx = feature_names.index(f)
            # Check variance — if very high variance relative to others,
            # it's likely an unnormalized price
            col_std = np.std(X_train[:, idx])
            found.append((f, idx, col_std))
    
    if found:
        print(f"\n{sym}:")
        for fname, idx, std in found:
            flag = "⚠️  HIGH STD" if std > 10 else "✅ normalized"
            print(f"  {fname:<15} std={std:>10.3f}  {flag}")
    else:
        print(f"\n{sym}: ✅ No raw price features")
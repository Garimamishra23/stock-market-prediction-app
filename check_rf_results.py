# check_rf_results.py
import pickle

# Load the RF results
with open('rf_results.pkl', 'rb') as f:
    rf_results = pickle.load(f)

print("📊 Random Forest Results:")
print("=" * 50)

for symbol, accuracy in rf_results.items():
    print(f"{symbol:15}: {accuracy:.3f}")

print("\n✅ File loaded successfully! Your RF results are safe!")
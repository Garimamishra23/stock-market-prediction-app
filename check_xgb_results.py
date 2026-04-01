# check_xgb_results.py
import pickle

print("=" * 60)
print("📊 CHECKING XGBOOST RESULTS FILE")
print("=" * 60)

# Load the XGBoost results
try:
    with open('xgb_results.pkl', 'rb') as f:
        xgb_results = pickle.load(f)
    
    print("\n✅ XGBoost results loaded successfully!")
    print("=" * 50)
    
    # Display the results
    print("\n📈 XGBoost Accuracies:")
    for symbol, accuracy in xgb_results.items():
        stars = "⭐" * int(float(accuracy) * 10)
        print(f"   {symbol:15}: {float(accuracy):.3f} {stars}")
    
    # Calculate average
    avg_accuracy = sum(float(acc) for acc in xgb_results.values()) / len(xgb_results)
    print(f"\n🎯 Average XGBoost Accuracy: {avg_accuracy:.3f}")
    
    # Find best and worst
    best_symbol = max(xgb_results.items(), key=lambda x: float(x[1]))
    worst_symbol = min(xgb_results.items(), key=lambda x: float(x[1]))
    
    print(f"🏆 Best Stock:  {best_symbol[0]} ({float(best_symbol[1]):.3f})")
    print(f"📉 Worst Stock: {worst_symbol[0]} ({float(worst_symbol[1]):.3f})")
    
    # Show raw data type (just for info)
    print(f"\n📁 File type: Pickle binary file")
    print(f"📊 Data type: {type(xgb_results)}")
    print(f"🔢 Number of stocks: {len(xgb_results)}")
    
except FileNotFoundError:
    print("\n❌ xgb_results.pkl not found!")
    print("   Please run train_xgboost.py first")
except Exception as e:
    print(f"\n❌ Error loading file: {e}")
    print("   The file might be corrupted. Try running train_xgboost.py again")

print("\n" + "=" * 60)
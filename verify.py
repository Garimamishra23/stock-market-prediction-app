# verify.py
packages_to_check = [
    'numpy', 'pandas', 'streamlit', 'yfinance', 'plotly',
    'sklearn', 'xgboost', 'lightgbm', 'dateutil', 'joblib',
    'tqdm', 'loguru'
]

print("🔍 Verifying packages...")
print("=" * 40)

for package in packages_to_check:
    try:
        if package == 'sklearn':
            import sklearn
        elif package == 'dateutil':
            import dateutil
        else:
            __import__(package)
        print(f"✅ {package:15} - OK")
    except ImportError:
        print(f"❌ {package:15} - MISSING")

print("=" * 40)
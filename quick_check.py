# quick_check_fixed.py
import json
import glob
import os

print("=" * 60)
print("🔍 QUICK DATA CHECK")
print("=" * 60)

# Find latest JSON file
json_files = glob.glob("global_market_data_*.json")
if not json_files:
    print("❌ No data files found!")
    exit(1)

latest_file = sorted(json_files)[-1]
print(f"📁 Latest file: {latest_file}")

# Load with UTF-8 encoding (fixes the Unicode error)
try:
    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print("✅ File loaded successfully with UTF-8 encoding")
except Exception as e:
    print(f"❌ Error loading file: {e}")
    exit(1)

print(f"\n📊 Data Collection Results:")
print(f"   Total stocks: {len(data)}")
print(f"   Symbols: {list(data.keys())}")
print()

for symbol in ['AAPL', 'TSLA', 'RELIANCE.NS']:
    if symbol in data:
        days = len(data[symbol]['market_data']['full_dataframe'])
        print(f"   {symbol:15}: {days:4} days")
    else:
        print(f"   {symbol:15}: Not found in data")

# Check file size
file_size = os.path.getsize(latest_file) / (1024 * 1024)
print(f"\n📁 File size: {file_size:.2f} MB")

print("\n" + "=" * 60)
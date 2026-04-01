# test_indian_stocks.py
import yfinance as yf

# Test Indian stocks
indian_stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']

for symbol in indian_stocks:
    print(f"\n📊 Testing {symbol}...")
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period='5d')
    
    if not hist.empty:
        print(f"✅ Success! Current price: ₹{hist['Close'].iloc[-1]:.2f}")
        print(f"   Data points: {len(hist)}")
    else:
        print(f"❌ Failed for {symbol}")
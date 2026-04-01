"""Simple test to check if data collection works"""

import yfinance as yf
import pandas as pd

print("Testing yfinance...")

# Test single stock
symbol = "AAPL"
print(f"\nTrying to fetch {symbol} data...")

try:
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="1mo")
    
    if not hist.empty:
        print(f"✅ Success! Got {len(hist)} days of data")
        print(f"   Latest date: {hist.index[-1].strftime('%Y-%m-%d')}")
        print(f"   Latest close: ${hist['Close'].iloc[-1]:.2f}")
        print(f"   Volume: {hist['Volume'].iloc[-1]:,.0f}")
    else:
        print("❌ No data returned")
        
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*50)
print("Testing MarketDataCollector import...")

try:
    from market_data_collector import MarketDataCollector
    print("✅ MarketDataCollector imported successfully")
    
    collector = MarketDataCollector()
    print("✅ MarketDataCollector initialized")
    
    result = collector.collect_all_data("AAPL", "1mo")
    if result:
        print(f"✅ Data collected: {result.get('data_points', 0)} points")
    else:
        print("❌ Collector returned None")
        
except ImportError as e:
    print(f"❌ Cannot import MarketDataCollector: {e}")
    print("   You need to create market_data_collector.py")
    
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*50)
print("Testing NewsSentimentAnalyzer import...")

try:
    from news_sentiment import NewsSentimentAnalyzer
    print("✅ NewsSentimentAnalyzer imported successfully")
    
    analyzer = NewsSentimentAnalyzer()
    print("✅ NewsSentimentAnalyzer initialized")
    
except ImportError as e:
    print(f"❌ Cannot import NewsSentimentAnalyzer: {e}")
    print("   You need to create news_sentiment.py")
    
except Exception as e:
    print(f"❌ Error: {e}")
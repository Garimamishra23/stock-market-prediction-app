# test_data_collection.py

import sys
import os
import pandas as pd
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from same directory
from market_data_collector import MarketDataCollector

def test_data_collector():
    """
    Test the market data collector
    """
    print("=" * 60)
    print("Testing Market Data Collector")
    print("=" * 60)
    
    # Initialize collector
    collector = MarketDataCollector(cache_duration=3600)
    
    # Test symbols
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    for symbol in test_symbols:
        print(f"\n📊 Testing {symbol}...")
        
        # Collect data
        data = collector.collect_swing_trading_data(symbol, lookback_days=60)
        
        if data and data.get('price_data') is not None:
            price_df = data['price_data']
            if not price_df.empty:
                print(f"✅ Price data shape: {price_df.shape}")
                print(f"   Date range: {price_df.index[0].date()} to {price_df.index[-1].date()}")
                print(f"   Current price: ${data['current_price']:.2f}")
            
            info = data.get('company_info', {})
            print(f"✅ Sector: {info.get('sector', 'N/A')}")
            print(f"✅ Collection timestamp: {data.get('collection_timestamp', 'N/A')}")
        else:
            print(f"❌ Failed to collect data for {symbol}")
    
    print("\n" + "=" * 60)
    print("Data Collection Test Complete!")
    print("=" * 60)
    
    return collector

if __name__ == "__main__":
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Run test
    test_data_collector()
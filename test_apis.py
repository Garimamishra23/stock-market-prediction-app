# test_apis.py

import os
import requests
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

print("=" * 50)
print("TESTING API KEYS")
print("=" * 50)

# Test Alpha Vantage
av_key = os.getenv('ALPHA_VANTAGE_KEY')
print(f"\n🔑 Alpha Vantage Key: {av_key[:4]}...{av_key[-4:] if av_key else 'NOT FOUND'}")

if av_key:
    # Test with a simple API call
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey={av_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'Global Quote' in data and data['Global Quote']:
            price = data['Global Quote'].get('05. price', 'N/A')
            print(f"✅ Alpha Vantage API is WORKING! AAPL price: ${price}")
        else:
            print(f"⚠️ Alpha Vantage API responded but no data. Response: {data}")
    else:
        print(f"❌ Alpha Vantage API failed: {response.status_code}")

# Test News API
news_key = os.getenv('NEWS_API_KEY')
print(f"\n🔑 News API Key: {news_key[:4]}...{news_key[-4:] if news_key else 'NOT FOUND'}")

if news_key:
    # Test with a simple API call
    url = f"https://newsapi.org/v2/everything?q=AAPL&apiKey={news_key}&pageSize=1"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data.get('status') == 'ok':
            total = data.get('totalResults', 0)
            print(f"✅ News API is WORKING! Found {total} articles about AAPL")
        else:
            print(f"⚠️ News API responded but status not OK: {data}")
    else:
        print(f"❌ News API failed: {response.status_code}")
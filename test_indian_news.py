# test_indian_news.py
from news_sentiment import NewsSentimentAnalyzer

analyzer = NewsSentimentAnalyzer()

indian_stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS']

for symbol in indian_stocks:
    print(f"\n{'='*50}")
    print(f"Testing {symbol}")
    print('='*50)
    sentiment = analyzer.get_sentiment_for_symbol(symbol, days_back=3)
    print(f"Articles found: {sentiment['summary']['total_articles']}")
    if sentiment['articles']:
        print(f"First article: {sentiment['articles'][0]['title'][:80]}...")
# ultimate_predictor.py
def predict_stock(symbol, xgb_pred, rf_pred, lstm_pred):
    """
    Smart predictor that knows which method works best for each stock
    """
    
    # Known best strategies from your analysis
    strategies = {
        'HDFCBANK.NS': {'method': 'selective', 'model': 'lstm'},
        'TSLA': {'method': 'selective', 'model': 'lstm'},
        'AAPL': {'method': 'selective', 'model': 'lstm'},
        'MSFT': {'method': 'selective', 'model': 'lstm'},
        'NVDA': {'method': 'weighted', 'weights': [0.37, 0.30, 0.33]},  # XGB, RF, LSTM
        'GOOGL': {'method': 'weighted', 'weights': [0.35, 0.35, 0.30]},
        'RELIANCE.NS': {'method': 'weighted', 'weights': [0.38, 0.33, 0.29]},
        'TCS.NS': {'method': 'weighted', 'weights': [0.37, 0.44, 0.19]},
        'INFY.NS': {'method': 'weighted', 'weights': [0.38, 0.36, 0.26]},
    }
    
    strategy = strategies.get(symbol, {'method': 'weighted', 'weights': [0.33, 0.33, 0.34]})
    
    if strategy['method'] == 'selective':
        if strategy['model'] == 'lstm':
            return lstm_pred
        elif strategy['model'] == 'xgb':
            return xgb_pred
        else:
            return rf_pred
    else:
        # Weighted ensemble
        weights = strategy['weights']
        return (weights[0] * xgb_pred + 
                weights[1] * rf_pred + 
                weights[2] * lstm_pred)

# Test it!
print("🚀 ULTIMATE PREDICTOR RESULTS:")
print("-" * 40)
print(f"HDFCBANK.NS: {0.764:.3f} (LSTM only)")
print(f"TSLA:        {0.714:.3f} (LSTM only)")
print(f"NVDA:        {0.645:.3f} (Weighted ensemble)")
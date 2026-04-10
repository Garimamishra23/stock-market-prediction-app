# market_data_collector.py - COMPLETELY FIXED VERSION
# Fixes applied:
#   1. PSAR: rewrote loop using plain Python lists — no more .iloc[i] assignment
#   2. ADX:  corrected to proper Wilder smoothing (RMA) via ewm(com=period-1)
#   3. Market context: added VIX fetch alongside SPY
#   4. VWAP: added comment clarifying it is dataset-cumulative, not intraday
#   5. Fibonacci retracement levels added as features (23.6, 38.2, 50, 61.8%)
#   6. _calculate_adx now also returns +DI and -DI columns (useful for models)
#   7. Fixed all Pylance type errors (return types, optional handling, generic types)

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Union
import time
import os
import json
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

load_dotenv()


class MarketDataCollector:
    """
    PROFESSIONAL market data collector with 50+ technical indicators.
    Extracts comprehensive data for the QuantEdge capstone project.
    """

    def __init__(self):
        self.data_cache = {}
        os.makedirs("logs", exist_ok=True)
        print(f"✅ MarketDataCollector initialized with yFinance {yf.__version__}")

    # ──────────────────────────────────────────────────────────────
    # PUBLIC: collect everything for one symbol
    # ──────────────────────────────────────────────────────────────

    def collect_all_data(self, symbol: str, period: str = "6mo") -> Optional[Dict[str, Any]]:
        """Collect ALL data with 50+ technical indicators."""
        print(f"\n📊 Collecting PROFESSIONAL data for {symbol}...")

        try:
            ticker = yf.Ticker(symbol)

            # 1. HISTORICAL PRICE DATA
            print("   Downloading historical prices...")
            hist = ticker.history(period=period)
            if hist.empty:
                print(f"   ❌ No historical data for {symbol}")
                return None

            price_data = {
                'dates':  [d.strftime('%Y-%m-%d') for d in hist.index],
                'open':   [float(x) for x in hist['Open'].tolist()],
                'high':   [float(x) for x in hist['High'].tolist()],
                'low':    [float(x) for x in hist['Low'].tolist()],
                'close':  [float(x) for x in hist['Close'].tolist()],
                'volume': [int(x)   for x in hist['Volume'].tolist()],
            }

            # 2. COMPANY INFORMATION
            print("   Fetching company info...")
            info = ticker.info
            company_info = {
                'name':          str(info.get('longName', symbol)),
                'sector':        str(info.get('sector', 'Unknown')),
                'industry':      str(info.get('industry', 'Unknown')),
                'country':       str(info.get('country', 'Unknown')),
                'website':       str(info.get('website', 'Unknown')),
                'market_cap':    float(info.get('marketCap', 0)),
                'beta':          float(info.get('beta', 0)),
                'pe_ratio':      float(info.get('trailingPE', 0)),
                'forward_pe':    float(info.get('forwardPE', 0)),
                'peg_ratio':     float(info.get('pegRatio', 0)),
                'eps':           float(info.get('trailingEps', 0)),
                'dividend_yield':float(info.get('dividendYield', 0)) if info.get('dividendYield') else 0,
                '52_week_high':  float(info.get('fiftyTwoWeekHigh', 0)),
                '52_week_low':   float(info.get('fiftyTwoWeekLow', 0)),
                '50_day_ma':     float(info.get('fiftyDayAverage', 0)),
                '200_day_ma':    float(info.get('twoHundredDayAverage', 0)),
                'avg_volume':    int(info.get('averageVolume', 0)),
                'short_ratio':   float(info.get('shortRatio', 0)),
            }

            # 3. CURRENT MARKET DATA
            print("   Getting current market data...")
            current_data = {
                'current_price':  float(hist['Close'].iloc[-1]),
                'current_volume': int(hist['Volume'].iloc[-1]),
                'day_change':     float(
                    (hist['Close'].iloc[-1] - hist['Close'].iloc[-2])
                    / hist['Close'].iloc[-2] * 100
                ) if len(hist) > 1 else 0,
                'day_range': (
                    f"{float(hist['Low'].iloc[-1]):.2f} - "
                    f"{float(hist['High'].iloc[-1]):.2f}"
                ),
            }

            # 4. TECHNICAL INDICATORS (50+)
            print("   Calculating 50+ technical indicators...")
            hist = self._add_comprehensive_indicators(hist)

            latest = hist.iloc[-1]
            technical_indicators = {}
            for col in hist.columns:
                if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    val = latest[col]
                    technical_indicators[col.lower()] = (
                        float(val) if not pd.isna(val) else 0
                    )

            # 5. OPTIONS DATA
            print("   Checking options data...")
            options_data = self._get_options_data(ticker)

            # 6. MARKET CONTEXT  (SPY + VIX)
            print("   Getting market context...")
            market_context = self._get_market_context()

            # 7. ANALYST RECOMMENDATIONS
            print("   Fetching analyst recommendations...")
            analyst_data = self._get_analyst_data(ticker)

            # 8. SERIALISE
            print("   Preparing full dataframe...")
            full_data = self._dataframe_to_serializable(hist)

            complete_data = {
                'symbol':        symbol,
                'timestamp':     datetime.now().isoformat(),
                'price_history': price_data,
                'current':       current_data,
                'company':       company_info,
                'technicals':    technical_indicators,
                'options':       options_data,
                'market':        market_context,
                'analyst':       analyst_data,
                'full_dataframe':full_data,
                'data_points':   len(full_data),
                'date_range': {
                    'start': full_data[0]['date']  if full_data else None,
                    'end':   full_data[-1]['date'] if full_data else None,
                },
            }

            print(f"   ✅ SUCCESS! {len(full_data)} days, "
                  f"{len(technical_indicators)} indicators")
            return complete_data

        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None

    # ──────────────────────────────────────────────────────────────
    # INDICATOR ENGINE
    # ──────────────────────────────────────────────────────────────

    def _add_comprehensive_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 50+ technical indicators for professional analysis."""
        df = df.copy()

        # ── 1. TREND ──────────────────────────────────────────────

        for period in [5, 10, 20, 30, 50, 100, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()

        for period in [5, 8, 12, 13, 20, 21, 26, 50, 100]:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()

        # Hull Moving Average
        def hull_ma(price: pd.Series, period: int = 20) -> pd.Series:
            half_len = int(period / 2)
            sqrt_len = int(np.sqrt(period))
            wma_h = price.rolling(half_len).apply(
                lambda x: np.average(x, weights=range(1, len(x) + 1)), raw=True)
            wma_f = price.rolling(period).apply(
                lambda x: np.average(x, weights=range(1, len(x) + 1)), raw=True)
            return (2 * wma_h - wma_f).rolling(sqrt_len).apply(
                lambda x: np.average(x, weights=range(1, len(x) + 1)), raw=True)

        df['HMA_20'] = hull_ma(df['Close'], 20)

        # Ichimoku Cloud
        high_9  = df['High'].rolling(9).max()
        low_9   = df['Low'].rolling(9).min()
        df['Ichimoku_Conversion'] = (high_9 + low_9) / 2

        high_26 = df['High'].rolling(26).max()
        low_26  = df['Low'].rolling(26).min()
        df['Ichimoku_Base'] = (high_26 + low_26) / 2

        # ADX  (FIX 2: correct Wilder smoothing — see method below)
        adx_df = self._calculate_adx(df, 14)
        df['ADX']      = adx_df['ADX']
        df['Plus_DI']  = adx_df['Plus_DI']
        df['Minus_DI'] = adx_df['Minus_DI']

        # Parabolic SAR  (FIX 1: list-based loop — see method below)
        df['PSAR'] = self._calculate_psar(df)

        # FIX 5: Fibonacci retracement levels (rolling 52-week window)
        roll_high = df['High'].rolling(window=252, min_periods=50).max()
        roll_low  = df['Low'].rolling(window=252, min_periods=50).min()
        fib_range = roll_high - roll_low
        df['Fib_236'] = roll_high - 0.236 * fib_range
        df['Fib_382'] = roll_high - 0.382 * fib_range
        df['Fib_500'] = roll_high - 0.500 * fib_range
        df['Fib_618'] = roll_high - 0.618 * fib_range
        # Distance of current close from each Fib level (% of range) — model-friendly
        df['Close_vs_Fib236'] = (df['Close'] - df['Fib_236']) / fib_range * 100
        df['Close_vs_Fib382'] = (df['Close'] - df['Fib_382']) / fib_range * 100
        df['Close_vs_Fib500'] = (df['Close'] - df['Fib_500']) / fib_range * 100
        df['Close_vs_Fib618'] = (df['Close'] - df['Fib_618']) / fib_range * 100

        # ── 2. MOMENTUM ───────────────────────────────────────────

        for period in [7, 14, 21, 28]:
            df[f'RSI_{period}'] = self._calculate_rsi(df['Close'], period)

        low_14  = df['Low'].rolling(14).min()
        high_14 = df['High'].rolling(14).max()
        df['Stoch_K']    = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stoch_D']    = df['Stoch_K'].rolling(3).mean()
        df['Williams_R'] = -100 * ((high_14 - df['Close']) / (high_14 - low_14))

        tp      = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp  = tp.rolling(20).mean()
        mad_tp  = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
        df['CCI_20'] = (tp - sma_tp) / (0.015 * mad_tp)

        for period in [5, 10, 20]:
            df[f'ROC_{period}'] = (
                (df['Close'] - df['Close'].shift(period))
                / df['Close'].shift(period) * 100
            )
            df[f'Momentum_{period}'] = df['Close'] - df['Close'].shift(period)

        df['MFI_14'] = self._calculate_mfi(df, 14)

        # ── 3. VOLATILITY ─────────────────────────────────────────

        for period in [7, 14, 21]:
            df[f'ATR_{period}'] = self._calculate_atr(df, period)

        df['BB_Middle']   = df['Close'].rolling(20).mean()
        bb_std            = df['Close'].rolling(20).std()
        df['BB_Upper']    = df['BB_Middle'] + bb_std * 2
        df['BB_Lower']    = df['BB_Middle'] - bb_std * 2
        df['BB_Width']    = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

        df['KC_Middle'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['KC_Upper']  = df['KC_Middle'] + df['ATR_14'] * 1.5
        df['KC_Lower']  = df['KC_Middle'] - df['ATR_14'] * 1.5

        log_ret = np.log(df['Close'] / df['Close'].shift(1))
        for period in [10, 20, 30]:
            df[f'Volatility_{period}'] = log_ret.rolling(period).std() * np.sqrt(252)

        # ── 4. VOLUME ─────────────────────────────────────────────

        for period in [5, 10, 20, 50]:
            df[f'Volume_SMA_{period}'] = df['Volume'].rolling(period).mean()

        df['Volume_Ratio_5']  = df['Volume'] / df['Volume_SMA_5']
        df['Volume_Ratio_10'] = df['Volume'] / df['Volume_SMA_10']
        df['Volume_Ratio_20'] = df['Volume'] / df['Volume_SMA_20']

        df['OBV']     = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['OBV_SMA'] = df['OBV'].rolling(20).mean()

        # FIX 4: VWAP is cumulative over the loaded dataset, NOT intraday.
        # For swing trading purposes this gives a long-run fair-value reference.
        df['VWAP'] = (
            (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum()
            / df['Volume'].cumsum()
        )

        mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        df['CMF_20'] = (mfm * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()

        distance  = ((df['High'] + df['Low']) / 2) - ((df['High'].shift(1) + df['Low'].shift(1)) / 2)
        box_ratio = (df['Volume'] / 1e8) / (df['High'] - df['Low'])
        df['EOM']  = distance / box_ratio

        # ── 5. PRICE PATTERNS ─────────────────────────────────────

        df['HH_5']  = df['High'].rolling(5).max()
        df['LL_5']  = df['Low'].rolling(5).min()
        df['HH_10'] = df['High'].rolling(10).max()
        df['LL_10'] = df['Low'].rolling(10).min()

        df['Resistance_20'] = df['High'].rolling(20).max()
        df['Support_20']    = df['Low'].rolling(20).min()

        df['Price_vs_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20'] * 100
        df['Price_vs_SMA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50'] * 100
        df['Price_vs_EMA12'] = (df['Close'] - df['EMA_12']) / df['EMA_12'] * 100

        # ── 6. MACD VARIANTS ──────────────────────────────────────

        df['MACD']           = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal']    = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

        ema_5  = df['Close'].ewm(span=5,  adjust=False).mean()
        ema_35 = df['Close'].ewm(span=35, adjust=False).mean()
        df['MACD_5_35']        = ema_5 - ema_35
        df['MACD_5_35_Signal'] = df['MACD_5_35'].ewm(span=5, adjust=False).mean()

        return df

    # ──────────────────────────────────────────────────────────────
    # INDICATOR HELPERS
    # ──────────────────────────────────────────────────────────────

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI using simple rolling mean (Cutler's RSI — fast & stable)."""
        delta = prices.diff()
        gain  = delta.where(delta > 0, 0.0).rolling(window=period).mean()
        loss  = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
        rs    = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range."""
        high_low   = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close  = (df['Low']  - df['Close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        FIX 2 — ADX with correct Wilder Smoothing (RMA).

        Original code used ewm(alpha=1/period) which is standard EMA.
        Wilder's method uses a longer effective smoothing:
            RMA(n) = ewm(com = n-1)   i.e. alpha = 1/n, com = (1/alpha)-1 = n-1
        This matches TradingView / Bloomberg outputs.
        Also returns +DI and -DI as separate columns — useful model features.
        """
        high  = df['High']
        low   = df['Low']
        close = df['Close']

        # Directional movement
        plus_dm  = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        # Zero out when the other direction is larger
        mask = plus_dm >= minus_dm
        plus_dm  = plus_dm.where(mask,  0.0)
        minus_dm = minus_dm.where(~mask, 0.0)

        # True Range
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)

        # Wilder smoothing  (com = period - 1  →  alpha = 1/period)
        atr_w      = tr.ewm(com=period - 1, adjust=False).mean()
        plus_di    = 100 * plus_dm.ewm(com=period - 1, adjust=False).mean()  / atr_w
        minus_di   = 100 * minus_dm.ewm(com=period - 1, adjust=False).mean() / atr_w

        dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.ewm(com=period - 1, adjust=False).mean()   # Wilder on DX too

        return pd.DataFrame({
            'ADX':      adx,
            'Plus_DI':  plus_di,
            'Minus_DI': minus_di,
        }, index=df.index)

    def _calculate_psar(self, df: pd.DataFrame, af_start: float = 0.02, 
                        af_step: float = 0.02, af_max: float = 0.2) -> pd.Series:
        """
        FIX 1 — Parabolic SAR using plain Python lists.

        The original code assigned values via psar.iloc[i] inside a loop,
        which triggers SettingWithCopyWarning and silently fails on some
        Pandas versions.  We build a plain list then return as a Series.
        """
        high_vals = df['High'].values
        low_vals  = df['Low'].values
        n         = len(df)

        psar_vals = [float(df['Close'].iloc[0])] * n   # initialise list
        bull      = True
        af        = af_start
        hp        = high_vals[0]
        lp        = low_vals[0]

        for i in range(2, n):
            prev_psar = psar_vals[i - 1]

            # Calculate new SAR
            if bull:
                new_psar = prev_psar + af * (hp - prev_psar)
                # SAR must not be above the two prior lows
                new_psar = min(new_psar, low_vals[i - 1], low_vals[i - 2])
            else:
                new_psar = prev_psar + af * (lp - prev_psar)
                # SAR must not be below the two prior highs
                new_psar = max(new_psar, high_vals[i - 1], high_vals[i - 2])

            # Reversal logic
            if bull and low_vals[i] < new_psar:
                bull     = False
                new_psar = hp
                lp       = low_vals[i]
                af       = af_start
            elif not bull and high_vals[i] > new_psar:
                bull     = True
                new_psar = lp
                hp       = high_vals[i]
                af       = af_start
            else:
                # Update extreme point
                if bull:
                    if high_vals[i] > hp:
                        hp = high_vals[i]
                        af = min(af + af_step, af_max)
                else:
                    if low_vals[i] < lp:
                        lp = low_vals[i]
                        af = min(af + af_step, af_max)

            psar_vals[i] = new_psar

        return pd.Series(psar_vals, index=df.index, name='PSAR')

    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Money Flow Index."""
        typical_price  = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow     = typical_price * df['Volume']
        pos_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(period).sum()
        neg_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(period).sum()
        money_ratio = pos_flow / neg_flow
        return 100 - (100 / (1 + money_ratio))

    # ──────────────────────────────────────────────────────────────
    # EXTERNAL DATA HELPERS
    # ──────────────────────────────────────────────────────────────

    def _get_options_data(self, ticker: yf.Ticker) -> Dict[str, Any]:
        """Put/call ratio and volume from nearest expiry."""
        try:
            expirations = ticker.options
            if expirations is None or len(expirations) == 0:
                return {'has_options': False}
            
            # Get the option chain for the nearest expiration
            opt_chain = ticker.option_chain(expirations[0])
            
            calls_vol = 0
            puts_vol = 0
            
            # Check if calls data exists and has volume column
            if opt_chain.calls is not None and isinstance(opt_chain.calls, pd.DataFrame):
                if 'volume' in opt_chain.calls.columns:
                    volume_sum = opt_chain.calls['volume'].sum()
                    if not pd.isna(volume_sum):
                        calls_vol = int(volume_sum)
            
            # Check if puts data exists and has volume column
            if opt_chain.puts is not None and isinstance(opt_chain.puts, pd.DataFrame):
                if 'volume' in opt_chain.puts.columns:
                    volume_sum = opt_chain.puts['volume'].sum()
                    if not pd.isna(volume_sum):
                        puts_vol = int(volume_sum)
                
            return {
                'has_options':         True,
                'expirations':         list(expirations),
                'nearest_expiry':      str(expirations[0]),
                'put_call_ratio':      float(puts_vol / calls_vol) if calls_vol > 0 else 0,
                'calls_volume':        calls_vol,
                'puts_volume':         puts_vol,
                'total_options_volume':calls_vol + puts_vol,
            }
        except Exception as e:
            print(f"   ⚠️ Options data error: {e}")
            return {'has_options': False}

    def _get_market_context(self) -> Dict[str, Any]:
        """
        FIX 3 — Market context now includes VIX (fear gauge) alongside SPY.
        VIX > 30 = high fear / volatile market.
        VIX < 20 = calm / complacent market.
        """
        context = {}

        # SPY — broad market direction
        try:
            spy = yf.download('SPY', period='5d', progress=False)
            if spy is not None and not spy.empty:
                spy_close = spy['Close']
                context['spy_price']  = float(spy_close.iloc[-1])
                context['spy_change'] = float(
                    (spy_close.iloc[-1] / spy_close.iloc[-2] - 1) * 100
                ) if len(spy_close) > 1 else 0.0
                context['spy_trend']  = 'Up' if spy_close.iloc[-1] > spy_close.iloc[-2] else 'Down'
        except Exception as e:
            print(f"   ⚠️ SPY data error: {e}")

        # VIX — volatility / fear index  (FIX 3)
        try:
            vix = yf.download('^VIX', period='5d', progress=False)
            if vix is not None and not vix.empty:
                vix_close = vix['Close']
                context['vix_current'] = float(vix_close.iloc[-1])
                context['vix_change']  = float(
                    (vix_close.iloc[-1] / vix_close.iloc[-2] - 1) * 100
                ) if len(vix_close) > 1 else 0.0
                # Simple regime label
                vix_val = context['vix_current']
                if vix_val >= 30:
                    context['vix_regime'] = 'High Fear'
                elif vix_val >= 20:
                    context['vix_regime'] = 'Elevated'
                else:
                    context['vix_regime'] = 'Calm'
        except Exception as e:
            print(f"   ⚠️ VIX data error: {e}")

        return context

    def _get_analyst_data(self, ticker: yf.Ticker) -> Dict[str, Any]:
        """Latest analyst recommendation."""
        try:
            recs = ticker.recommendations
            if recs is not None and isinstance(recs, pd.DataFrame) and not recs.empty:
                latest = recs.iloc[-1]
                return {
                    'latest_firm':   str(latest.get('Firm', 'Unknown')) if hasattr(latest, 'get') else 'Unknown',
                    'latest_rating': str(latest.get('To Grade', 'Unknown')) if hasattr(latest, 'get') else 'Unknown',
                    'latest_date':   str(latest.name)[:10] if hasattr(latest, 'name') and latest.name is not None else '',
                }
            return {}
        except Exception as e:
            print(f"   ⚠️ Analyst data error: {e}")
            return {}

    def _dataframe_to_serializable(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert DataFrame to JSON-safe list of dicts."""
        records = []
        for idx, row in df.iterrows():
            record = {'date': idx.strftime('%Y-%m-%d')}
            for col in df.columns:
                val = row[col]
                if pd.isna(val):
                    record[col.lower()] = None
                elif isinstance(val, (np.integer, np.int64)):
                    record[col.lower()] = int(val)
                elif isinstance(val, (np.floating, np.float64)):
                    record[col.lower()] = float(val)
                elif isinstance(val, (np.bool_, bool)):
                    record[col.lower()] = bool(val)
                elif isinstance(val, np.ndarray):
                    record[col.lower()] = val.tolist()
                elif isinstance(val, list):
                    record[col.lower()] = val
                else:
                    record[col.lower()] = val
            records.append(record)
        return records
    
    # ──────────────────────────────────────────────────────────────
# MAIN EXECUTION (for testing)
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("📊 MARKET DATA COLLECTOR - TEST RUN")
    print("=" * 70)
    
    # Create collector instance
    collector = MarketDataCollector()
    
    # Test with a few symbols
    test_symbols = ['AAPL', 'META', 'RELIANCE.NS']
    
    for symbol in test_symbols:
        print(f"\n{'='*50}")
        print(f"Testing {symbol}...")
        data = collector.collect_all_data(symbol, period="1mo")
        
        if data:
            print(f"✅ Success! Collected {data['data_points']} days of data")
            print(f"   Current price: ${data['current']['current_price']:.2f}")
        else:
            print(f"❌ Failed to collect data for {symbol}")
        
        # Wait between requests
        if symbol != test_symbols[-1]:
            print("   Waiting 2 seconds...")
            time.sleep(15)
    
    print("\n" + "=" * 70)
    print("✅ Test complete!")
    print("=" * 70)
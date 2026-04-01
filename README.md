# QuantEdge — Stock Signal System ⚡📈

A real-time AI-powered swing trading signal system for US and Indian equity markets. This system collects live market data, engineers 93 technical features, predicts 5-day price direction using a **three-model ensemble** (XGBoost, Random Forest, LSTM), and displays actionable trading signals with **SHAP explainability** and **news sentiment analysis** — all through a premium dark-terminal **Streamlit dashboard**.

---

## 🎯 Goal

Assist swing traders in making data-informed decisions by:
- Predicting whether a stock will be higher or lower 5 trading days from now
- Explaining *why* the model made that call using SHAP feature importance
- Incorporating live news sentiment to validate or dampen model signals
- Honestly suppressing signals when model confidence is insufficient

---

## 🧩 Key Technologies

| Component | Tool Used |
|---|---|
| Data Collection | yfinance (2–3 years OHLCV), Yahoo Finance API |
| Feature Engineering | pandas, NumPy, TA-Lib (93 technical indicators) |
| Model 1 — Gradient Boosting | XGBoost with CalibratedClassifierCV |
| Model 2 — Ensemble Trees | scikit-learn Random Forest (500 estimators) |
| Model 3 — Sequential Learning | TensorFlow / Keras LSTM (20-day sequences) |
| Ensemble Selection | AUC-based best-model routing per stock |
| Explainability | SHAP (TreeExplainer for XGB/RF, GradientExplainer for LSTM) |
| Sentiment NLP | VADER + FinBERT via Alpha Vantage |
| Dashboard UI | Streamlit + Plotly |
| Validation | Walk-forward cross-validation (5 folds, time-based splits) |

---

## 🔁 How It Works

### 📌 Architecture Overview

[System Architecture](images/system_architecture.jpg)

The pipeline runs in 5 sequential stages:

1. 📥 **Data Collection** — 2–3 years of daily OHLCV data fetched per stock with exponential backoff
2. ⚙️ **Feature Engineering** — 93 technical indicators computed; NaN guards applied; 5-day forward label created
3. 🤖 **Model Training** — XGBoost, Random Forest, and LSTM each trained independently per stock with walk-forward validation
4. 🏆 **Stacking Ensemble** — AUC scores compared on held-out test set; best model routed per stock and saved to `model_results.pkl`
5. 📊 **Live Dashboard** — Real-time price, signal, SHAP breakdown, sentiment, and charts served via Streamlit

---

## 🤖 Model Layer & Ensemble Approach

![Model Layer](images/model_layer.jpg)

Three fundamentally different models are trained per stock. Each captures something the others cannot:

| Model | What It Learns | Input Shape |
|---|---|---|
| **XGBoost** | If-then rules across all 93 indicators simultaneously | `(1, 93)` — single row |
| **Random Forest** | Democratic voting across 500 independent decision trees | `(1, 93)` — single row |
| **LSTM** | Sequential patterns across the last 20 trading days | `(20, 93)` — time sequence |
| **Stacking Ensemble** | Routes each stock to its best-performing model by AUC | — |

> LSTM is the only model that reads price data as a time sequence — capturing multi-day patterns like "RSI falling for 3 consecutive days before a MACD crossover" that tree-based models cannot detect.

---

## 📁 Dataset & Coverage

Market data sourced live from **Yahoo Finance** via yfinance. No static dataset — the collector fetches fresh 2–3 year OHLCV history per run.

**US Equities (NASDAQ)**

`AAPL` · `MSFT` · `NVDA` · `TSLA` · `GOOGL`

**Indian Equities (NSE)**

`RELIANCE.NS` · `TCS.NS` · `HDFCBANK.NS` · `INFY.NS`

- 📅 History: 2–3 years of daily candles per stock
- 🧮 Features: 93 technical indicators (RSI, MACD, ATR, Bollinger Bands, VWAP, Ichimoku, OBV, ADX, Fibonacci, SMA/EMA and more)
- 🏷️ Label: Binary — `1` if close price is higher 5 trading days later, `0` if lower
- ✂️ Split: 70% train / 15% validation / 15% test — always time-based, never random

---

## 🛠️ Code Highlights

```python
# Live inference in app.py — how a signal is generated
feature_names = training_data[symbol]['feature_names']   # 93 features
scaler        = training_data[symbol]['scaler']
best_model    = ensemble_models[symbol]                  # 'XGBoost' / 'RF' / 'LSTM'

# For LSTM: build 20-day sequence
X_seq         = np.array([last_20_rows])                 # shape (1, 20, 93)
proba         = lstm_model.predict(X_seq_scaled)

# For XGBoost / RF: single row
X_live        = scaler.transform([today_features])       # shape (1, 93)
proba         = model.predict_proba(X_live)[0][1]        # P(UP in 5 days)
```

---

## 🚀 Setup & Execution Guide

### 1. Environment Setup

> ⚠️ Python **3.10** is required for TensorFlow 2.13.0 compatibility. Other versions will cause dependency conflicts.

```bash
# Create the virtual environment with Python 3.10 specifically
py -3.10 -m venv venv

# Activate the environment
.\venv\Scripts\activate

# Upgrade pip to the latest version
python -m pip install --upgrade pip

# Install the critical dependency bridge first
pip install "typing-extensions>=4.5.0,<4.6.0"

# Install all remaining requirements using the legacy resolver
pip install -r requirements.txt --use-deprecated=legacy-resolver
```

### 2. Execution Pipeline (Run in Order)

All `.pkl` model files are generated automatically by running the scripts below — do not create them manually.

| Step | Script | Description |
|---|---|---|
| 1 | `market_data_collector.py` | Fetches historical price data (OHLCV) from Yahoo Finance |
| 2 | `news_sentiment.py` | *(Optional)* Processes financial news for sentiment scores |
| 3 | `feature_engineering.py` | Generates 93 technical indicators (RSI, MACD, ATR, etc.) and creates labels |
| 4 | `train_rf.py` | Trains the Random Forest base model |
| 5 | `train_xgboost.py` | Trains the XGBoost base model |
| 6 | `lstm_model.py` | Trains the Deep Learning LSTM model via TensorFlow |
| 7 | `improved_stacking.py` | Combines all models into a meta-ensemble, selects best model per stock by AUC |
| 8 | `strategy_analysis.py` | Runs backtests to simulate trading performance |
| 9 | `generate_stats_report.py` | Compiles results into final PDF and PNG visualisations |
| 10 | `app.py` | Launches the live Streamlit dashboard |

```bash
# Run in this exact order
python market_data_collector.py
python news_sentiment.py          # optional but recommended
python feature_engineering.py
python train_rf.py
python train_xgboost.py
python lstm_model.py              # takes 10–20 min
python improved_stacking.py
python strategy_analysis.py
python generate_stats_report.py
streamlit run app.py
```

---

## 📂 Project Structure

```
QuantEdge/
│
├── app.py                       # Streamlit dashboard — premium dark-terminal UI
├── market_data_collector.py     # Stage 1: Fetches OHLCV data from Yahoo Finance
├── news_sentiment.py            # Stage 2: VADER + FinBERT sentiment pipeline
├── feature_engineering.py       # Stage 3: Computes 93 features, creates labels, saves scaler
├── train_rf.py                  # Stage 4: Random Forest training (500 estimators)
├── train_xgboost.py             # Stage 5: XGBoost training with calibration
├── lstm_model.py                # Stage 6: LSTM sequence model (20-day windows)
├── improved_stacking.py         # Stage 7: AUC-based ensemble — picks best model per stock
├── strategy_analysis.py         # Stage 8: Backtesting and trading simulation
├── generate_stats_report.py     # Stage 9: PDF + PNG performance report generation
│
├── requirements.txt             # All Python dependencies
├── images/                      # README screenshots and slide exports
│   ├── system_architecture.png
│   ├── model_layer.png
│   ├── dashboard_overview.png
│   ├── dashboard_signals.png
│   ├── signal_explainer.png
│   ├── model_performance.png
│   └── shap_plot.png
└── README.md
```

> All `.pkl` files (`model_results.pkl`, `training_data.pkl`, `xgb_models.pkl`, etc.) are generated automatically when you run the pipeline above — they are not committed to the repository.

---

## ⚠️ Limitations

This section exists because honest evaluation is more valuable than overclaiming.

| Limitation | Detail |
|---|---|
| **Prediction horizon** | 5-day direction only — not magnitude, not intraday moves |
| **Dataset size** | ~350–500 training rows per stock; LSTM is most affected by this constraint |
| **TSLA, HDFCBANK, INFY** | Event-driven or range-bound stocks — technical indicators alone cannot predict them reliably; signals are suppressed automatically |
| **Backtest ≠ forward performance** | A backtest AUC of 0.75 does not mean 75% accuracy on truly unseen future data; market regimes shift |
| **Python version** | Must use Python 3.10 exactly — TensorFlow 2.13.0 is incompatible with 3.11+ |
| **Not financial advice** | This is an academic capstone project — do not use signals for real capital allocation |

> "Our system correctly identifies TSLA and HDFCBANK as unpredictable from technical indicators — that's a feature, not a bug. A model that knows when not to trade is more useful than one that's always confidently wrong."

---

## 📊 Sample Output

```
Stock:        NVDA  (NASDAQ)
Model Used:   XGBoost  ·  AUC 0.752
Signal:       ▲▲ STRONG BUY
Confidence:   76.3% UP probability
Agreement:    High  (log-odds margin: 1.84 from neutral)

SHAP — Top Features Driving This Signal:
  RSI_14           −0.12  → SELL pressure
  MACD_diff        −0.09  → SELL pressure
  volume_ratio     −0.07  → SELL pressure
  EMA_cross        +0.04  → BUY support
  ATR_14           −0.03  → SELL pressure

─────────────────────────────────────────
Stock:        HDFCBANK.NS  (NSE)
Model Used:   AUC 0.477 — below reliability threshold
Signal:       ⚠ LOW CONFIDENCE  (suppressed)
```

✔️ Final outputs generated by `generate_stats_report.py`:
- Performance report as `.pdf`
- AUC and accuracy charts as `.jpg`
- Backtest simulation results via `strategy_analysis.py`

---

## 🖥️ Dashboard Features

![Dashboard](images/dashboard_overview.jpg)

### KPI Cards
Five cards update in real time for the selected stock — Current Price · RSI (14) · News Sentiment · Volume Ratio · AI Signal

### Signal Explainer
Clicking **"🔍 Explain This Signal"** reveals:
- Model used and confidence probability
- **Random Forest:** exact vote count from all 500 trees (e.g. 387 BUY vs 113 SELL)
- **XGBoost:** raw log-odds margin from the neutral point
- **LSTM:** probability distance from 50/50 midpoint across 20-day sequence
- **SHAP bar chart** — top 5 features ranked by impact on prediction

### Interactive Chart
Full candlestick chart with SMA 20/50 overlay, volume bars, and RSI subplot — adjustable date range via picker.

### Tabs

| Tab | Content |
|---|---|
| 📰 News & Sentiment | Up to 10 recent articles with VADER score, source, date, and read link |
| 📊 Technical Indicators | RSI, MACD, ATR, Volume Ratio, SMA grid, Indicator Reference glossary |
| 🏢 Company Profile | Sector, industry, market cap, P/E ratio, beta, dividend yield |
| 🤖 AI Model Performance | AUC by model, accuracy bar chart, top stocks table, confident signal count |

---

## 🔍 SHAP Explainability

![SHAP Plot](images/shap_plot.jpg)

Every signal includes a SHAP (SHapley Additive exPlanations) breakdown showing which technical features pushed the model toward BUY or SELL — and by how much.

```python
# XGBoost / Random Forest
import shap
explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# LSTM
explainer   = shap.GradientExplainer(lstm_model, X_background)
shap_values = explainer.shap_values(X_test)
```

Top features by average SHAP impact across all stocks:

| Feature | Role |
|---|---|
| `RSI_14` | Momentum — oversold/overbought detection |
| `MACD_diff` | Trend direction and crossover strength |
| `ATR_14` | Volatility — filters out high-noise periods |
| `SMA_20_50_cross` | Medium-term trend confirmation |
| `volume_ratio` | Unusual activity detection |

---

## 📈 Model Performance Results

Test set = most recent 15% of data (Nov 2025 → Mar 2026), never seen during training.

| Stock | Best Model | AUC | Confident Accuracy | Signals Fired |
|---|---|---|---|---|
| **NVDA** | XGBoost | **0.752** | 78.3% | 46 / 76 days |
| **AAPL** | Random Forest | 0.659 | 88.9% | 18 / 76 days |
| **MSFT** | XGBoost | 0.634 | 68.8% | 16 / 76 days |
| **GOOGL** | Ensemble | 0.616 | 58.9% | 56 / 76 days |
| **TCS.NS** | LSTM | 0.615 | 100% | 1 / 75 days |
| **RELIANCE.NS** | LSTM | 0.584 | 66.7% | 15 / 75 days |
| **HDFCBANK.NS** | — | 0.477 | Suppressed | — |
| **INFY.NS** | — | 0.511 | Suppressed | — |
| **TSLA** | — | 0.420 | Suppressed | — |

> LSTM justified its inclusion by winning on TCS and RELIANCE — both NSE stocks with smoother multi-day trends where sequential modelling outperformed tree-based approaches. Stocks with AUC < 0.60 are automatically gated and show `LOW CONFIDENCE` instead of a misleading signal.

---

## 🚧 Project Status & Vision

This project is a **fully functional capstone submission**. The end-to-end pipeline — from data collection through live dashboard — runs completely and generates real AI-driven signals from trained model objects.

### 🛠️ Current Status
- Full 5-stage pipeline works end-to-end ✓
- Live price feed via yfinance ✓
- Real model inference (not rule-based fallback) ✓
- Sentiment dampening integrated ✓
- AUC confidence gate active ✓
- SHAP signal breakdown in dashboard ✓

### 🎯 Future Vision
We aim to evolve QuantEdge into a production-grade signal platform by:
- Replacing the 58% confidence threshold with a dynamically calibrated per-stock threshold based on rolling AUC
- Integrating live SHAP computation (currently illustrative values are shown; architecture is complete)
- Adding a correlation heatmap between US and Indian market returns
- Extending to positional trading signals (holding period: 3–6 weeks) with a separate feature set including fundamentals

> This system is designed with transparency at its core — a model that suppresses weak signals is more useful to a trader than one that always speaks confidently.

---

## 🔮 Future Work

- Live SHAP computation per prediction (pipeline complete, awaiting final model freeze)
- Walk-forward threshold calibration per stock
- US vs India return correlation heatmap (Plotly, ~10 lines)
- Positional trading mode with earnings dates, sector rotation, and 200-day SMA features
- Deploy as public web app with Streamlit Community Cloud + custom domain
- Multilingual news sentiment (Hindi financial news via IndicBERT)

---

## 👨‍💻 Contributors

- **Amritha K**
- **Garima Mishra** 


**Institution:** VIT Chennai- Capstone 2026

---

## 📌 Tags

`#StockMarket` `#SwingTrading` `#MachineLearning` `#XGBoost` `#LSTM` `#RandomForest` `#SHAP` `#ExplainableAI` `#SentimentAnalysis` `#Streamlit` `#FinTech` `#Ensemble` `#TechnicalAnalysis`

---

## 📖 Citations

If you use any of the tools, models, or datasets referenced in this project, please cite the following:

---

### 📌 XGBoost

**Paper:** [XGBoost: A Scalable Tree Boosting System (KDD 2016)](https://arxiv.org/abs/1603.02754)

```bibtex
@inproceedings{chen2016xgboost,
  title     = {XGBoost: A Scalable Tree Boosting System},
  author    = {Chen, Tianqi and Guestrin, Carlos},
  booktitle = {Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages     = {785--794},
  year      = {2016}
}
```

---

### 📌 SHAP (SHapley Additive exPlanations)

**Paper:** [A Unified Approach to Interpreting Model Predictions (NeurIPS 2017)](https://arxiv.org/abs/1705.07874)

```bibtex
@inproceedings{lundberg2017unified,
  title     = {A Unified Approach to Interpreting Model Predictions},
  author    = {Lundberg, Scott M and Lee, Su-In},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {30},
  year      = {2017}
}
```

---

### 📌 LSTM (Long Short-Term Memory)

**Paper:** [Long Short-Term Memory (Neural Computation, 1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)

```bibtex
@article{hochreiter1997lstm,
  title   = {Long Short-Term Memory},
  author  = {Hochreiter, Sepp and Schmidhuber, J{\"u}rgen},
  journal = {Neural Computation},
  volume  = {9},
  number  = {8},
  pages   = {1735--1780},
  year    = {1997}
}
```

---

### 📌 Random Forest

**Paper:** [Random Forests (Machine Learning, 2001)](https://link.springer.com/article/10.1023/A:1010933404324)

```bibtex
@article{breiman2001random,
  title   = {Random Forests},
  author  = {Breiman, Leo},
  journal = {Machine Learning},
  volume  = {45},
  number  = {1},
  pages   = {5--32},
  year    = {2001}
}
```

---

### 📌 VADER Sentiment

**Paper:** [VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text (ICWSM 2014)](https://ojs.aaai.org/index.php/ICWSM/article/view/14550)

```bibtex
@inproceedings{hutto2014vader,
  title     = {VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text},
  author    = {Hutto, Clayton and Gilbert, Eric},
  booktitle = {Proceedings of the International AAAI Conference on Web and Social Media},
  volume    = {8},
  number    = {1},
  year      = {2014}
}
```

---

### 📌 yfinance

Library: [`yfinance`](https://github.com/ranaroussi/yfinance)
Author: Ran Aroussi

### ⚠️ Disclaimer

This project is for educational purposes only.
It provides probabilistic insights, not financial advice.

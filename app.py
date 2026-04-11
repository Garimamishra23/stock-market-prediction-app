# app.py — QuantEdge Premium UI · S-Grade Capstone Edition with Advanced SHAP
# ══════════════════════════════════════════════════════════════
# ENHANCED FEATURES:
#   1. SHAP Interaction Effects - Feature inmodel_type = 'XGBoost'

#   2. SHAP Time-Series Evolution - For LSTM sequence importance
#   3. SHAP Stability Analysis - Bootstrap confidence intervals
#   4. SHAP Confidence Correlation - Validation of explanation quality
# ══════════════════════════════════════════════════════════════
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import glob
import pickle
from datetime import datetime, timedelta
from dateutil import parser as dateutil_parser
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# ── SHAP imports ──────────────────────────────────────────────
SHAP_AVAILABLE = False
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────
# SAFE CONVERSION HELPERS
# ─────────────────────────────────────────────────────────────────

def safe_float(value):
    if value is None or value == '' or value == 'N/A':
        return None
    try:
        f = float(value)
        return None if np.isnan(f) else f
    except (ValueError, TypeError):
        return None

def _fmt(value, fmt="{:.2f}", none_str="—") -> str:
    if value is None:
        return none_str
    try:
        return fmt.format(value)
    except Exception:
        return none_str

def safe_float_convert(value, default=0.0) -> float:
    result = safe_float(value)
    return result if result is not None else default

# ─────────────────────────────────────────────────────────────────
# MODEL LOADING WITH CACHING
# ─────────────────────────────────────────────────────────────────

_model_cache = {}
_shap_cache = {}              # for regular SHAP explainers
_shap_interaction_cache = {}  # for interaction SHAP values

def _load_once(path, key):
    if key in _model_cache:
        return _model_cache[key]
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            _model_cache[key] = pickle.load(f)
        return _model_cache[key]
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────
# FIX-A: load_model_results — read model_results.pkl directly
# ─────────────────────────────────────────────────────────────────

@st.cache_data
def load_model_results():
    mr = _load_once('model_results.pkl', 'model_results')
    if mr:
        return mr

    results = {}
    xgb_r = _load_once('xgb_results.pkl', 'xgb_results')
    rf_r  = _load_once('rf_results.pkl',  'rf_results')

    all_symbols = set()
    if xgb_r: all_symbols.update(xgb_r.keys())
    if rf_r:  all_symbols.update(rf_r.keys())

    for sym in all_symbols:
        xgb_acc  = float(xgb_r.get(sym, 0.5)) if xgb_r else 0.5
        rf_acc   = float(rf_r.get(sym,  0.5)) if rf_r  else 0.5
        best_acc = max(xgb_acc, rf_acc)
        results[sym] = {
            'best_model':        'XGBoost' if xgb_acc >= rf_acc else 'Random Forest',
            'best_acc':          best_acc,
            'best_auc':          0.5,
            'best_f1':           0.5,
            'xgb_auc':           0.5,
            'rf_auc':            0.5,
            'lstm_auc':          0.5,
            'ensemble_auc':      0.5,
            'confident_signals': 0,
            'confident_acc':     0.5,
            'best_individual':   best_acc,
            'weighted':          (xgb_acc + rf_acc) / 2,
            'selective':         best_acc,
            'best_method':       best_acc,
        }

    return results if results else None

# ─────────────────────────────────────────────────────────────────
# FIX-B + FIX-1/2/3: get_live_prediction — full latest version
# ─────────────────────────────────────────────────────────────────

def get_live_prediction(symbol):
    current_price = None
    daily_return  = 0.0
    volume_ratio  = 1.0

    if YFINANCE_AVAILABLE:
        try:
            ticker = yf.Ticker(symbol)
            hist   = ticker.history(period='5d')
            if not hist.empty:
                current_price = float(hist['Close'].iloc[-1])
                if len(hist) > 1:
                    prev         = float(hist['Close'].iloc[-2])
                    daily_return = (current_price - prev) / prev if prev else 0.0
                vol_avg      = hist['Volume'].mean()
                volume_ratio = float(hist['Volume'].iloc[-1] / vol_avg) if vol_avg > 0 else 1.0
        except Exception:
            pass

    training_data = _load_once('training_data.pkl', 'training_data')
    xgb_models    = _load_once('xgb_models.pkl',    'xgb_models')
    rf_models     = _load_once('rf_models.pkl',     'rf_models')
    ensemble_mdls = _load_once('ensemble_models.pkl','ensemble_models')

    confidence = None
    model_name = 'rule-based fallback'
    agreement_data = {'label': 'Unknown', 'type': 'unknown'}

    # FIX-2: AUC gate
    mr_check = _load_once('model_results.pkl', 'model_results')
    best_auc      = (mr_check or {}).get(symbol, {}).get('best_auc', 0.5)
    best_model_nm = (mr_check or {}).get(symbol, {}).get('best_model', 'Unknown')

    

    # AUC gate — only suppress genuinely worse than random
    if best_auc < 0.50:
        if daily_return > 0.005:
            fallback_signal = 'WEAK BUY'
        elif daily_return < -0.005:
            fallback_signal = 'WEAK SELL'
        else:
            fallback_signal = 'HOLD'
        return {
            'signal':           fallback_signal,
            'confidence':       best_auc,
            'price':            current_price,
            'change':           daily_return * 100,
            'volume_ratio':     volume_ratio,
            'model_used':       f'Momentum fallback · {best_model_nm} AUC {best_auc:.3f}',
            'auc_gated':        True,
            'auc_gated_reason': f'Model AUC {best_auc:.3f} is below random chance (0.50). '
                                f'Showing momentum-based directional signal only.',
        }

    if training_data and symbol in training_data:
        stock_td      = training_data[symbol]
        feature_names = stock_td.get('feature_names', [])
        scaler        = stock_td.get('scaler')
        # Read declared best model from ensemble_models.pkl
        # This now correctly includes LSTM when it's the best model
        best_model_type = 'XGBoost'
        if ensemble_mdls and symbol in ensemble_mdls:
            declared = ensemble_mdls[symbol]
        try:
            st.session_state['shap_debug2'] = f"model_type:{model_type}"
        except:
            pass
            if declared == 'None':
                best_model_type = 'XGBoost'  # fallback for excluded stocks
            else:
                best_model_type = declared

        # Load JSON full_dataframe
        full_df = []
        json_files = glob.glob("global_market_data_*.json")
        if json_files:
            try:
                with open(sorted(json_files)[-1], 'r', encoding='utf-8') as f:
                    raw_json = json.load(f)
                stock_entry  = raw_json.get(symbol, {})
                market_block = stock_entry.get('market_data', stock_entry)
                full_df      = market_block.get('full_dataframe', [])
            except Exception:
                full_df = []
        return []

    # Branch A — XGBoost / Random Forest
    if model_type in ('XGBoost', 'Random Forest', 'Ensemble'):
        model = None
        resolved_type = model_type
        if model_type in ('XGBoost', 'Ensemble'):
            if xgb_models and symbol in xgb_models:
                model = xgb_models[symbol]
                resolved_type = 'XGBoost'
        if model is None and rf_models and symbol in rf_models:
            model = rf_models[symbol]
            resolved_type = 'Random Forest'
        if model is None:
            return []

        try:
            latest_row = full_df[-1]
            feat_vec   = [float(latest_row.get(f.lower()) or 0.0) for f in feature_names]
            X_live     = np.array([feat_vec])
            X_scaled   = scaler.transform(X_live)

            base_model = model
            if hasattr(model, 'calibrated_classifiers_'):
                base_model = model.calibrated_classifiers_[0].estimator
            elif hasattr(model, 'base_estimator_'):
                base_model = model.base_estimator_

            cache_key = (symbol, resolved_type)
            if cache_key not in _shap_cache:
                _shap_cache[cache_key] = shap.TreeExplainer(
                    base_model,
                    feature_perturbation="interventional",
                )
            explainer = _shap_cache[cache_key]

            sv = explainer.shap_values(X_scaled)
            if isinstance(sv, list) and len(sv) == 2:
                vals = np.array(sv[1]).flatten()
            elif isinstance(sv, list) and len(sv) == 1:
                vals = np.array(sv[0]).flatten()
            elif hasattr(sv, 'ndim') and sv.ndim == 3:
                vals = sv[0, :, 1]
            else:
                vals = np.array(sv).flatten()

        except Exception:
            st.session_state['shap_debug2'] = f"XGB/RF SHAP failed: {str(e)[:300]}"
            return []

    # Branch B — LSTM
    elif model_type == 'LSTM':
        if not lstm_models or symbol not in lstm_models:
            return []
        if len(full_df) < 20:
            return []

        try:
            lstm_model = lstm_models[symbol]
            last_20 = full_df[-20:]
            seq = []
            for row in last_20:
                fv = [float(row.get(f.lower()) or 0.0) for f in feature_names]
                seq.append(fv)
            X_seq         = np.array([seq])
            X_flat        = X_seq.reshape(-1, X_seq.shape[-1])
            X_flat_scaled = scaler.transform(X_flat)
            X_input       = X_flat_scaled.reshape(1, 20, -1)

            cache_key = (symbol, 'LSTM')
            if cache_key not in _shap_cache:
                if X_train is not None and len(X_train) >= 20:
                    n       = len(X_train)
                    indices = np.linspace(20, n - 1, min(10, n - 20), dtype=int)
                    bg_seqs = np.array([X_train[i - 20:i] for i in indices])
                    bg_flat = bg_seqs.reshape(-1, bg_seqs.shape[-1])
                    bg_scaled   = scaler.transform(bg_flat)
                    background  = bg_scaled.reshape(len(indices), 20, -1)
                else:
                    background = np.zeros((5, 20, len(feature_names)), dtype=np.float32)
                _shap_cache[cache_key] = shap.GradientExplainer(lstm_model, background)

            explainer = _shap_cache[cache_key]
            sv = explainer.shap_values(X_input)
            if isinstance(sv, list):
                sv = sv[0]
            sv_arr = np.array(sv)
            mean_abs  = np.abs(sv_arr[0]).mean(axis=0)
            last_sign = np.sign(sv_arr[0, -1, :])
            vals      = mean_abs * last_sign

        except Exception:
            st.session_state['shap_debug2'] = f"LSTM SHAP failed: {str(e)[:300]}"
            return []

    else:
        return []

    if len(vals) != len(feature_names):
        return []

    indexed = sorted(enumerate(vals), key=lambda x: abs(x[1]), reverse=True)[:7]
    result  = []
    for idx, sv_val in indexed:
        result.append({
            'name':      feature_names[idx],
            'shap':      round(float(sv_val), 4),
            'direction': 'BUY' if sv_val > 0 else 'SELL',
        })
    return result

# ─────────────────────────────────────────────────────────────────
# ADVANCED SHAP UI DISPLAY
# ─────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────
# ENHANCED ADVANCED SHAP UI WITH MULTIPLE PLOT TYPES (FIXED)
# ─────────────────────────────────────────────────────────────────

def display_advanced_shap_analysis(symbol: str, live_pred: dict):
    """Display advanced SHAP analysis with multiple visualization types"""
    if not SHAP_AVAILABLE:
        st.info("🔧 Install SHAP library to enable advanced analysis: `pip install shap`")
        return
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #071829 0%, #040f22 100%);
                border: 1px solid rgba(0,200,255,0.2);
                border-radius: 12px;
                padding: 1rem;
                margin: 1rem 0;">
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
            <span style="font-size: 2rem;">🎓</span>
            <div>
                <span style="font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 600; color: #00c8ff;">
                    Advanced SHAP Analysis
                </span>
                <span style="font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #3a7ca5; margin-left: 1rem;">
                    S-GRADE FEATURE
                </span>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; color: #5ba3cc;">
                    Feature Interactions · Time-Series Importance · Confidence Validation
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different visualizations
    adv_tab1, adv_tab2, adv_tab3, adv_tab4 = st.tabs([
        "📊 SHAP Summary Statistics",
        "🎯 Signal Confidence Breakdown", 
        "🔬 SHAP Stability Analysis",
        "🌐 SHAP Feature Importance"
    ])
    
    with adv_tab1:
        st.markdown("""
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;
                    color:#3a7ca5;margin-bottom:1rem;">
            <strong>📊 SHAP Value Distribution — What Drives This Model</strong><br>
            Statistical summary of how much each feature category contributes
            to predictions on average across all training samples.
        </div>
        """, unsafe_allow_html=True)

        try:
            training_data = _load_once('training_data.pkl', 'training_data')
            xgb_models    = _load_once('xgb_models.pkl', 'xgb_models')
            rf_models     = _load_once('rf_models.pkl', 'rf_models')

            model = None
            if xgb_models and symbol in xgb_models:
                model = xgb_models[symbol]
            elif rf_models and symbol in rf_models:
                model = rf_models[symbol]

            if training_data and symbol in training_data and model:
                stock_td      = training_data[symbol]
                X_train       = stock_td.get('X_train')
                feature_names = stock_td.get('feature_names', [])

                if X_train is not None and len(X_train) > 0:
                    sample = X_train[:min(150, len(X_train))]
                    base_model = model
                    if hasattr(model, 'calibrated_classifiers_'):
                        base_model = model.calibrated_classifiers_[0].estimator

                    explainer   = shap.TreeExplainer(base_model)
                    shap_vals   = explainer.shap_values(sample)
                    if isinstance(shap_vals, list) and len(shap_vals) == 2:
                        shap_vals = shap_vals[1]

                    mean_abs = np.abs(shap_vals).mean(axis=0)

                    # Group by category
                    cat_totals = {
                        'Momentum':    0.0,
                        'Trend':       0.0,
                        'Volume':      0.0,
                        'Volatility':  0.0,
                        'Price Level': 0.0,
                        'Other':       0.0,
                    }
                    cat_map = {
                        'momentum':    'Momentum',
                        'trend':       'Trend',
                        'volume':      'Volume',
                        'volatility':  'Volatility',
                        'price_level': 'Price Level',
                    }
                    for i, fname in enumerate(feature_names):
                        cat = FEATURE_CATEGORY.get(fname.lower(), 'other')
                        label = cat_map.get(cat, 'Other')
                        cat_totals[label] += float(mean_abs[i]) if i < len(mean_abs) else 0.0

                    total = sum(cat_totals.values()) or 1.0
                    cat_pct = {k: v/total*100 for k, v in cat_totals.items() if v > 0}

                    # Donut chart
                    colors = ['#00c8ff','#00e896','#f5a623','#b06eff','#ff4b6e','#3a7ca5']
                    fig_donut = go.Figure(data=[go.Pie(
                        labels=list(cat_pct.keys()),
                        values=list(cat_pct.values()),
                        hole=0.55,
                        marker=dict(colors=colors[:len(cat_pct)],
                                    line=dict(color='#020912', width=2)),
                        textfont=dict(family='IBM Plex Mono', size=11, color='#c8d8e8'),
                        hovertemplate='<b>%{label}</b><br>Contribution: %{value:.1f}%<extra></extra>',
                    )])
                    fig_donut.update_layout(
                        title=dict(
                            text="Prediction Influence by Indicator Category",
                            font=dict(family='Syne, sans-serif', size=14, color='#c8d8e8'), x=0
                        ),
                        paper_bgcolor='rgba(4,15,34,0)',
                        plot_bgcolor='rgba(4,15,34,0)',
                        font=dict(color='#8fb0cc'),
                        height=380,
                        legend=dict(
                            font=dict(family='IBM Plex Mono', size=11, color='#8fb0cc'),
                            bgcolor='rgba(0,0,0,0)'
                        ),
                        annotations=[dict(
                            text='Feature<br>Mix',
                            x=0.5, y=0.5, font_size=13,
                            font=dict(family='Syne', color='#c8d8e8'),
                            showarrow=False
                        )]
                    )
                    st.plotly_chart(fig_donut, use_container_width=True)

                    # Category bars
                    st.markdown("<br><span style='color:#00c8ff;font-size:0.85rem;'>📋 Category Breakdown</span>", unsafe_allow_html=True)
                    for cat, pct in sorted(cat_pct.items(), key=lambda x: -x[1]):
                        bar_w = int(pct * 3)
                        st.markdown(
                            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
                            f'<span style="font-family:IBM Plex Mono,monospace;font-size:0.75rem;color:#c8d8e8;width:100px;">{cat}</span>'
                            f'<div style="background:#00c8ff;height:8px;width:{bar_w}px;border-radius:4px;opacity:0.7;"></div>'
                            f'<span style="font-family:IBM Plex Mono,monospace;font-size:0.75rem;color:#00c8ff;">{pct:.1f}%</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No training data available for this symbol.")
            else:
                st.info("Trained model required for SHAP summary statistics.")
        except Exception as e:
            st.info(f"SHAP summary unavailable: {str(e)[:80]}")
           
    
    with adv_tab2:
        st.markdown("""
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;
                    color:#3a7ca5;margin-bottom:1rem;">
            <strong>🎯 Signal Confidence Breakdown — How Sure Is The Model?</strong><br>
            Shows how prediction confidence is distributed across all 9 stocks
            and where this stock sits relative to others.
        </div>
        """, unsafe_allow_html=True)

        try:
            mr = _load_once('model_results.pkl', 'model_results')
            if mr and len(mr) > 1:
                stocks  = list(mr.keys())
                aucs    = [mr[s].get('best_auc', 0.5) for s in stocks]
                models  = [mr[s].get('best_model', 'XGBoost') for s in stocks]
                colors  = []
                for s, a in zip(stocks, aucs):
                    if s == symbol:
                        colors.append('#f5a623')   # highlight current
                    elif a >= 0.62:
                        colors.append('#00e896')
                    elif a >= 0.58:
                        colors.append('#00c8ff')
                    elif a >= 0.55:
                        colors.append('#b06eff')
                    else:
                        colors.append('#ff4b6e')

                clean_labels = [s.replace('.NS','') for s in stocks]

                fig_bar = go.Figure(go.Bar(
                    x=clean_labels,
                    y=aucs,
                    marker=dict(color=colors, line=dict(width=0)),
                    text=[f"{a:.3f}" for a in aucs],
                    textposition='outside',
                    textfont=dict(family='IBM Plex Mono', size=10, color='#8fb0cc'),
                    hovertemplate='<b>%{x}</b><br>AUC: %{y:.3f}<extra></extra>',
                ))
                fig_bar.add_hline(
                    y=0.60,
                    line=dict(color='#f5a623', width=1.5, dash='dash'),
                    annotation_text='Reliability threshold (0.60)',
                    annotation_font_color='#f5a623',
                    annotation_font_size=10,
                )
                fig_bar.add_hline(
                    y=0.5,
                    line=dict(color='#ff4b6e', width=1, dash='dot'),
                    annotation_text='Random chance (0.50)',
                    annotation_font_color='#ff4b6e',
                    annotation_font_size=10,
                )
                fig_bar.update_layout(
                    title=dict(
                        text=f"AUC Scores Across All Stocks  ·  {symbol.replace('.NS','')} highlighted in orange",
                        font=dict(family='Syne, sans-serif', size=14, color='#c8d8e8'), x=0
                    ),
                    yaxis=dict(
                        title="AUC Score", range=[0.4, max(aucs)+0.08],
                        gridcolor='rgba(0,200,255,0.05)',
                        tickfont=dict(family='IBM Plex Mono', size=10, color='#3a7ca5'),
                        zeroline=False,
                    ),
                    xaxis=dict(tickfont=dict(family='IBM Plex Mono', size=10, color='#c8d8e8')),
                    plot_bgcolor='rgba(4,15,34,0)',
                    paper_bgcolor='rgba(4,15,34,0)',
                    font=dict(color='#8fb0cc'),
                    height=400,
                    bargap=0.3,
                    margin=dict(l=0, r=0, t=50, b=0),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # Current stock callout
                cur_auc = mr.get(symbol, {}).get('best_auc', 0.5)
                rank    = sorted(aucs, reverse=True).index(cur_auc) + 1
                tier    = (
                    "🟢 High confidence — signals are reliable"        if cur_auc >= 0.62 else
                    "🔵 Moderate confidence — signals are directional" if cur_auc >= 0.58 else
                    "🟡 Low confidence — treat signals with caution"   if cur_auc >= 0.55 else
                    "🔴 Below threshold — signals suppressed"
                )
                st.markdown(f"""
                <div style="background:#040f22;border:1px solid rgba(245,166,35,0.3);
                            border-radius:8px;padding:1rem;margin-top:0.5rem;">
                    <span style="color:#f5a623;font-family:'IBM Plex Mono',monospace;
                                 font-size:0.7rem;">CURRENT STOCK — {symbol.replace('.NS','')}</span><br>
                    <span style="color:#e8f4ff;font-family:'Inter',sans-serif;font-size:0.9rem;">
                        AUC Score: <strong>{cur_auc:.3f}</strong> &nbsp;·&nbsp;
                        Rank: <strong>#{rank}</strong> of {len(stocks)} stocks<br>
                        {tier}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Model results required for confidence breakdown.")
        except Exception as e:
            st.info(f"Confidence breakdown unavailable: {str(e)[:80]}")
    
    with adv_tab3:
        st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;
                color:#3a7ca5;margin-bottom:1rem;">
        <strong>🔬 SHAP Stability Analysis — Bootstrap Confidence Intervals</strong><br>
        Validates explanation robustness by measuring SHAP value variance across 30 bootstrap samples.
        Narrow intervals = stable, trustworthy explanations.
    </div>
    """, unsafe_allow_html=True)

        try:
            training_data = _load_once('training_data.pkl', 'training_data')
            xgb_models    = _load_once('xgb_models.pkl', 'xgb_models')
            rf_models     = _load_once('rf_models.pkl', 'rf_models')

            model = None
            if xgb_models and symbol in xgb_models:
                model = xgb_models[symbol]
            elif rf_models and symbol in rf_models:
                model = rf_models[symbol]

            if training_data and symbol in training_data and model:
                stock_td      = training_data[symbol]
                X_train       = stock_td.get('X_train')
                feature_names = stock_td.get('feature_names', [])

                if X_train is not None and len(X_train) >= 30:
                    base_model = model
                    if hasattr(model, 'calibrated_classifiers_'):
                        base_model = model.calibrated_classifiers_[0].estimator

                    explainer = shap.TreeExplainer(base_model)

                    n_bootstrap = 30
                    n_sample    = min(60, len(X_train))
                    boot_means  = []
                    for _ in range(n_bootstrap):
                        idx   = np.random.choice(len(X_train), n_sample, replace=True)
                        sv    = explainer.shap_values(X_train[idx])
                        if isinstance(sv, list) and len(sv) == 2:
                            sv = sv[1]
                        boot_means.append(np.abs(sv).mean(axis=0))

                    boot_arr = np.array(boot_means)
                    means    = boot_arr.mean(axis=0)
                    ci_low   = np.percentile(boot_arr, 2.5,  axis=0)
                    ci_high  = np.percentile(boot_arr, 97.5, axis=0)

                    top_idx      = np.argsort(means)[-10:][::-1]
                    top_names    = [feature_names[i] for i in top_idx]
                    top_means    = means[top_idx]
                    top_low      = ci_low[top_idx]
                    top_high     = ci_high[top_idx]
                    top_ci_width = top_high - top_low

                    fig_stab = go.Figure()
                    fig_stab.add_trace(go.Bar(
                        x=top_names, y=top_ci_width,
                        base=top_low,
                        name='95% CI',
                        marker=dict(color='rgba(0,200,255,0.15)',
                                    line=dict(color='rgba(0,200,255,0.4)', width=1)),
                    ))
                    fig_stab.add_trace(go.Scatter(
                        x=top_names, y=top_means,
                        mode='markers',
                        name='Mean SHAP',
                        marker=dict(color='#00e896', size=9,
                                    line=dict(color='#020912', width=1.5)),
                    ))
                    fig_stab.update_layout(
                        title=dict(
                            text="SHAP Stability · 30-Bootstrap 95% Confidence Intervals",
                            font=dict(family='Syne, sans-serif', size=14, color='#c8d8e8'), x=0
                        ),
                        yaxis=dict(
                            title="Mean |SHAP| Value",
                            gridcolor='rgba(0,200,255,0.05)',
                            tickfont=dict(family='IBM Plex Mono', size=10, color='#3a7ca5'),
                            zeroline=False,
                        ),
                        xaxis=dict(tickfont=dict(family='IBM Plex Mono', size=10, color='#c8d8e8')),
                        plot_bgcolor='rgba(4,15,34,0)',
                        paper_bgcolor='rgba(4,15,34,0)',
                        barmode='overlay',
                        height=420,
                        legend=dict(font=dict(family='IBM Plex Mono', size=11, color='#8fb0cc'),
                                    bgcolor='rgba(0,0,0,0)'),
                        margin=dict(l=0, r=0, t=50, b=0),
                    )
                    st.plotly_chart(fig_stab, use_container_width=True)

                    avg_cv = (top_ci_width / (top_means + 1e-9)).mean()
                    if avg_cv < 0.3:
                        st.success("✅ **High Stability** — SHAP explanations are consistent across data samples")
                    elif avg_cv < 0.6:
                        st.info("📊 **Moderate Stability** — Minor variance across samples, explanations are reliable")
                    else:
                        st.warning("⚠️ **Low Stability** — High variance; consider more training data")

                    st.markdown(f"""
                    <div style="background:#071829;border:1px solid rgba(0,200,255,0.15);
                                border-radius:8px;padding:0.8rem;margin-top:0.5rem;">
                        <span style="color:#3a7ca5;font-family:'IBM Plex Mono',monospace;font-size:0.65rem;">
                        📖 HOW TO READ · Green dots = mean importance · Blue bars = 95% confidence interval<br>
                        Narrow bars = stable feature rankings · 
                        Avg coefficient of variation: <strong style="color:#00c8ff;">{avg_cv:.2f}</strong>
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Need at least 30 training samples for stability analysis.")
            else:
                st.info("Stability analysis requires a trained model with training data.")
        except Exception as e:
            st.info(f"Stability analysis unavailable: {str(e)[:80]}")

    with adv_tab4:
        st.markdown("""
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #3a7ca5; margin-bottom: 1rem;">
            <strong>🌐 Global Feature Importance (Bar Chart)</strong><br>
            Shows the average absolute SHAP values across all predictions - most important features have highest bars.
        </div>
        """, unsafe_allow_html=True)

        try:
            training_data = _load_once('training_data.pkl', 'training_data')
            xgb_models = _load_once('xgb_models.pkl', 'xgb_models')

            if training_data and symbol in training_data and xgb_models and symbol in xgb_models:
                stock_td = training_data[symbol]
                X_train_sample = stock_td.get('X_train')

                if X_train_sample is not None and len(X_train_sample) > 0:
                    if len(X_train_sample) > 200:
                        X_train_sample = X_train_sample[:200]

                    model = xgb_models[symbol]
                    base_model = model
                    if hasattr(model, 'calibrated_classifiers_'):
                        base_model = model.calibrated_classifiers_[0].estimator
                    elif hasattr(model, 'base_estimator_'):
                        base_model = model.base_estimator_

                    explainer = shap.TreeExplainer(base_model)
                    shap_values_sample = explainer.shap_values(X_train_sample)

                    if isinstance(shap_values_sample, list) and len(shap_values_sample) == 2:
                        shap_values_sample = shap_values_sample[1]

                    mean_shap = np.abs(shap_values_sample).mean(axis=0)
                    feature_names = stock_td.get('feature_names', [])

                    top_indices = np.argsort(mean_shap)[-15:][::-1]
                    top_features = [feature_names[i] for i in top_indices]
                    top_values = [mean_shap[i] for i in top_indices]

                    fig_importance = go.Figure()
                    fig_importance.add_trace(go.Bar(
                        x=top_values,
                        y=top_features,
                        orientation='h',
                        marker=dict(
                            color=top_values,
                            colorscale='Viridis',
                            colorbar=dict(title="Mean |SHAP|", title_font=dict(color='#8fb0cc')),
                            line=dict(width=0)
                        ),
                        text=[f"{val:.4f}" for val in top_values],
                        textposition='outside',
                        textfont=dict(family='IBM Plex Mono', size=10, color='#c8d8e8')
                    ))
                    fig_importance.update_layout(
                        title=dict(
                            text="Top 15 Features by Mean Absolute SHAP Value",
                            font=dict(family='Syne, sans-serif', size=14, color='#c8d8e8'),
                            x=0
                        ),
                        xaxis=dict(
                            title="Mean |SHAP| Value (Impact on Prediction)",
                            gridcolor='rgba(0,200,255,0.1)',
                            tickfont=dict(family='IBM Plex Mono', size=10, color='#8fb0cc')
                        ),
                        yaxis=dict(
                            tickfont=dict(family='IBM Plex Mono', size=10, color='#c8d8e8'),
                            gridcolor='rgba(0,200,255,0.1)'
                        ),
                        plot_bgcolor='rgba(4,15,34,0)',
                        paper_bgcolor='rgba(4,15,34,0)',
                        height=500,
                        margin=dict(l=150, r=50, t=60, b=40)
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)

                    st.markdown("""
                    <div style="background: #030e1e; padding: 0.6rem; border-radius: 6px; margin-top: 0.5rem;">
                        <span style="color: #3a7ca5; font-size: 0.65rem;">
                        📖 <strong>How to read:</strong> Longer bars indicate features that have greater influence on predictions.<br>
                        Higher values = more important features. These are the key drivers of your model's decisions.
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No training data available for this symbol.")
            else:
                st.info("Global feature importance requires trained XGBoost model with available training data.")
        except Exception as e:
            st.info(f"Feature importance plot currently unavailable. {str(e) if 'X_train' in str(e) else ''}")
# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="QuantEdge — Global Market Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════
# GLOBAL CSS — Premium Terminal-Finance Aesthetic
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, .stApp {
    background: #020912 !important;
    color: #c8d8e8 !important;
    font-family: 'Inter', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stToolbar"] { display: none; }

[data-testid="stSidebar"] {
    background: #030e1e !important;
    border-right: 1px solid rgba(0,200,255,0.12) !important;
}
[data-testid="stSidebar"] * { color: #8fb0cc !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #e0f0ff !important; }

.block-container { padding: 0 2rem 2rem 2rem !important; max-width: 100% !important; }

.qe-hero {
    position: relative; padding: 3rem 0 2rem 0;
    text-align: center; overflow: hidden;
}
.qe-hero::before {
    content: ''; position: absolute;
    top: -60px; left: 50%; transform: translateX(-50%);
    width: 700px; height: 300px;
    background: radial-gradient(ellipse, rgba(0,180,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.qe-wordmark {
    font-family: 'Syne', sans-serif; font-size: 4.2rem; font-weight: 800;
    letter-spacing: -2px;
    background: linear-gradient(135deg, #ffffff 0%, #7dd3fc 40%, #00e5ff 75%, #38bdf8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1; margin-bottom: 0.5rem;
}
.qe-tagline {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem;
    letter-spacing: 0.3em; color: #3a7ca5; text-transform: uppercase;
}

.ticker-band {
    background: linear-gradient(90deg, transparent, rgba(0,200,255,0.04), transparent);
    border-top: 1px solid rgba(0,200,255,0.1);
    border-bottom: 1px solid rgba(0,200,255,0.1);
    padding: 0.5rem 0; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem; color: #3a7ca5; letter-spacing: 0.15em;
    text-align: center; margin-bottom: 1.5rem;
}

.stock-header-card {
    background: linear-gradient(135deg, #040f22 0%, #071829 100%);
    border: 1px solid rgba(0,200,255,0.15); border-radius: 16px;
    padding: 1.6rem 2rem; display: flex; align-items: center;
    justify-content: space-between; margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
}
.stock-header-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #00c8ff, transparent);
}
.stock-symbol {
    font-family: 'Syne', sans-serif; font-size: 2.4rem;
    font-weight: 800; color: #e8f4ff; letter-spacing: -1px;
}
.stock-exchange-pill {
    display: inline-block; background: rgba(0,200,255,0.08);
    border: 1px solid rgba(0,200,255,0.2); border-radius: 20px;
    padding: 0.2rem 0.9rem; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem; color: #00c8ff; letter-spacing: 0.1em; margin-left: 0.8rem;
}
.stock-price-main {
    font-family: 'Syne', sans-serif; font-size: 2.4rem; font-weight: 700; color: #e8f4ff;
}
.change-up   { color: #00e896 !important; font-family:'IBM Plex Mono',monospace; font-size:0.95rem; }
.change-down { color: #ff4b6e !important; font-family:'IBM Plex Mono',monospace; font-size:0.95rem; }
.live-dot {
    display: inline-block; width: 7px; height: 7px; border-radius: 50%;
    background: #00e896; box-shadow: 0 0 8px #00e896; margin-right: 6px;
    animation: pulse 2s infinite;
}
.cached-dot {
    display: inline-block; width: 7px; height: 7px; border-radius: 50%;
    background: #f5a623; box-shadow: 0 0 8px #f5a623; margin-right: 6px;
}
@keyframes pulse {
    0%,100% { opacity:1; box-shadow:0 0 8px #00e896; }
    50%      { opacity:0.5; box-shadow:0 0 16px #00e896; }
}

.kpi-grid {
    display: grid; grid-template-columns: repeat(5, 1fr);
    gap: 1rem; margin-bottom: 1.5rem;
}
.kpi-card {
    background: #040f22; border: 1px solid rgba(0,200,255,0.1);
    border-radius: 12px; padding: 1.2rem 1.4rem;
    position: relative; overflow: hidden; transition: border-color 0.2s;
}
.kpi-card:hover { border-color: rgba(0,200,255,0.3); }
.kpi-card::after {
    content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 2px;
    background: var(--kpi-accent, rgba(0,200,255,0.3));
}
.kpi-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem;
    letter-spacing: 0.2em; color: #3a7ca5; text-transform: uppercase; margin-bottom: 0.5rem;
}
.kpi-value {
    font-family: 'Syne', sans-serif; font-size: 1.55rem;
    font-weight: 700; color: #e8f4ff; line-height: 1.1;
}
.kpi-sub { font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; margin-top: 0.3rem; }
.kpi-bull { color: #00e896; }
.kpi-bear { color: #ff4b6e; }
.kpi-neu  { color: #f5a623; }

.signal-badge {
    display: inline-block; padding: 0.25rem 0.9rem; border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem;
    font-weight: 600; letter-spacing: 0.08em;
}
.sig-strong-buy  { background:rgba(0,232,150,0.12);  color:#00e896; border:1px solid rgba(0,232,150,0.3); }
.sig-buy         { background:rgba(56,189,248,0.10); color:#38bdf8; border:1px solid rgba(56,189,248,0.3); }
.sig-hold        { background:rgba(245,166,35,0.10); color:#f5a623; border:1px solid rgba(245,166,35,0.3); }
.sig-sell        { background:rgba(255,75,110,0.10); color:#ff4b6e; border:1px solid rgba(255,75,110,0.3); }
.sig-strong-sell { background:rgba(180,30,60,0.15);  color:#ff2255; border:1px solid rgba(255,34,85,0.3); }
.sig-warn        { background:rgba(245,166,35,0.10); color:#f5a623; border:1px solid rgba(245,166,35,0.3); }

.section-title {
    font-family: 'Syne', sans-serif; font-size: 1.05rem; font-weight: 700;
    color: #c8d8e8; letter-spacing: 0.02em; margin-bottom: 1rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.section-title::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(0,200,255,0.2), transparent);
}

.chart-wrapper {
    background: #040f22; border: 1px solid rgba(0,200,255,0.1);
    border-radius: 16px; padding: 1.2rem; margin-bottom: 1.5rem;
}

.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(0,200,255,0.12) !important; gap: 0.25rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; border: none !important;
    color: #3a7ca5 !important; font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important; letter-spacing: 0.05em !important;
    padding: 0.6rem 1.2rem !important; border-radius: 8px 8px 0 0 !important;
    transition: all 0.2s !important;
}
.stTabs [data-baseweb="tab"]:hover { color:#7dd3fc !important; background:rgba(0,200,255,0.05) !important; }
.stTabs [aria-selected="true"] {
    color: #00c8ff !important; background: rgba(0,200,255,0.08) !important;
    border-bottom: 2px solid #00c8ff !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.2rem !important; }

[data-testid="metric-container"] {
    background: #040f22 !important; border: 1px solid rgba(0,200,255,0.1) !important;
    border-radius: 10px !important; padding: 1rem !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace !important; font-size: 0.68rem !important;
    letter-spacing: 0.15em !important; color: #3a7ca5 !important; text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important; font-size: 1.4rem !important;
    font-weight: 700 !important; color: #e8f4ff !important;
}
[data-testid="stMetricDelta"] { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.75rem !important; }

.stSelectbox > div > div {
    background: #040f22 !important; border: 1px solid rgba(0,200,255,0.2) !important;
    border-radius: 8px !important; color: #c8d8e8 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

.stDateInput > div > div {
    background: #040f22 !important; border: 1px solid rgba(0,200,255,0.2) !important;
    border-radius: 8px !important; color: #c8d8e8 !important;
    font-family: 'IBM Plex Mono', monospace !important; font-size: 0.82rem !important;
}

.streamlit-expanderHeader {
    background: #040f22 !important; border: 1px solid rgba(0,200,255,0.1) !important;
    border-radius: 8px !important; font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important; color: #3a7ca5 !important; letter-spacing: 0.08em !important;
}
.streamlit-expanderContent {
    background: #030e1e !important; border: 1px solid rgba(0,200,255,0.08) !important;
    border-top: none !important; border-radius: 0 0 8px 8px !important;
}

[data-testid="stLinkButton"] a {
    background: rgba(0,200,255,0.06) !important;
    border: 1px solid rgba(0,200,255,0.25) !important;
    border-radius: 6px !important; color: #00c8ff !important;
    font-family: 'IBM Plex Mono', monospace !important; font-size: 0.7rem !important;
    font-weight: 600 !important; letter-spacing: 0.1em !important;
    padding: 0.3rem 0.9rem !important; text-decoration: none !important;
    transition: background 0.2s, border-color 0.2s !important; display: inline-block !important;
}
[data-testid="stLinkButton"] a:hover {
    background: rgba(0,200,255,0.12) !important;
    border-color: rgba(0,200,255,0.5) !important; color: #7dd3fc !important;
}

.stAlert { border-radius: 8px !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.8rem !important; }
hr { border-color: rgba(0,200,255,0.1) !important; }

.sidebar-logo {
    font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 800;
    background: linear-gradient(135deg, #fff, #00c8ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0.1rem;
}
.sidebar-sub {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.6rem;
    letter-spacing: 0.2em; color: #3a7ca5; text-transform: uppercase; margin-bottom: 1rem;
}
.sidebar-section {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem;
    letter-spacing: 0.2em; color: #3a7ca5; text-transform: uppercase; margin: 1rem 0 0.4rem 0;
}
.symbol-chip {
    display: inline-block; background: rgba(0,200,255,0.07);
    border: 1px solid rgba(0,200,255,0.15); border-radius: 5px;
    padding: 0.15rem 0.5rem; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem; color: #5ba3cc; margin: 2px;
}
.phase-row {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem;
    color: #3a7ca5; padding: 0.2rem 0; display: flex; align-items: center; gap: 0.4rem;
}
.phase-done { color: #00e896; }
.phase-live { color: #f5a623; }

.stDataFrame { border: 1px solid rgba(0,200,255,0.1) !important; border-radius: 10px !important; overflow: hidden !important; }

.article-card {
    background: #040f22; border: 1px solid rgba(0,200,255,0.08);
    border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.3rem; transition: border-color 0.2s;
}
.article-card:hover { border-color: rgba(0,200,255,0.22); }
.article-title {
    font-family: 'Inter', sans-serif; font-size: 0.88rem;
    font-weight: 500; color: #c8d8e8; margin-bottom: 0.4rem; line-height: 1.4;
}
.article-meta { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; color: #3a7ca5; letter-spacing: 0.05em; }
.sentiment-pill {
    display: inline-block; padding: 0.1rem 0.6rem; border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; white-space: nowrap;
}

.model-metric-card {
    background: #040f22; border: 1px solid rgba(0,200,255,0.1);
    border-radius: 10px; padding: 1rem 1.2rem; text-align: center;
}
.model-metric-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; color: #3a7ca5;
    letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 0.3rem;
}
.model-metric-val { font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 700; color: #00c8ff; }

.glossary-table { width: 100%; border-collapse: collapse; }
.glossary-table th {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: #3a7ca5;
    letter-spacing: 0.1em; text-transform: uppercase;
    border-bottom: 1px solid rgba(0,200,255,0.12); padding: 0.4rem 0.6rem; text-align: left;
}
.glossary-table td {
    font-family: 'Inter', sans-serif; font-size: 0.78rem; color: #8fb0cc;
    padding: 0.5rem 0.6rem; border-bottom: 1px solid rgba(0,200,255,0.05);
}
.glossary-table tr:hover td { background: rgba(0,200,255,0.03); }

.qe-footer {
    text-align: center; padding: 2rem 0 1rem 0;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem;
    color: #1e3a52; letter-spacing: 0.1em;
    border-top: 1px solid rgba(0,200,255,0.06); margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────
import pandas as pd
@st.cache_data(ttl=3600)
def load_live_data():
    """Fetch fresh live data from yfinance - replaces json file"""
    
    SYMBOLS = [
        'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA',
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS'
    ]
    
    live_data = {}
    
    for symbol in SYMBOLS:
        try:
            ticker = yf.Ticker(symbol)
            hist   = ticker.history(period='2y')
            info   = {}
            
            try:
                info = ticker.info
            except:
                pass
            
            if hist.empty:
                continue
            
            # Build rows list
            rows = []
            for date, row in hist.iterrows():
                rows.append({
                    'date':   str(date)[:10],
                    'open':   float(row['Open']),
                    'high':   float(row['High']),
                    'low':    float(row['Low']),
                    'close':  float(row['Close']),
                    'volume': int(row['Volume']),
                })
            
            if len(rows) < 2:
                continue
            
            current_price = rows[-1]['close']
            prev_price    = rows[-2]['close']
            day_change    = (current_price - prev_price) / prev_price * 100
            
            # Compute technicals
            close = hist['Close']
            high  = hist['High']
            low   = hist['Low']
            vol   = hist['Volume']
            
            delta = close.diff()
            gain  = delta.where(delta > 0, 0).rolling(14).mean()
            loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rsi   = 100 - (100 / (1 + gain / loss))
            
            ema12       = close.ewm(span=12, adjust=False).mean()
            ema26       = close.ewm(span=26, adjust=False).mean()
            macd        = ema12 - ema26
            macd_signal = macd.ewm(span=9, adjust=False).mean()
            
            hl  = high - low
            hc  = (high - close.shift()).abs()
            lc  = (low  - close.shift()).abs()
            atr = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
            
            vol_ratio = vol / vol.rolling(20).mean()
            sma20     = close.rolling(20).mean()
            sma50     = close.rolling(50).mean()
            
            def safe(series):
                val = series.iloc[-1]
                return float(val) if not pd.isna(val) else None
            
            currency = '₹' if symbol.endswith('.NS') else '$'
            
            live_data[symbol] = {
                'symbol':               symbol,
                'exchange':             'NSE' if symbol.endswith('.NS') else 'NASDAQ',
                'currency':             currency,
                'collection_timestamp': datetime.now().isoformat(),
                'market_data': {
                    'company': {
                        'name':           info.get('longName', symbol),
                        'sector':         info.get('sector', 'Unknown'),
                        'industry':       info.get('industry', 'Unknown'),
                        'country':        info.get('country', 'Unknown'),
                        'market_cap':     float(info.get('marketCap', 0) or 0),
                        'pe_ratio':       float(info.get('trailingPE', 0) or 0),
                        'beta':           float(info.get('beta', 0) or 0),
                        'dividend_yield': float(info.get('dividendYield', 0) or 0),
                    },
                    'current': {
                        'current_price': current_price,
                        'day_change':    round(day_change, 4),
                    },
                    'technicals': {
                        'rsi_14':          safe(rsi),
                        'macd':            safe(macd),
                        'macd_signal':     safe(macd_signal),
                        'atr_14':          safe(atr),
                        'volume_ratio_20': safe(vol_ratio),
                        'sma_20':          safe(sma20),
                        'sma_50':          safe(sma50),
                    },
                    'full_dataframe': rows,
                    'data_points':    len(rows),
                    'date_range': {
                        'start': rows[0]['date'],
                        'end':   rows[-1]['date'],
                    },
                },
                'sentiment_data': {
                    'articles':       [],
                    'combined_score': 0,
                    'sentiment_label':'NEUTRAL',
                    'summary': {
                        'total_articles': 0,
                        'combined_score': 0,
                        'vader_avg':      0,
                        'alpha_avg':      0,
                        'sentiment':      'NEUTRAL',
                    }
                },
                'summary': {
                    'data_points':       len(rows),
                    'date_range':        {'start': rows[0]['date'], 'end': rows[-1]['date']},
                    'current_price':     current_price,
                    'currency':          currency,
                    'total_articles':    0,
                    'overall_sentiment': 'NEUTRAL',
                    'sentiment_score':   0,
                }
            }
            
        except Exception as e:
            st.warning(f"Could not load {symbol}: {e}")
            continue
    
    return live_data if live_data else None
@st.cache_data(ttl=3600)
def fetch_news_sentiment(symbol):
    try:
        try:
            api_key = st.secrets.get("ALPHA_VANTAGE_KEY", "")
        except:
            api_key = os.getenv("ALPHA_VANTAGE_KEY", "")

        if not api_key:
            return {}

        import requests
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        import nltk
        nltk.download('vader_lexicon', quiet=True)
        vader = SentimentIntensityAnalyzer()

        av_symbol = symbol.replace('.NS', '')

        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers':  av_symbol,
            'limit':    10,
            'apikey':   api_key,
        }
        r = requests.get(url, params=params, timeout=10)
        data = r.json()

        if 'feed' not in data:
            return {}

        articles = []
        scores   = []

        for item in data['feed'][:10]:
            title = item.get('title', '')
            score = vader.polarity_scores(title)['compound']
            scores.append(score)
            articles.append({
                'title':           title,
                'description':     item.get('summary', '')[:200],
                'source':          item.get('source', 'Unknown'),
                'published':       item.get('time_published', '')[:8],
                'url':             item.get('url', '#'),
                'sentiment_score': round(score, 4),
                'sentiment_label': 'Positive' if score >= 0.05 else 'Negative' if score <= -0.05 else 'Neutral',
            })

        avg_score = sum(scores) / len(scores) if scores else 0
        label = ('BULLISH' if avg_score >= 0.05
                 else 'BEARISH' if avg_score <= -0.05
                 else 'NEUTRAL')

        return {
            'articles':        articles,
            'combined_score':  round(avg_score, 4),
            'sentiment_label': label,
            'summary': {
                'total_articles': len(articles),
                'combined_score': round(avg_score, 4),
                'vader_avg':      round(avg_score, 4),
                'alpha_avg':      0.0,
                'sentiment':      label,
            }
        }
    except Exception:
        return {}
@st.cache_data
def load_data():
    possible_files = glob.glob("global_market_data_*.json")
    if possible_files:
        filename = sorted(possible_files)[-1]
        
    else:
        filename = 'global_market_data_latest.json'
        if not os.path.exists(filename):
            alt_files = glob.glob("*market_data*.json")
            if alt_files:
                filename = sorted(alt_files)[-1]
            else:
                st.error("❌ No data files found. Run final_data_collector.py first.")
                return None
    if not os.path.exists(filename):
        json_files = glob.glob("*.json")
        st.error(f"❌ {filename} not found!")
        if json_files:
            st.info(f"Available: {', '.join(json_files)}")
        return None
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        file_size = os.path.getsize(filename)
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def get_exchange(symbol):
    if str(symbol).endswith('.NS'):
        return 'NSE', '₹'
    return 'NASDAQ', '$'

def extract_stock_data(data, symbol):
    if symbol not in data:
        return None
    stock_data = data[symbol]
    result = {
        'symbol': symbol, 'exchange': get_exchange(symbol),
        'raw_data': stock_data, 'price_history': None,
        'company_info': {}, 'sentiment': {}, 'technicals': {},
        'current': {}, 'summary': stock_data.get('summary', {}),
    }
    def _parse_df(df_list):
        try:
            df = pd.DataFrame(df_list)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        except Exception:
            return None
    if 'market_data' in stock_data:
        market = stock_data['market_data']
        result['company_info'] = market.get('company', {})
        result['current']      = market.get('current', {})
        result['technicals']   = market.get('technicals', {})
        result['data_points']  = market.get('data_points', 0)
        result['date_range']   = market.get('date_range', {})
        if 'full_dataframe' in market:
            df = _parse_df(market['full_dataframe'])
            if df is not None:
                result['price_history'] = df
                result['data_points']   = len(df)
    elif 'price_history' in stock_data:
        result['company_info'] = stock_data.get('company', {})
        result['current']      = stock_data.get('current', {})
        result['technicals']   = stock_data.get('technicals', {})
        result['data_points']  = stock_data.get('data_points', 0)
        result['date_range']   = stock_data.get('date_range', {})
        if 'full_dataframe' in stock_data:
            df = _parse_df(stock_data['full_dataframe'])
            if df is not None:
                result['price_history'] = df
    result['sentiment'] = (
        stock_data.get('sentiment_data') or stock_data.get('sentiment') or {}
    )
    return result

# ─────────────────────────────────────────────────────────────────
# STALENESS BADGE & FORMAT DATE
# ─────────────────────────────────────────────────────────────────

def _staleness_badge(date_range: dict) -> str:
    end_str = date_range.get('end', '')
    if not end_str:
        return '<span class="cached-dot"></span><span style="color:#f5a623;font-family:\'IBM Plex Mono\',monospace;font-size:0.7rem;">CACHED</span>'
    try:
        latest_date = dateutil_parser.parse(end_str).date()
        delta_days  = (datetime.now().date() - latest_date).days
        if delta_days <= 3:
            return '<span class="live-dot"></span><span style="color:#00e896;font-family:\'IBM Plex Mono\',monospace;font-size:0.7rem;letter-spacing:0.1em;">LIVE</span>'
        return f'<span class="cached-dot"></span><span style="color:#f5a623;font-family:\'IBM Plex Mono\',monospace;font-size:0.7rem;">{delta_days}D AGO</span>'
    except Exception:
        return '<span class="cached-dot"></span><span style="color:#f5a623;font-family:\'IBM Plex Mono\',monospace;font-size:0.7rem;">CACHED</span>'

def format_date(date_str):
    if not date_str or date_str == 'Unknown':
        return '—'
    try:
        return dateutil_parser.parse(date_str).strftime("%d %b %Y")
    except Exception:
        return date_str

# ─────────────────────────────────────────────────────────────────
# SIGNAL HELPERS
# ─────────────────────────────────────────────────────────────────

def _signal_class(signal: str) -> str:
    s = signal.upper()
    if 'INSUFFICIENT'   in s: return 'sig-warn'
    if 'WEAK BUY'       in s: return 'sig-warn'
    if 'WEAK SELL'      in s: return 'sig-warn'
    if 'LOW CONFIDENCE' in s: return 'sig-warn'
    if 'STRONG BUY'     in s: return 'sig-strong-buy'
    if 'BUY'            in s: return 'sig-buy'
    if 'HOLD'           in s: return 'sig-hold'
    if 'STRONG SELL'    in s: return 'sig-strong-sell'
    if 'SELL'           in s: return 'sig-sell'
    return 'sig-hold'

def _signal_icon(signal: str) -> str:
    s = signal.upper()
    if 'INSUFFICIENT'   in s: return '✕'
    if 'WEAK BUY'       in s: return '△'
    if 'WEAK SELL'      in s: return '▽'
    if 'LOW CONFIDENCE' in s: return '⚠'
    if 'STRONG BUY'     in s: return '▲▲'
    if 'BUY'            in s: return '▲'
    if 'HOLD'           in s: return '◆'
    if 'STRONG SELL'    in s: return '▼▼'
    if 'SELL'           in s: return '▼'
    return '◆'

# ─────────────────────────────────────────────────────────────────
# CHART
# ─────────────────────────────────────────────────────────────────

def create_price_chart(df, symbol, exchange, currency, date_from=None, date_to=None):
    if df is None or df.empty:
        return None
    df_plot = df.copy()
    if date_from: df_plot = df_plot[df_plot.index.date >= date_from]
    if date_to:   df_plot = df_plot[df_plot.index.date <= date_to]
    if df_plot.empty: return None

    CHART_BG   = 'rgba(4,15,34,0)'
    GRID_COLOR = 'rgba(0,200,255,0.05)'
    TEXT_COLOR = '#3a7ca5'
    FONT       = 'IBM Plex Mono, monospace'
    MIN_VALID  = 20

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04, row_heights=[0.55, 0.22, 0.23])

    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot['open'], high=df_plot['high'],
        low=df_plot['low'],   close=df_plot['close'],
        name='Price', showlegend=False,
        increasing_line_color='#00e896', increasing_fillcolor='rgba(0,232,150,0.85)',
        decreasing_line_color='#ff4b6e', decreasing_fillcolor='rgba(255,75,110,0.85)',
    ), row=1, col=1)

    if 'sma_20' in df_plot.columns and df_plot['sma_20'].notna().sum() >= MIN_VALID:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['sma_20'],
            line=dict(color='#f5a623', width=1.5), name='SMA 20', opacity=0.9), row=1, col=1)

    if 'sma_50' in df_plot.columns and df_plot['sma_50'].notna().sum() >= MIN_VALID:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['sma_50'],
            line=dict(color='#b06eff', width=1.5), name='SMA 50', opacity=0.9), row=1, col=1)

    c_arr  = df_plot['close'].values
    o_arr  = df_plot['open'].values
    colors = ['#ff4b6e' if o_arr[i] > c_arr[i] else '#00e896' for i in range(len(c_arr))]
    fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['volume'],
        name='Volume', marker_color=colors, marker_opacity=0.55, showlegend=False), row=2, col=1)

    if 'rsi_14' in df_plot.columns and df_plot['rsi_14'].notna().sum() >= MIN_VALID:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['rsi_14'],
            line=dict(color='#00c8ff', width=2), name='RSI (14)',
            fill='tozeroy', fillcolor='rgba(0,200,255,0.04)'), row=3, col=1)
        fig.add_hline(y=70, line=dict(dash='dot', color='#ff4b6e', width=1),
                      annotation_text='OB', annotation_font_color='#ff4b6e',
                      annotation_font_size=9, row=3, col=1)
        fig.add_hline(y=30, line=dict(dash='dot', color='#00e896', width=1),
                      annotation_text='OS', annotation_font_color='#00e896',
                      annotation_font_size=9, row=3, col=1)

    axis_style = dict(gridcolor=GRID_COLOR, gridwidth=1, zeroline=False, color=TEXT_COLOR,
                      tickfont=dict(family=FONT, size=10, color=TEXT_COLOR), showline=False, showgrid=True)
    fig.update_layout(
        height=700, paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        hovermode='x unified', showlegend=True,
        margin=dict(l=0, r=16, t=10, b=0),
        legend=dict(orientation='h', yanchor='top', y=1.06, xanchor='left', x=0,
                    font=dict(family=FONT, size=10, color='#5ba3cc'),
                    bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)'),
        hoverlabel=dict(bgcolor='#071829', bordercolor='rgba(0,200,255,0.25)',
                        font=dict(family=FONT, size=11, color='#e8f4ff')),
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(axis_style)
    fig.update_yaxes(axis_style, row=1, col=1, tickprefix=currency, tickformat=',.2f')
    fig.update_yaxes(axis_style, row=2, col=1)
    fig.update_yaxes(axis_style, row=3, col=1, range=[0, 100])
    return fig

# ─────────────────────────────────────────────────────────────────
# SENTIMENT TAB
# ─────────────────────────────────────────────────────────────────

def display_sentiment_analysis(sentiment_data):
    if not sentiment_data:
        st.info("No sentiment data available.")
        return

    summary  = sentiment_data.get('summary', {})
    articles = sentiment_data.get('articles', [])
    combined = safe_float(summary.get('combined_score'))
    vader    = safe_float(summary.get('vader_avg'))
    alpha    = safe_float(summary.get('alpha_avg'))
    total    = summary.get('total_articles', 0)
    label    = summary.get('sentiment', 'NEUTRAL')

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        dc = "normal" if combined and combined >= 0 else "inverse"
        st.metric("Combined Score", _fmt(combined, '{:+.3f}'), label, delta_color=dc)
    with c2: st.metric("VADER Avg",   _fmt(vader, '{:+.3f}'))
    with c3: st.metric("FinBERT Avg", _fmt(alpha, '{:+.3f}'))
    with c4: st.metric("Articles Analyzed", total)

    st.markdown('<p class="section-title" style="margin-top:1.2rem;">📰 Recent News Coverage</p>', unsafe_allow_html=True)
    if not articles:
        st.info("No articles found.")
        return

    for article in articles[:10]:
        sentiment = safe_float(article.get('sentiment_score')) or 0.0
        if sentiment >= 0.05:
            pill_style = "background:rgba(0,232,150,0.12);color:#00e896;border:1px solid rgba(0,232,150,0.3);"
            pill_text  = f"+{sentiment:.3f} BULLISH"
        elif sentiment <= -0.05:
            pill_style = "background:rgba(255,75,110,0.12);color:#ff4b6e;border:1px solid rgba(255,75,110,0.3);"
            pill_text  = f"{sentiment:.3f} BEARISH"
        else:
            pill_style = "background:rgba(245,166,35,0.10);color:#f5a623;border:1px solid rgba(245,166,35,0.3);"
            pill_text  = f"{sentiment:+.3f} NEUTRAL"

        url   = article.get('url', '')
        title = article.get('title', 'No title')
        src   = article.get('source', 'Unknown')
        pub   = format_date(article.get('published', ''))

        st.markdown(f"""
        <div class="article-card">
            <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:1rem;margin-bottom:0.5rem;">
                <p class="article-title" style="margin:0;">{title}</p>
                <span class="sentiment-pill" style="{pill_style}">{pill_text}</span>
            </div>
            <div class="article-meta">
                <span>📰 {src}</span>&nbsp;&nbsp;·&nbsp;&nbsp;<span>📅 {pub}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if url and url != '#':
            st.link_button("Read Article ↗", url, use_container_width=False)

# ─────────────────────────────────────────────────────────────────
# COMPANY PROFILE TAB
# ─────────────────────────────────────────────────────────────────

def display_company_info(company_info, currency):
    if not company_info:
        st.info("No company information available.")
        return

    mc = safe_float(company_info.get('market_cap')) or 0.0
    if mc > 1e12:  cap_str = f"{currency}{mc/1e12:.2f}T"
    elif mc > 1e9: cap_str = f"{currency}{mc/1e9:.2f}B"
    elif mc > 1e6: cap_str = f"{currency}{mc/1e6:.2f}M"
    else:          cap_str = f"{currency}{mc:,.0f}" if mc > 0 else "—"

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p class="section-title">🏢 Company Profile</p>', unsafe_allow_html=True)
        for label, val in [("Full Name", company_info.get('name','—')),
                            ("Sector",    company_info.get('sector','—')),
                            ("Industry",  company_info.get('industry','—')),
                            ("Country",   company_info.get('country','—'))]:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid rgba(0,200,255,0.06);">
                <span style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#3a7ca5;letter-spacing:0.08em;">{label.upper()}</span>
                <span style="font-family:'Inter',sans-serif;font-size:0.82rem;color:#c8d8e8;">{val}</span>
            </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown('<p class="section-title">💰 Financial Snapshot</p>', unsafe_allow_html=True)
        for label, val in [("Market Cap",     cap_str),
                            ("P/E Ratio",      _fmt(safe_float(company_info.get('pe_ratio')), '{:.2f}×')),
                            ("Beta",           _fmt(safe_float(company_info.get('beta')), '{:.2f}')),
                            ("Dividend Yield", _fmt(safe_float(company_info.get('dividend_yield')), '{:.2f}%'))]:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid rgba(0,200,255,0.06);">
                <span style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#3a7ca5;letter-spacing:0.08em;">{label.upper()}</span>
                <span style="font-family:'Inter',sans-serif;font-size:0.82rem;color:#c8d8e8;">{val}</span>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS TAB
# ─────────────────────────────────────────────────────────────────

def display_technical_indicators(technicals, df):
    if not technicals and df is None:
        st.info("No technical data available.")
        return

    def get_val(key):
        raw = technicals.get(key)
        if raw is not None and raw != 'N/A':
            return safe_float(raw)
        if df is not None and key in df.columns:
            v = df[key].iloc[-1]
            return None if pd.isna(v) else safe_float(v)
        return None

    rsi       = get_val('rsi_14')
    macd      = get_val('macd')
    sig       = get_val('macd_signal')
    atr       = get_val('atr_14')
    vol_ratio = get_val('volume_ratio_20') or get_val('volume_ratio')

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if rsi is None:   st.metric("RSI (14)", "—")
        elif rsi > 70:    st.metric("RSI (14)", f"{rsi:.1f}", "Overbought", delta_color="inverse")
        elif rsi < 30:    st.metric("RSI (14)", f"{rsi:.1f}", "Oversold",   delta_color="normal")
        else:             st.metric("RSI (14)", f"{rsi:.1f}", "Neutral")
    with c2:
        if macd is not None and sig is not None:
            diff = macd - sig
            st.metric("MACD", f"{macd:.4f}", "Bullish" if diff > 0 else "Bearish",
                      delta_color="normal" if diff > 0 else "inverse")
        else:
            st.metric("MACD", "—")
    with c3:
        if atr is not None and df is not None and not df.empty:
            price = safe_float(df['close'].iloc[-1])
            if price and price > 0:
                st.metric("ATR (14)", f"{atr:.2f}", "High Vol" if (atr/price*100) > 5 else "Normal")
            else:
                st.metric("ATR (14)", f"{atr:.2f}" if atr else "—")
        else:
            st.metric("ATR (14)", "—")
    with c4:
        if vol_ratio is None:   st.metric("Volume Ratio", "—")
        elif vol_ratio > 1.5:   st.metric("Volume Ratio", f"{vol_ratio:.2f}×", "Unusual", delta_color="normal")
        elif vol_ratio < 0.5:   st.metric("Volume Ratio", f"{vol_ratio:.2f}×", "Weak",    delta_color="inverse")
        else:                   st.metric("Volume Ratio", f"{vol_ratio:.2f}×", "Normal")

    st.markdown('<p class="section-title" style="margin-top:1.5rem;">📐 Moving Averages</p>', unsafe_allow_html=True)
    ma_cols = st.columns(4)
    for col, key, label in zip(ma_cols, ['sma_5','sma_10','sma_20','sma_50'],
                                        ['SMA 5','SMA 10','SMA 20','SMA 50']):
        with col: st.metric(label, _fmt(get_val(key)))

    if df is not None and not df.empty:
        close_price = safe_float(df['close'].iloc[-1])
        comp_cols   = st.columns(2)
        sma20 = get_val('sma_20')
        sma50 = get_val('sma_50')
        with comp_cols[0]:
            if sma20 and close_price and sma20 > 0:
                st.metric("vs SMA 20", f"{(close_price-sma20)/sma20*100:+.2f}%")
        with comp_cols[1]:
            if sma50 and close_price and sma50 > 0:
                st.metric("vs SMA 50", f"{(close_price-sma50)/sma50*100:+.2f}%")

    st.markdown('<p class="section-title" style="margin-top:1.5rem;">📖 Indicator Reference</p>', unsafe_allow_html=True)
    st.markdown("""
    <table class="glossary-table">
        <thead><tr><th>Metric</th><th>What It Measures</th><th>Interpretation</th></tr>
        </thead>
        <tbody>
             <tr><td><strong>RSI (14)</strong></td><td>Momentum oscillator 0–100</td><td>&lt;30 oversold · &gt;70 overbought</td></tr>
             <tr><td><strong>MACD</strong></td><td>Trend &amp; momentum divergence</td><td>MACD &gt; Signal line = bullish crossover</td></tr>
             <tr><td><strong>ATR (14)</strong></td><td>Average daily trading range</td><td>&gt;5% of price = elevated volatility</td></tr>
             <tr><td><strong>Volume Ratio</strong></td><td>Volume ÷ 20-day average</td><td>&gt;1.5× = unusual market activity</td></tr>
             <tr><td><strong>SMA 20 / 50</strong></td><td>Smoothed price trend line</td><td>Price above SMA = uptrend confirmed</td></tr>
             <tr><td><strong>Sentiment</strong></td><td>News tone −1 to +1</td><td>&gt;0.05 bullish · &lt;−0.05 bearish</td></tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# AI MODEL PERFORMANCE TAB
# ─────────────────────────────────────────────────────────────────

def display_model_performance(model_results, selected_symbol):
    if not model_results or selected_symbol not in model_results:
        st.info("Run improved_stacking.py first to see AI model performance metrics.")
        return

    pred = model_results[selected_symbol]
    auc  = pred.get('best_auc', 0.5)

    if auc < 0.50:
        st.warning(f"⚠️ Model performs below random chance (AUC = {auc:.3f}) — predictions suppressed")
    elif auc < 0.55:
        st.info(f"📊 Weak predictive power detected (AUC = {auc:.3f}) — signals shown with caution")

    methods = [
        ("Best Individual",   pred.get('best_individual', 0.5)),
        ("Weighted Ensemble", pred.get('weighted', 0.5)),
        ("Selective Method",  pred.get('selective', 0.5)),
        ("Best Method",       pred.get('best_method', 0.5)),
    ]
    cols = st.columns(4)
    for col, (label, val) in zip(cols, methods):
        with col:
            col_color = "#00e896" if val > 0.6 else "#f5a623" if val > 0.5 else "#ff4b6e"
            st.markdown(f"""
            <div class="model-metric-card">
                <div class="model-metric-label">{label}</div>
                <div class="model-metric-val" style="color:{col_color};">{val:.1%}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    fig = go.Figure(data=[go.Bar(
        x=['Best Individual','Weighted Ensemble','Selective Method','Best Method'],
        y=[pred.get('best_individual',0.5)*100, pred.get('weighted',0.5)*100,
           pred.get('selective',0.5)*100,        pred.get('best_method',0.5)*100],
        marker=dict(color=['rgba(122,154,184,0.8)','rgba(176,110,255,0.8)',
                           'rgba(0,200,255,0.8)','rgba(0,232,150,0.8)'], line=dict(width=0)),
        text=[f"{v:.1%}" for v in [pred.get('best_individual',0.5), pred.get('weighted',0.5),
                                    pred.get('selective',0.5), pred.get('best_method',0.5)]],
        textposition='outside',
        textfont=dict(family='IBM Plex Mono, monospace', size=11, color='#8fb0cc'),
    )])
    fig.update_layout(
        title=dict(text="Model Accuracy Comparison", font=dict(family='Syne, sans-serif', size=14, color='#c8d8e8'), x=0),
        yaxis=dict(title="Accuracy (%)", range=[0,105], gridcolor='rgba(0,200,255,0.05)',
                   tickfont=dict(family='IBM Plex Mono', size=10, color='#3a7ca5'), color='#3a7ca5', zeroline=False),
        xaxis=dict(tickfont=dict(family='IBM Plex Mono', size=10, color='#3a7ca5'), color='#3a7ca5'),
        plot_bgcolor='rgba(4,15,34,0)', paper_bgcolor='rgba(4,15,34,0)',
        font=dict(color='#7a9ab8'), height=360, margin=dict(l=0, r=0, t=40, b=0), bargap=0.35,
    )
    st.plotly_chart(fig, use_container_width=True)

    xgb_auc  = pred.get('xgb_auc',      0.5)
    rf_auc   = pred.get('rf_auc',       0.5)
    lstm_auc = pred.get('lstm_auc',     0.5)
    ens_auc  = pred.get('ensemble_auc', 0.5)

    if any(v != 0.5 for v in [xgb_auc, rf_auc, lstm_auc]):
        st.markdown('<p class="section-title" style="margin-top:1rem;">📊 AUC by Model</p>', unsafe_allow_html=True)
        a1, a2, a3, a4 = st.columns(4)
        with a1: st.metric("XGBoost AUC",       f"{xgb_auc:.3f}")
        with a2: st.metric("Random Forest AUC", f"{rf_auc:.3f}")
        with a3: st.metric("LSTM AUC",          f"{lstm_auc:.3f}")
        with a4: st.metric("Ensemble AUC",      f"{ens_auc:.3f}")
        st.success(f"✅ Best model: **{pred.get('best_model','XGBoost')}** · AUC = {auc:.3f}")

    st.markdown('<p class="section-title" style="margin-top:1.2rem;">🏆 Top Performing Stocks</p>', unsafe_allow_html=True)
    top = pd.DataFrame([
        {'Stock': s, 'Accuracy': d.get('best_method',0.5)*100,
         'AUC': d.get('best_auc',0.5), 'Model': d.get('best_model','—')}
        for s, d in model_results.items()
    ]).sort_values('AUC', ascending=False).head(5)
    top['Signal'] = top['AUC'].apply(
        lambda x: 'STRONG BUY' if x >= 0.65 else 'BUY' if x >= 0.58 else 'HOLD' if x >= 0.55 else '⚠ LOW CONF')
    st.dataframe(top, column_config={
        "Stock":    "Symbol",
        "Accuracy": st.column_config.ProgressColumn("Model Accuracy", format="%.1f%%", min_value=0, max_value=100),
        "AUC":      st.column_config.NumberColumn("AUC Score", format="%.3f"),
        "Model":    "Best Model",
        "Signal":   "Signal",
    }, hide_index=True, use_container_width=True)

    conf_signals = pred.get('confident_signals', 0)
    conf_acc     = pred.get('confident_acc', 0.0)
    if conf_signals > 0:
        st.info(f"🎯 High-confidence signals: **{conf_signals}** trades · accuracy when confident: **{conf_acc:.1%}**")

# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():

    # ── SIDEBAR ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="sidebar-logo">QuantEdge</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-sub">Global Market Intelligence</div>', unsafe_allow_html=True)

        with st.spinner('⚡ Fetching live market data...'):
            data = load_live_data()
            if not data:
                st.error("Could not fetch live market data. Please refresh.")
                return
        model_results = load_model_results()
     

        symbols   = list(data.keys())
        us_stocks = [s for s in symbols if not s.endswith('.NS')]
        in_stocks = [s for s in symbols if s.endswith('.NS')]

        st.markdown('<div class="sidebar-section">US Equities</div>', unsafe_allow_html=True)
        st.markdown(' '.join([f'<span class="symbol-chip">{s}</span>' for s in us_stocks]), unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section">Indian Equities (NSE)</div>', unsafe_allow_html=True)
        st.markdown(' '.join([f'<span class="symbol-chip">{s.replace(".NS","")}</span>' for s in in_stocks]), unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">Select Instrument</div>', unsafe_allow_html=True)
        selected_symbol = st.selectbox("Stock", symbols, index=0, label_visibility="collapsed")

        

        st.markdown('<div class="sidebar-section">Research Team</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:#3a7ca5;line-height:1.8;">
            Amritha K · 22BAI1318<br>Garima Mishra · 22BPS1153<br> VIT Chennai<br>
           
            
            
        </div>""", unsafe_allow_html=True)

    # ── HERO ─────────────────────────────────────────────────────
    st.markdown("""
    <div class="qe-hero">
        <div class="qe-wordmark">QuantEdge</div>
        <div class="qe-tagline">Multi-Horizon Technical &amp; Sentiment-Driven Signal System </div>
    </div>
    
    """, unsafe_allow_html=True)

    # ── EXTRACT DATA ─────────────────────────────────────────────
    stock_data = extract_stock_data(data, selected_symbol)
    if not stock_data:
        st.error(f"Could not extract data for {selected_symbol}")
        return

    exchange, currency = stock_data['exchange']
    current_price = safe_float(stock_data['current'].get('current_price'))
    day_change    = safe_float(stock_data['current'].get('day_change'))
    rsi_val       = safe_float(stock_data['technicals'].get('rsi_14'))
    sent_score    = safe_float(stock_data['sentiment'].get('combined_score'))
    vol_ratio     = safe_float(
        stock_data['technicals'].get('volume_ratio_20') or
        stock_data['technicals'].get('volume_ratio')
    )
    date_range = stock_data.get('date_range', {})

    dc         = day_change or 0.0
    price_str  = f"{currency}{current_price:,.2f}" if current_price is not None else "—"
    safe_sym   = str(selected_symbol).replace('.NS', '')
    change_cls = "change-up" if dc >= 0 else "change-down"
    change_sgn = "▲" if dc >= 0 else "▼"

    # ── STOCK HEADER ─────────────────────────────────────────────
    st.markdown(f"""
    <div class="stock-header-card">
        <div style="display:flex;align-items:center;gap:0.6rem;">
            <span class="stock-symbol">{safe_sym}</span>
            <span class="stock-exchange-pill">● {exchange}</span>
            {_staleness_badge(date_range)}
        </div>
        <div style="display:flex;align-items:center;gap:1.2rem;">
            <span class="stock-price-main">{price_str}</span>
            <span class="{change_cls}">{change_sgn} {abs(dc):.2f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── LIVE PREDICTION ──────────────────────────────────────────
    model_results = load_model_results()
    live_pred     = get_live_prediction(selected_symbol)

    if live_pred:
        pred_signal = live_pred['signal']
        pred_conf   = live_pred['confidence']
        model_used  = live_pred.get('model_used', 'live model')
        is_gated    = live_pred.get('auc_gated', False)
        if is_gated:
            reason   = live_pred.get('auc_gated_reason', '')
            pred_sub = reason if reason else f"AUC {pred_conf:.3f} · below reliability threshold"
            pred_agree = ""
        elif pred_conf >= 0.58:
            pred_sub = f"{pred_conf*100:.0f}% UP probability · {model_used.split('(')[0].strip()}"
        elif pred_conf <= 0.42:
            pred_sub = f"{(1-pred_conf)*100:.0f}% DOWN probability · {model_used.split('(')[0].strip()}"
        else:
            pred_sub = f"{pred_conf*100:.0f}% UP · {(1-pred_conf)*100:.0f}% DOWN · {model_used.split('(')[0].strip()}"
        agreement   = live_pred.get('agreement', {})
        agree_label = agreement.get('label', '')
        if agree_label and agree_label != 'Unknown' and not is_gated:
            pred_agree = f"Model agreement: {agree_label}"
        else:
            pred_agree = ""
    else:
        prediction  = (model_results or {}).get(selected_symbol)
        pred_conf   = prediction.get('best_method', 0.5) if prediction else 0.5
        pred_signal = 'STRONG BUY' if pred_conf >= 0.60 else 'HOLD' if pred_conf >= 0.50 else 'SELL'
        pred_sub    = f"{pred_conf:.1%} accuracy"
        is_gated    = False
        pred_agree  = ""

    sig_cls  = _signal_class(pred_signal)
    sig_icon = _signal_icon(pred_signal)

    # ── KPI LABELS ───────────────────────────────────────────────
    if rsi_val is None:
        rsi_str, rsi_cls, rsi_lbl = "—", "kpi-neu", "—"
    elif rsi_val > 70:
        rsi_str, rsi_cls, rsi_lbl = f"{rsi_val:.1f}", "kpi-bear", "Overbought"
    elif rsi_val < 30:
        rsi_str, rsi_cls, rsi_lbl = f"{rsi_val:.1f}", "kpi-bull", "Oversold"
    else:
        rsi_str, rsi_cls, rsi_lbl = f"{rsi_val:.1f}", "kpi-neu", "Neutral"

    if sent_score is None:
        sent_str, sent_cls, sent_lbl = "—", "kpi-neu", "—"
    elif sent_score >= 0.05:
        sent_str, sent_cls, sent_lbl = f"{sent_score:+.3f}", "kpi-bull", "Bullish"
    elif sent_score <= -0.05:
        sent_str, sent_cls, sent_lbl = f"{sent_score:+.3f}", "kpi-bear", "Bearish"
    else:
        sent_str, sent_cls, sent_lbl = f"{sent_score:+.3f}", "kpi-neu", "Neutral"

    vol_str = f"{vol_ratio:.2f}×" if vol_ratio else "—"
    vol_lbl = ("Unusual" if vol_ratio and vol_ratio > 1.5 else
               "Low"     if vol_ratio and vol_ratio < 0.5 else "Normal") if vol_ratio else "—"

    price_cls   = "kpi-bull" if dc >= 0 else "kpi-bear"
    rsi_accent  = '#ff4b6e' if rsi_val and rsi_val > 70 else '#00e896' if rsi_val and rsi_val < 30 else '#f5a623'
    sent_accent = '#00e896' if sent_score and sent_score > 0.05 else '#ff4b6e' if sent_score and sent_score < -0.05 else '#f5a623'

    # ── KPI CARDS ────────────────────────────────────────────────
    # ── KPI CARDS ────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""
            <div style="background:#040f22;border:1px solid rgba(0,200,255,0.1);border-radius:12px;padding:1rem;text-align:center">
                <div style="color:#3a7ca5;font-size:0.7rem;letter-spacing:2px">
                    AI SIGNAL · NEXT-DAY
                </div>
                <div style="color:#00e896;font-size:1.2rem;font-weight:bold;margin:0.5rem 0">
                    {sig_icon} {pred_signal}
                </div>
                <div style="color:#3a7ca5;font-size:0.7rem">
                    {pred_sub}
                </div>
                <div style="color:#f5a623;font-size:0.6rem">
                    {pred_agree}
                
            
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div style="background:#040f22;border:1px solid rgba(0,200,255,0.1);border-radius:12px;padding:1rem;text-align:center">
                <div style="color:#3a7ca5;font-size:0.7rem;letter-spacing:2px">
                    RSI (14)
                </div>
                <div style="color:#e8f4ff;font-size:1.8rem;font-weight:bold;margin:0.5rem 0">
                    {rsi_str}
                </div>
                <div style="color:#f5a623;font-size:0.7rem">
                    {rsi_lbl}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""
            <div style="background:#040f22;border:1px solid rgba(0,200,255,0.1);border-radius:12px;padding:1rem;text-align:center">
                <div style="color:#3a7ca5;font-size:0.7rem;letter-spacing:2px">
                    SENTIMENT
                </div>
                <div style="color:#00e896;font-size:1.8rem;font-weight:bold;margin:0.5rem 0">
                    {sent_str}
                </div>
                <div style="color:#00e896;font-size:0.7rem">
                    {sent_lbl}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            f"""
            <div style="background:#040f22;border:1px solid rgba(0,200,255,0.1);border-radius:12px;padding:1rem;text-align:center">
                <div style="color:#3a7ca5;font-size:0.7rem;letter-spacing:2px">
                    VOLUME RATIO
                </div>
                <div style="color:#e8f4ff;font-size:1.8rem;font-weight:bold;margin:0.5rem 0">
                    {vol_str}
                </div>
                <div style="color:#f5a623;font-size:0.7rem">
                    {vol_lbl}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    # ── TECHNICAL BREAKDOWN + SHAP EXPANDER ──────────────────────
    if live_pred and not live_pred.get('auc_gated'):
        agreement        = live_pred.get('agreement', {})
        agree_type       = agreement.get('type', '')
        agree_lbl        = agreement.get('label', 'Unknown')
        model_name_display = live_pred.get('model_used', '').split('(')[0].strip()

        with st.expander("🔍  Explain This Signal  ·  click to expand"):

            # ── Model confidence header ───────────────────────────
            st.markdown(f"""
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;
                        color:#8fb0cc;line-height:1.8;padding:0.5rem;">
                <span style="color:#00c8ff;font-size:0.85rem;font-weight:600;">
                📊 MODEL CONFIDENCE BREAKDOWN</span><br><br>
                <span style="color:#3a7ca5;">Model Used:</span>
                &nbsp;&nbsp;&nbsp;&nbsp;{model_name_display}<br>
                <span style="color:#3a7ca5;">Signal:</span>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{pred_signal}<br>
                <span style="color:#3a7ca5;">Direction:</span>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{pred_sub}<br>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            if agree_type == 'rf':
                n_trees    = agreement.get('n_trees', 500)
                buy_votes  = agreement.get('buy_votes', 0)
                sell_votes = agreement.get('sell_votes', 0)
                pct        = agreement.get('pct', 0.5)
                dominant_votes = buy_votes if pct >= 0.5 else sell_votes
                minority_votes = sell_votes if pct >= 0.5 else buy_votes
                dominant_dir   = "BUY" if pct >= 0.5 else "SELL"
                minority_dir   = "SELL" if pct >= 0.5 else "BUY"
                st.markdown(f"""
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;
                            color:#8fb0cc;line-height:2;padding:0.5rem;">
                    <span style="color:#00c8ff;">HOW THE MODEL DECIDED:</span><br>
                    {n_trees} independent decision trees each analysed<br>
                    today's 93 market indicators and cast one vote.<br><br>
                    <span style="color:#00e896;">✅ {dominant_votes} trees voted → {dominant_dir}</span><br>
                    <span style="color:#ff4b6e;">❌ {minority_votes} trees voted → {minority_dir}</span><br><br>
                    <span style="color:#3a7ca5;">Agreement:</span>
                    &nbsp;{pct*100:.0f}% · <span style="color:#f5a623;">{agree_lbl} confidence</span><br><br>
                    <span style="color:#1e3a52;font-size:0.7rem;">
                    The more trees that agree, the more reliable the signal.</span>
                </div>
                """, unsafe_allow_html=True)

            elif agree_type == 'xgb':
                margin = agreement.get('margin', 0)
                direction_word = "BUY" if pred_conf >= 0.5 else "SELL"
                sign = "+" if pred_conf >= 0.5 else "-"
                st.markdown(f"""
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;
                            color:#8fb0cc;line-height:2;padding:0.5rem;">
                    <span style="color:#00c8ff;">HOW THE MODEL DECIDED:</span><br>
                    XGBoost produces a raw score measuring how strongly<br>
                    it leans toward BUY or SELL before converting to %.<br><br>
                    <span style="color:#3a7ca5;">Raw score:</span>
                    &nbsp;&nbsp;&nbsp;{sign}{margin:.3f}<br>
                    <span style="color:#3a7ca5;">Neutral point:</span>
                    &nbsp;0.000<br>
                    <span style="color:#3a7ca5;">Distance:</span>
                    &nbsp;&nbsp;&nbsp;&nbsp;{margin:.3f} into {direction_word} territory<br><br>
                    <span style="color:#3a7ca5;">Agreement:</span>
                    &nbsp;<span style="color:#f5a623;">{agree_lbl} confidence</span>
                    &nbsp;(threshold: &gt;1.5 = High)<br><br>
                    <span style="color:#1e3a52;font-size:0.7rem;">
                    The further from zero, the stronger the signal conviction.</span>
                </div>
                """, unsafe_allow_html=True)

            elif agree_type == 'lstm':
                margin = agreement.get('margin', 0)
                output = agreement.get('output', pred_conf)
                st.markdown(f"""
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;
                            color:#8fb0cc;line-height:2;padding:0.5rem;">
                    <span style="color:#00c8ff;">HOW THE MODEL DECIDED:</span><br>
                    LSTM analysed the last 20 trading days as a sequence<br>
                    to detect patterns over time.<br><br>
                    <span style="color:#3a7ca5;">Output probability:</span>
                    &nbsp;&nbsp;{output:.3f}<br>
                    <span style="color:#3a7ca5;">Neutral midpoint:</span>
                    &nbsp;&nbsp;&nbsp;0.500<br>
                    <span style="color:#3a7ca5;">Distance from 50/50:</span>
                    &nbsp;{margin:.3f} ({margin*100:.0f}%)<br><br>
                    <span style="color:#3a7ca5;">Agreement:</span>
                    &nbsp;<span style="color:#f5a623;">{agree_lbl} confidence</span>
                    &nbsp;(needs &gt;8% for BUY/SELL)<br><br>
                    <span style="color:#1e3a52;font-size:0.7rem;">
                    The further from 0.5, the stronger the model conviction.</span>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.info("Technical breakdown not available for this signal type.")

            # ══════════════════════════════════════════════════════
            # REAL SHAP SECTION
            # ══════════════════════════════════════════════════════
            st.markdown("---")
            st.markdown("""
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;
                        color:#8fb0cc;line-height:1.6;padding:0.5rem;">
                <span style="color:#00c8ff;font-size:0.85rem;font-weight:600;">
                🔎 TOP FEATURES DRIVING THIS SIGNAL</span><br>
                <span style="color:#3a7ca5;font-size:0.7rem;">
                SHAP values · ranked by impact · TreeExplainer (XGB/RF) · GradientExplainer (LSTM)</span><br><br>
            </div>
            """, unsafe_allow_html=True)

            shap_features = get_shap_explanation(selected_symbol, live_pred)
            try:
                pass  # shap_features already computed above
            except Exception as e:
                shap_features = []
                st.sidebar.error(f"SHAP error: {str(e)[:200]}")

            st.sidebar.write("--- SHAP DEBUG ---")
            st.sidebar.write(f"SHAP_AVAILABLE: {SHAP_AVAILABLE}")
            st.sidebar.write(f"SHAP features: {len(shap_features) if shap_features else 'NONE'}")
            training_data_check = _load_once('training_data.pkl', 'training_data')
            xgb_check = _load_once('xgb_models.pkl', 'xgb_models')
            st.sidebar.write(f"training_data loaded: {training_data_check is not None}")
            st.sidebar.write(f"xgb_models loaded: {xgb_check is not None}")
            st.sidebar.write(f"SHAP internal: {st.session_state.get('shap_debug', 'not set')}")
            st.sidebar.write(f"SHAP debug2: {st.session_state.get('shap_debug2', 'not set')}")
            if training_data_check and selected_symbol in training_data_check:
                st.sidebar.write(f"Symbol in training_data: True")
            else:
                st.sidebar.write(f"Symbol in training_data: FALSE")
            st.sidebar.write("--- END DEBUG ---")

            if shap_features:
                max_abs = max(abs(f['shap']) for f in shap_features) or 1.0

                # ── Plain English Summary Card ─────────────────
                summary_text = generate_plain_english_summary(shap_features, pred_signal)
                if summary_text:
                    st.markdown(f"""
                    <div style="background: rgba(0,200,255,0.05);
                                border-left: 3px solid #00c8ff;
                                border-radius: 0 8px 8px 0;
                                padding: 0.9rem 1.1rem;
                                margin-bottom: 1.2rem;">
                        <div style="font-family:'IBM Plex Mono',monospace;
                                    font-size:0.62rem; color:#3a7ca5;
                                    letter-spacing:0.12em; margin-bottom:0.4rem;">
                            💬 IN PLAIN ENGLISH
                        </div>
                        <div style="font-family:'Inter',sans-serif;
                                    font-size:0.88rem; color:#c8d8e8; line-height:1.6;">
                            {summary_text}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Feature Bars with Plain Labels + Descriptions ──
                rows_html = ""
                for feat in shap_features:
                    short_label, description = get_feature_plain(feat['name'])
                    bar_color = "#00e896" if feat['direction'] == "BUY" else "#ff4b6e"
                    bar_width = int(abs(feat['shap']) / max_abs * 120)
                    val_str   = f"{feat['shap']:+.4f}"
                    rows_html += (
                        f'<div style="display:flex;align-items:flex-start;gap:12px;margin-bottom:14px;padding-bottom:12px;border-bottom:1px solid rgba(0,200,255,0.06);">'
                        f'<div style="width:170px;flex-shrink:0;">'
                        f'<div style="color:#c5dff0;font-family:IBM Plex Mono,monospace;font-size:0.76rem;font-weight:500;">{short_label}</div>'
                        f'<div style="color:#3a7ca5;font-family:Inter,sans-serif;font-size:0.63rem;line-height:1.35;margin-top:3px;">{description}</div>'
                        f'</div>'
                        f'<div style="display:flex;align-items:center;gap:8px;margin-top:4px;flex:1;">'
                        f'<div style="background:{bar_color};height:8px;width:{bar_width}px;border-radius:4px;opacity:0.85;flex-shrink:0;"></div>'
                        f'<span style="color:{bar_color};font-family:IBM Plex Mono,monospace;font-size:0.76rem;min-width:60px;">{val_str}</span>'
                        f'<span style="color:#3a7ca5;font-size:0.68rem;">&#8594; {feat["direction"]}</span>'
                        f'</div>'
                        f'</div>'
                    )
                
                
                st.markdown(
                    f'<div style="padding:0.5rem 0.2rem;">{rows_html}</div>'
                    f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.63rem;'
                    f'color:#3a7ca5;margin-top:0.4rem;padding:0.3rem;line-height:1.6;">'
                    f'Green bars pushed the signal toward BUY &nbsp;·&nbsp; '
                    f'Red bars pushed toward SELL &nbsp;·&nbsp; '
                    f'Longer bar = stronger influence on this prediction.</div>',
                    unsafe_allow_html=True
                )
                
                # Plotly bar chart
                names  = [f['name'] for f in shap_features]
                values = [f['shap'] for f in shap_features]
                colors = ["rgba(0,232,150,0.8)" if v > 0 else "rgba(255,75,110,0.8)"
                          for v in values]
                fig_shap = go.Figure(go.Bar(
                    x=names, y=values,
                    marker_color=colors, marker_line_width=0,
                    text=[f"{v:+.4f}" for v in values],
                    textposition='outside',
                    textfont=dict(family='IBM Plex Mono, monospace', size=10, color='#8fb0cc'),
                ))
                fig_shap.add_hline(
                    y=0,
                    line=dict(color='rgba(0,200,255,0.25)', width=1, dash='dot')
                )
                fig_shap.update_layout(
                    title=dict(
                        text="SHAP Feature Importance · Top 5 Features",
                        font=dict(family='Syne, sans-serif', size=13, color='#c8d8e8'), x=0
                    ),
                    yaxis=dict(
                        title="SHAP value  (contribution to UP probability)",
                        gridcolor='rgba(0,200,255,0.05)',
                        zeroline=False,
                        tickfont=dict(family='IBM Plex Mono', size=9, color='#3a7ca5'),
                        color='#3a7ca5',
                    ),
                    xaxis=dict(
                        tickfont=dict(family='IBM Plex Mono', size=10, color='#c8d8e8')
                    ),
                    plot_bgcolor='rgba(4,15,34,0)', paper_bgcolor='rgba(4,15,34,0)',
                    font=dict(color='#7a9ab8'), height=290,
                    margin=dict(l=0, r=0, t=40, b=0), bargap=0.4,
                )
                st.plotly_chart(fig_shap, use_container_width=True)

            else:
                # Fallback
                st.markdown(
                    '<div style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;'
                    'color:#f5a623;padding:0.3rem 0.5rem;margin-bottom:0.5rem;">'
                    '⚠ Live SHAP unavailable (pkl files not found or model not loaded). '
                    'Showing illustrative values below.</div>',
                    unsafe_allow_html=True
                )
                _shap_mock = {
                    'XGBoost': [
                        ("RSI_14",          -0.12, "SELL"),
                        ("MACD_diff",       -0.09, "SELL"),
                        ("volume_ratio",    -0.07, "SELL"),
                        ("EMA_cross",       +0.04, "BUY"),
                        ("ATR_14",          -0.03, "SELL"),
                    ],
                    'Random Forest': [
                        ("SMA_20_50_cross", +0.11, "BUY"),
                        ("RSI_14",          +0.08, "BUY"),
                        ("OBV_trend",       +0.06, "BUY"),
                        ("MACD_signal",     -0.04, "SELL"),
                        ("Bollinger_pct",   +0.03, "BUY"),
                    ],
                    'LSTM': [
                        ("RSI_14",          -0.10, "SELL"),
                        ("EMA_20",          -0.08, "SELL"),
                        ("MACD_diff",       +0.05, "BUY"),
                        ("ATR_14",          -0.04, "SELL"),
                        ("ADX_14",          +0.02, "BUY"),
                    ],
                }
                _display_model = live_pred.get('model_used', 'XGBoost').split()[0] if live_pred else 'XGBoost'
                if _display_model not in _shap_mock:
                    _display_model = 'XGBoost'
                _features = _shap_mock[_display_model]
                rows_html = ""
                for fname, val, direction in _features:
                    bar_color = "#00e896" if direction == "BUY" else "#ff4b6e"
                    bar_width = int(abs(val) * 600)
                    rows_html += (
                        f'<div style="display:flex;align-items:center;gap:10px;'
                        f'margin-bottom:6px;font-size:0.76rem;">'
                        f'<span style="color:#c5dff0;width:140px;font-family:IBM Plex Mono,monospace;">'
                        f'{fname}</span>'
                        f'<div style="background:{bar_color};height:8px;width:{bar_width}px;'
                        f'border-radius:4px;opacity:0.85;"></div>'
                        f'<span style="color:{bar_color};font-family:IBM Plex Mono,monospace;'
                        f'min-width:40px;">{val:+.2f}</span>'
                        f'<span style="color:#3a7ca5;font-size:0.68rem;">→ {direction}</span>'
                        f'</div>'
                    )
                st.markdown(
                    f'<div style="padding:0.5rem 0.5rem 0.2rem;">{rows_html}</div>',
                    unsafe_allow_html=True
                )

        # ── ADVANCED SHAP ANALYSIS SECTION (S-GRADE FEATURE) ──────
        if not live_pred.get('auc_gated'):
            with st.expander("🎓  Advanced SHAP Analysis  ·  click to expand"):
                display_advanced_shap_analysis(selected_symbol, live_pred)

    # ── PRICE CHART ───────────────────────────────────────────────
    st.markdown('<p class="section-title">📈 Price Action &amp; Technical Chart</p>', unsafe_allow_html=True)
    ph = stock_data['price_history']
    if ph is not None and not ph.empty:
        min_date = ph.index.min().date()
        max_date = ph.index.max().date()
        with st.expander("📅  DATE RANGE · click to adjust", expanded=False):
            dc1, dc2 = st.columns(2)
            with dc1:
                date_from = st.date_input("From", value=max_date - timedelta(days=180),
                                          min_value=min_date, max_value=max_date)
            with dc2:
                date_to = st.date_input("To", value=max_date,
                                        min_value=min_date, max_value=max_date)
        with st.expander("❓  HOW TO READ THIS CHART"):
            st.markdown("""
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#3a7ca5;line-height:2;">
                <b style="color:#00c8ff;">Green candles</b> = price closed higher than open &nbsp;·&nbsp;
                <b style="color:#ff4b6e;">Red candles</b> = price closed lower than open<br>
                <b style="color:#f5a623;">SMA 20</b> = 20-day average (short trend) &nbsp;·&nbsp;
                <b style="color:#b06eff;">SMA 50</b> = 50-day average (medium trend)<br>
                <b style="color:#00c8ff;">RSI</b> = momentum · above 70 overbought · below 30 oversold
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="chart-wrapper">', unsafe_allow_html=True)
        fig = create_price_chart(ph, selected_symbol, exchange, currency,
                                 date_from=date_from, date_to=date_to)
        if fig:
            st.plotly_chart(fig, use_container_width=True,
                            config={'displayModeBar': True, 'displaylogo': False,
                                    'modeBarButtonsToRemove': ['select2d','lasso2d','autoScale2d']})
        else:
            st.warning("No data in selected date range.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("No price history data available for charting.")

    # ── TABS ─────────────────────────────────────────────────────

    # CHANGE THIS: Add the 5th tab for Backtesting
    tab1, tab2, tab3, tab4 = st.tabs([
        "📰  News & Sentiment",
        "📊  Technical Indicators",
        "🏢  Company Profile",
        "🤖  AI Model Performance",
        
    ])
    
    with tab1:
        live_sentiment = fetch_news_sentiment(selected_symbol)
        sentiment_to_show = live_sentiment if live_sentiment else stock_data['sentiment']
        display_sentiment_analysis(sentiment_to_show)
        
    with tab2: 
        display_technical_indicators(stock_data['technicals'], stock_data['price_history'])
        
    with tab3: 
        display_company_info(stock_data['company_info'], currency)
        
    with tab4: 
        display_model_performance(model_results, selected_symbol)
        
    

    # ── FOOTER ───────────────────────────────────────────────────
    st.markdown("""
    <div class="qe-footer">
        QUANTEDGE &nbsp;·&nbsp; MULTI-HORIZON EQUITY SIGNAL SYSTEM &nbsp;·&nbsp; VIT CHENNAI CAPSTONE 2026<br>
        AMRITHA K · 22BAI1318 &nbsp;·&nbsp; GARIMA MISHRA · 22BPS1153 &nbsp;·&nbsp; GUIDE: DR. THANIKACHALAM<br>
        
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

import subprocess
import sys

# Install in the correct order to avoid conflicts
packages_in_order = [
    # 1. First install numpy with compatible version
    "numpy==1.23.5",
    
    # 2. Then install packages that depend on numpy
    "pandas==2.1.4",
    "scipy==1.11.0",
    "scikit-learn==1.3.2",
    
    # 3. Then install others
    "streamlit==1.28.0",
    "yfinance==0.2.33",
    "requests==2.31.0",
    "python-dotenv==1.0.0",
    "alpha-vantage==2.3.1",
    "ta==0.10.2",
    "xgboost==1.7.6",
    "lightgbm==4.1.0",
    "plotly==5.17.0",
    "matplotlib==3.8.0",
    "seaborn==0.13.0",
    "statsmodels==0.14.0",
    "shap==0.43.0",
    "tqdm==4.66.1",
    "loguru==0.7.2",
    "joblib==1.3.2",
    "colorama==0.4.6"
    "nltk==3.9.2"
]

print("Installing packages in correct order to avoid conflicts...")
print("=" * 60)

for i, package in enumerate(packages_in_order, 1):
    print(f"\n[{i}/{len(packages_in_order)}] Installing {package}")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ Successfully installed {package}")
        else:
            print(f"⚠️  Issue with {package}, trying without version...")
            # Try without version constraint
            pkg_name = package.split('==')[0]
            subprocess.run([sys.executable, "-m", "pip", "install", pkg_name], check=False)
            
    except Exception as e:
        print(f"❌ Error installing {package}: {e}")

print("\n" + "=" * 60)
print("Installation complete! Testing imports...")

# Test imports
test_imports = """
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import ta
import sklearn
import xgboost
import lightgbm
import plotly
import matplotlib
import seaborn
import statsmodels
import shap
print('✅ All packages imported successfully!')
print(f'Numpy version: {np.__version__}')
print(f'Pandas version: {pd.__version__}')
"""

try:
    exec(test_imports)
except Exception as e:
    print(f"❌ Import test failed: {e}")
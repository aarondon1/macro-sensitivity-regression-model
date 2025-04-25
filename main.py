import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.gridspec as gridspec

# --- Parameters ---
tickers = ["BSM"]
start_date = "2000-01-01"
end_date = "2025-03-01"
macro_file = "FRED_macro_data_2000_2025.csv"
window_size = 252  # Rolling window size (approximately 1 year of trading days)

# --- Get stock data from Yahoo Finance ---
stock_data = {}
for ticker in tickers:
    print(f"Downloading data for {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date)
    
    # Flatten MultiIndex if it exists
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.reset_index(inplace=True)
    df = df[['Date', 'Close']].rename(columns={'Date': 'date', 'Close': f'{ticker}_price'})
    df[f'{ticker}_returns'] = df[f'{ticker}_price'].pct_change()
    stock_data[ticker] = df

# --- Load macroeconomic data ---
macro_df = pd.read_csv(macro_file, parse_dates=['date'])

# --- Merge all data ---
# Start with the first ticker
merged_df = stock_data[tickers[0]]

# Add other tickers
for ticker in tickers[1:]:
    merged_df = pd.merge(merged_df, stock_data[ticker][['date', f'{ticker}_price', f'{ticker}_returns']], 
                         on='date', how='outer')

# Add macro data
merged_df = pd.merge(merged_df, macro_df, on='date', how='inner')
merged_df.sort_values('date', inplace=True)
merged_df.dropna(inplace=True)

# --- Set up figure for plotting ---
plt.style.use('ggplot')
n_tickers = len(tickers)
n_factors = 5  # CPI, Fed Funds, Oil, Unemployment, VIX

# Create multi-panel figure
fig = plt.figure(figsize=(20, 15))
gs = gridspec.GridSpec(6, 1, height_ratios=[1, 2, 2, 2, 2, 2])

# Top panel: Returns comparison
ax0 = plt.subplot(gs[0])
for ticker in tickers:
    ax0.plot(merged_df['date'], merged_df[f'{ticker}_returns'], alpha=0.7, linewidth=0.8, label=ticker)
ax0.set_title('Daily Returns Comparison', fontsize=14)
ax0.legend(loc='upper left', ncol=n_tickers)
ax0.set_ylabel('Return')
ax0.grid(True, alpha=0.3)

# --- Perform rolling window regression and visualize results ---
# Initialize DataFrames to store results
rolling_betas = {ticker: pd.DataFrame(index=merged_df.index) for ticker in tickers}
rolling_r2 = {ticker: [] for ticker in tickers}

# Create factor labels for consistent naming
factor_labels = ['CPI', 'Fed Funds', 'Oil', 'Unemployment', 'VIX']
factor_columns = ['cpi', 'fed_funds', 'oil', 'unemployment', 'vix']

# Perform rolling regression
for ticker in tickers:
    print(f"Performing rolling regression for {ticker}...")
    for i in range(window_size, len(merged_df)):
        window = merged_df.iloc[i-window_size:i]
        X = window[factor_columns]
        y = window[f'{ticker}_returns']
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Store regression coefficients
        for j, col in enumerate(factor_columns):
            if col not in rolling_betas[ticker].columns:
                rolling_betas[ticker][col] = np.nan
            rolling_betas[ticker].iloc[i-1, rolling_betas[ticker].columns.get_loc(col)] = model.coef_[j]
        
        # Calculate and store R²
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        rolling_r2[ticker].append((merged_df.iloc[i-1]['date'], r2))

# Convert R^2 lists to DataFrames
for ticker in tickers:
    rolling_r2[ticker] = pd.DataFrame(rolling_r2[ticker], columns=['date', 'r2'])

# --- Create separate panel for each macro factor ---
axes = []
for i, (factor_label, factor_col) in enumerate(zip(factor_labels, factor_columns)):
    ax = plt.subplot(gs[i+1])
    axes.append(ax)
    
    for ticker in tickers:
        ax.plot(merged_df['date'], rolling_betas[ticker][factor_col], 
                label=ticker, linewidth=1.5, alpha=0.7)
    
    ax.set_title(f'Rolling Beta: {factor_label}', fontsize=14)
    ax.set_ylabel('Sensitivity')
    ax.legend(loc='upper left', ncol=n_tickers)
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Format dates nicely on x-axis
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))

# --- Create a separate figure for R^2 scores ---
plt.figure(figsize=(15, 6))
for ticker in tickers:
    plt.plot(rolling_r2[ticker]['date'], rolling_r2[ticker]['r2'], 
             label=ticker, linewidth=1.5, alpha=0.7)

plt.title('Rolling Window R² Score (Model Fit Quality)', fontsize=14)
plt.ylabel('R² Score')
plt.legend(loc='upper left', ncol=n_tickers)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# --- Adjust the main figure layout ---
plt.figure(fig.number)
plt.tight_layout()
plt.subplots_adjust(hspace=0.3)

# --- Perform full-period regression for each ticker and print results ---
print("\nFull Period Macro Sensitivity Results:")
print("=" * 60)

for ticker in tickers:
    X = merged_df[factor_columns]
    y = merged_df[f'{ticker}_returns']
    
    model = LinearRegression()
    model.fit(X, y)
    
    r2 = r2_score(y, model.predict(X))
    
    print(f"\nMacro Sensitivity Betas for {ticker}:")
    print("-" * 40)
    for i, col in enumerate(factor_columns):
        print(f"  Beta {factor_labels[i]:<13}: {model.coef_[i]:+.6f}")
    print(f"  R² Score: {r2:.4f}")

plt.show()
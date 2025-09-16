import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, skew, kurtosis

import get_stock_data as xf
import portfolio_analysis_skeleton as pa

# Load data for SPY, NVDA, and DAL
symbols = ['NVDA', 'DAL']  
all_symbols = symbols + ['SPY']
start_date = '2020-01-01'
end_date = '2022-12-31'

returns = pa.load_stock_data(all_symbols, start_date, end_date)

# Analytics for each stock
print("ALPHA, BETA, AND CORRELATION ANALYSIS")

spy_returns = returns['SPY']
symbol_to_returns = {
    'NVDA': returns['NVDA'],
    'DAL': returns['DAL'],
}

for symbol in ['NVDA', 'DAL']:
    y_series = symbol_to_returns[symbol]
    y_axis = symbol + ' Returns'
    title = 'SPY vs ' + symbol + ' Returns'

    # Create scatter plot and get regression metrics
    slope, intercept, r_value = pa.scatter_with_regression(spy_returns, y_series, 'SPY Returns', y_axis, title)
    
    print(f'\n{symbol} Analysis:')
    print(f'  Alpha (intercept): {intercept:.6f}')
    print(f'  Beta (slope): {slope:.4f}')
    print(f'  Correlation: {r_value:.4f}')



# Create histogram comparison chart
fig, ax = plt.subplots(figsize=(12, 8))

assets = ['SPY', 'NVDA', 'DAL']
colors = ['blue', 'red', 'green']

for i, asset in enumerate(assets):
    returns_data = returns[asset]
    mean_return = returns_data.mean()
    
    # Create histogram with reduced opacity
    ax.hist(returns_data, bins=50, alpha=0.6, color=colors[i], 
            label=f'{asset} (μ={mean_return:.4f})', density=True)
    
    # Draw vertical line at mean return
    ax.axvline(mean_return, color=colors[i], linestyle='--', linewidth=2, alpha=0.8)

ax.set_xlabel('Daily Returns')
ax.set_ylabel('Density')
ax.set_title('Histogram Comparison: SPY, NVDA, and DAL Daily Returns (2020-2022)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nVisual Analysis:")
print("- Width of histogram = Volatility (σ)")
print("- Horizontal position of mean = Alpha potential")
print("- Shape shows distribution characteristics")



#------------------------


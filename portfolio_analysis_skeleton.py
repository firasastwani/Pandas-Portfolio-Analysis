"""
HW03 Stock Analysis Skeleton
----------------------------

This script provides a scaffold for the HW03 assignment, which asks you to
analyze stock return data using histograms, correlation, and regression
metrics such as alpha and beta. Unlike previous examples that relied on
`yfinance` to fetch data from the internet, this skeleton uses a local
utility module `LL_get_local_data` to load data from CSV files stored in a
directory on your machine. The provided module mimics the behavior of
`yf.download(...)["Adj Close"]` but reads from your local `data` folder.

You should fill in the TODO sections with your own implementations. The
functions below are designed to guide you through each part of the
assignment:

  * Task 0: Loading adjusted close data and computing daily returns.
  * Task 1: Plotting histograms of daily returns and calculating skewness
    and kurtosis.
  * Task 2: Selecting stocks based on the histogram analysis.
  * Task 3: Performing scatter plot and correlation analysis.
  * Task 4: Calculating alpha and beta relative to the benchmark.
  * Task 5: Comparing selections made from histogram analysis with those
    derived from alpha/beta analysis.

This file does not implement the full functionality. Instead, it lays
out the structure you can follow and expand upon. Be sure to import any
additional modules you need as you complete the TODOs (e.g., `scipy.stats`
for skewness and kurtosis or `scipy.stats.linregress` for regression).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, skew, kurtosis

# Import the local data loader. This module provides the `download_local`
# function, which reads CSV files containing stock price data and returns
# a DataFrame of adjusted close prices.
import get_stock_data as xf

def load_stock_data(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """ Load adjusted close prices for the given symbols and compute daily returns.

    This helper function uses `xf.download_local` to load adjusted close
    prices for each symbol over the given date range, then computes
    percentage daily returns from the price data. The resulting DataFrame
    has dates as the index and symbols as columns.

    Args:
        symbols: A list of ticker symbols to load (e.g., ['SPY', 'IBM']).
        start: Start date in 'YYYY-MM-DD' format.
        end: End date in 'YYYY-MM-DD' format.

    Returns:
        A DataFrame of daily return percentages with the same index as
        the price data. Missing values are dropped.
    """
    prices = xf.download_local(symbols, start=start, end=end)
    print(prices.head())
    # Compute daily returns (percentage change) and drop the first NaN row
    returns = prices.pct_change(fill_method=None).dropna()
    return returns


def plot_histogram(returns: pd.Series, symbol: str) -> tuple[float, float]:
    """ Plot a histogram of daily returns and overlay a normal distribution.

    This function should create a histogram of the provided return series,
    overlay a Gaussian (normal) curve using the sample mean and standard
    deviation, and compute the skewness and kurtosis of the distribution.
    It should return the skewness and kurtosis values.

    Args:
        returns: A pandas Series of daily return values for a single stock.
        symbol: The ticker symbol, used for the plot title and legend.

    Returns:
        A tuple containing (skewness, kurtosis).

    TODO: Implement the plotting using matplotlib. You may also use
    scipy.stats to compute skewness and kurtosis.
    """

    #Example structure (remove when implementing):
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(returns, bins=50, density=True, alpha=0.6, label=f"{symbol} Returns")
    #Compute the Gaussian curve and overlay it
    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 100)
    gaussian = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu)/sigma)**2)
    ax.plot(x, gaussian, color='red', label='Normal Distribution')
    ax.set_title(f"Histogram of {symbol} Daily Returns")
    ax.legend()
    plt.show()



    # Calculate skewness and kurtosis
    skewness = skew(returns)
    kurt = kurtosis(returns)
    
    return (skewness, kurt)


def scatter_with_regression(x: pd.Series, y: pd.Series, x_label: str, y_label: str, title: str) -> tuple[
    float, float, float]:
    """Create a scatter plot with a regression line and compute correlation metrics.

    Given two aligned Series of daily returns (e.g., market and a stock), this
    function should plot the returns against each other, fit a linear
    regression line, and return the slope (beta), intercept (alpha), and
    Pearson correlation coefficient. Use `matplotlib` for plotting and
    `scipy.stats.linregress` for regression.

    Args:
        x: Series of returns for the independent variable (e.g., SPY).
        y: Series of returns for the dependent variable (e.g., a stock).
        x_label: Label for the x-axis.
        y_label: Label for the y-axis.
        title: Title for the plot.

    Returns:
        A tuple (slope, intercept, r_value) representing the regression
        coefficients and correlation.

    TODO: Implement the scatter plot and regression using linregress.
    """
    # TODO: Replace the following placeholder values with real computations

    # x is the market
    # y is the stock

    plt.scatter(x, y, alpha=0.6)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # Fit simple linear regression (y = m*x + b)
    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = m * x_line + b
    plt.plot(x_line, y_line, color='red', label='Regression Line')
    plt.legend()

    plt.show()

    # Pearson correlation
    r = float(np.corrcoef(x, y)[0, 1])
    return (m, b, r)


def compute_alpha_beta(stock_returns: pd.Series, market_returns: pd.Series) -> tuple[float, float]:
    """Compute alpha and beta of a stock relative to the market.

    Alpha and beta are obtained from the slope and intercept of the
    regression line when regressing stock returns against market returns.
    Use `scipy.stats.linregress` to obtain these values and return them.

    Args:
        stock_returns: A Series of daily returns for the stock.
        market_returns: A Series of daily returns for the market benchmark.

    Returns:
        A tuple (alpha, beta).

    TODO: Implement the regression to compute alpha (intercept) and beta (slope).
    """

    result = linregress(stock_returns, market_returns)

    beta = result.slope
    alpha = result.intercept

    print(f'beta = {beta}')
    print(f'alpha = {alpha}')

    # TODO: Remove the placeholder and return actual (alpha, beta)
    return (alpha, beta)


def main() -> None:
    """ Entry point for the HW03 analysis.

    Set up your list of symbols and date range here. Then call the helper
    functions to perform each task. Use SPY (or another benchmark) as
    the market index for comparison. Replace the placeholder lists and
    dates with your own choices. Throughout this function, you'll see
    comments indicating where you should implement the logic for each
    assignment task.
    """

    # TODO: Specify the stock symbols you want to analyze (do not include 'SPY' here)
    symbols = ['IBM', 'GOOG', 'SMCI']  # e.g., ['IBM', 'GOOG', 'AAPL']

    # TODO: Define the date range for your analysis
    start_date = '2020-01-01'  # e.g., '2020-01-01'
    end_date = '2022-12-31'  # e.g., '2022-12-31'

    # Include the benchmark symbol (e.g., SPY) to obtain market returns
    all_symbols = symbols + ['SPY']
    returns = load_stock_data(all_symbols, start_date, end_date)

    # Use the already loaded data from the returns DataFrame
    spy_returns = returns['SPY']
    ibm_returns = returns['IBM']
    goog_returns = returns['GOOG']
    smci_returns = returns['SMCI']

    # Map symbols to their return Series for easy lookup
    symbol_to_returns = {
        'IBM': ibm_returns,
        'GOOG': goog_returns,
        'SMCI': smci_returns,
    }


    # Task 1: Histogram analysis
    # -------------------------------------
    # For each stock (not including SPY), call `plot_histogram` to create
    # a histogram and compute skewness and kurtosis. Use these metrics
    # to inform your selection of stocks in Task 2.
    # Example:
    # for symbol in symbols:
    #     skewness, kurt = plot_histogram(returns[symbol], symbol)
    #     print(f"{symbol}: skew={skewness:.4f}, kurt={kurt:.4f}")

    histogram_metrics = {}
    for symbol in symbols:
        print(f"\n{symbol} Histogram Analysis:")
        skewness, kurt = plot_histogram(returns[symbol], symbol)
        histogram_metrics[symbol] = {'skewness': skewness, 'kurtosis': kurt}
        print(f"  Skewness: {skewness:.4f}")
        print(f"  Kurtosis: {kurt:.4f}")



    # Task 2: Select stocks based on histogram results
    # -------------------------------------
    # Use the skewness and kurtosis from Task 1 to choose two 'good'
    # stocks and one 'poor' stock. You can store the selections in
    # lists such as good_stocks and poor_stock.

    # Based on histogram analysis results:
    # IBM: Skewness: -0.5214, Kurtosis: 8.0979 (moderate negative skew, high kurtosis)
    # GOOG: Skewness: -0.0352, Kurtosis: 3.0623 (nearly normal distribution)
    # SMCI: Skewness: 1.1548, Kurtosis: 15.9335 (strong positive skew, very high kurtosis)
    
    # Histogram-based selection:
    good_stocks_histogram = ['GOOG', 'IBM']  # GOOG is most normal, IBM is second best
    poor_stock_histogram = ['SMCI']  # SMCI has extreme skewness and kurtosis
    
    print("Good stocks (based on histogram analysis):", good_stocks_histogram)
    print("Poor stock (based on histogram analysis):", poor_stock_histogram)
    print("\nReasoning:")
    print("- GOOG: Nearly normal distribution (skew ≈ 0, kurtosis ≈ 3)")
    print("- IBM: Moderate deviation from normal (acceptable for stable investment)")
    print("- SMCI: Extreme distribution (high positive skew, very high kurtosis = risky)")


    # Task 3: Scatter plot and correlation analysis
    # -------------------------------------
    # For each selected stock, plot its returns against the market
    # returns using `scatter_with_regression`, and record the
    # slope (beta), intercept (alpha), and correlation coefficient.

    # Task 4: Compute alpha and beta directly
    # -------------------------------------
    # For each selected stock, call `compute_alpha_beta` to obtain
    # it's alpha and beta relative to the benchmark.


 # Task 3 & 4: Scatter plots and alpha/beta analysis
    print("\n" + "=" * 60)
    print("TASK 3 & 4: SCATTER PLOTS AND ALPHA/BETA ANALYSIS")
    print("=" * 60)
    
    alpha_beta_metrics = {}
    correlation_metrics = {}
    
    for symbol in symbols:
        y_series = symbol_to_returns.get(symbol)
        if y_series is None:
            continue
        y_axis = symbol + ' Returns'
        title = 'SPY vs ' + symbol + ' Returns'

        # Create scatter plot and get regression metrics
        slope, intercept, r_value = scatter_with_regression(spy_returns, y_series, 'SPY Returns', y_axis, title)
        
        # Store metrics
        alpha_beta_metrics[symbol] = {'alpha': intercept, 'beta': slope}
        correlation_metrics[symbol] = r_value
        
        print(f'\n{symbol} Analysis:')

        alpha, beta = compute_alpha_beta(spy_returns, y_series)

        print(f'  Alpha (intercept): {intercept:.6f}')
        print(f'  Beta (slope): {slope:.4f}')
        print(f'  Correlation: {r_value:.4f}')


    # Task 5: Compare histogram-based picks with alpha/beta picks
    # -------------------------------------
    # Compare the stocks chosen in Task 2 with those that appear
    # attractive according to alpha/beta and correlation metrics.
    # Summarize which method you find more informative and why.

    print('Histogram method: ')
    print('- Focus on distribution shape and normality')
    print('- Good for indenitifying stable predictable retruns')

    print('Alpha and Beta method: ')
    print('- Focus on risk-adjusted returns and market relationship')
    print('- Good for indentifiying stocks that perform better than market (alpha)')

    print("\nRECOMMENDATION:")
    print("- For conservative investors: Use histogram analysis (focus on distribution stability)")
    print("- For growth investors: Use alpha/beta analysis (focus on excess returns)")
    print("- For balanced approach: Combine both methods and look for stocks that rank well in both")

    # NOTE: You are encouraged to create additional helper functions or
    # modules as needed. This skeleton is merely a starting point.
    pass


if __name__ == '__main__':
    main()
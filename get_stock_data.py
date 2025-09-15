import pandas as pd
import os
from typing import List

def download_local(symbols: List[str], start: str, end: str, data_dir: str = 'data/data') -> pd.DataFrame:
    """
    Reads stock data from local CSV files and returns a DataFrame of Adjusted Close prices.

    This function mimics the behavior of yf.download(symbols, start, end)['Adj Close'].

    Args:
        symbols: A list of stock symbols (e.g., ['IBM', 'GOOG']).
        start: The start date for the data in 'YYYY-MM-DD' format.
        end: The end date for the data in 'YYYY-MM-DD' format.
        data_dir: The directory where the CSV files are stored. Defaults to 'data'.

    Returns:
        A pandas DataFrame with dates as the index and 'Adj Close' prices for
        each symbol as columns. Returns an empty DataFrame if no data is found.
    """
    all_adj_closes = []

    # Convert start and end strings to datetime objects for comparison
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)

    for symbol in symbols:
        file_path = os.path.join(data_dir, f"{symbol}.csv")

        if not os.path.exists(file_path):
            print(f"Warning: Data file not found for symbol '{symbol}' at '{file_path}'. Skipping.")
            continue

        try:
            # Read the CSV, parsing the 'Date' column into datetime objects and setting it as the index
            df = pd.read_csv(file_path, index_col='Date', parse_dates=True)

            # Filter the DataFrame to the specified date range
            df_filtered = df.loc[start_date:end_date]

            # Select the 'Adj Close' column and rename it to the symbol
            adj_close = df_filtered[['Adj Close']].rename(columns={'Adj Close': symbol})

            all_adj_closes.append(adj_close)

        except Exception as e:
            print(f"Error processing file for symbol '{symbol}': {e}")

    if not all_adj_closes:
        return pd.DataFrame()

    # Concatenate all the individual symbol DataFrames along the columns
    final_df = pd.concat(all_adj_closes, axis=1)

    return final_df


if __name__ == '__main__':
    # --- Example Usage ---

    # 1. Define your symbols and date range
    symbols_to_load = ['IBM', 'GOOG', 'SPY', 'FAKE']  # 'FAKE' is included to show the warning for a missing file
    start_date = '2020-01-01'
    end_date = '2022-12-31'

    print(f"Loading data for symbols: {symbols_to_load}")
    print(f"Date Range: {start_date} to {end_date}\n")

    # 2. Call the function to load the data
    # Ensure you have a 'data' directory with 'IBM.csv', 'GOOG.csv', 'SPY.csv' in it.
    adj_close_df = download_local(symbols=symbols_to_load, start=start_date, end=end_date)

    # 3. Print the resulting DataFrame
    if not adj_close_df.empty:
        print("--- Resulting Adjusted Close DataFrame ---")
        #print(adj_close_df)

        print(adj_close_df.head(10))
    else:
        print("No data was loaded. Please check your file paths and date range.")


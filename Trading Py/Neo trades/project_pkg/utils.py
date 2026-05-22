# Import the libraries
import pyautogui
import time
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import os
import itertools

from datetime import date

import pandas as pd
from statsmodels.tsa.stattools import adfuller
from hurst import compute_Hc
from MFDFA import MFDFA
from typing import Tuple

pd.set_option('display.max_columns', None)

#Function 1: #load the data and basic analyses

#Sub-function: Get slope for the MAs
def numpy_slope(series, window=10):
    def calc_slope(x):
        if len(x) < window:
            return np.nan
        coeffs = np.polyfit(range(len(x)), x, 1)
        return coeffs[0]
    
    return series.rolling(window=window).apply(calc_slope)

#Run function
def Run(df,name):
    #Currency and dates
    df['Currency'] = name
    df['<DATE>'] = pd.to_datetime(df['<DATE>'], format='%Y.%m.%d')
    df.set_index('<DATE>')
    # Note these below apply to Lows and Highs
    df['1m_MA_Filter'] = 0 
    df['3m_MA_Filter'] = 0
    df['6m_MA_Filter'] = 0
    df['12m_MA_Filter'] = 0
    df['AT_MA_Filter'] = 0

    #High/Low since 1 month: 23 days or so
    df['1m_High'] = df['<CLOSE>'].rolling(window=23).max()
    df['1m_Low'] = df['<CLOSE>'].rolling(window=23).min()
    df['1m_Mean'] = df['<CLOSE>'].rolling(23).mean()
        
    #High/Low since 3 months: 70 days or so
    df['3m_High'] = df['<CLOSE>'].rolling(window=70).max()
    df['3m_Low'] = df['<CLOSE>'].rolling(window=70).min()
    df['3m_Mean'] = df['<CLOSE>'].rolling(70).mean()
    
    #High/Low since 6 months: 135 days or so
    df['6m_High'] = df['<CLOSE>'].rolling(window=135).max()
    df['6m_Low'] = df['<CLOSE>'].rolling(window=135).min()
    df['6m_Mean'] = df['<CLOSE>'].rolling(135).mean()
    
    #High/Low since 12 months: 260 days or so
    df['12m_High'] = df['<CLOSE>'].rolling(window=260).max()
    df['12m_Low'] = df['<CLOSE>'].rolling(window=260).min()
    df['12m_Mean'] = df['<CLOSE>'].rolling(260).mean()
        
    #High/Low since 2022
    df['AT_High'] = df['<CLOSE>'].max()
    df['AT_Low'] = df['<CLOSE>'].min()
    df['AT_Mean'] = df['<CLOSE>'].mean()
    df['AT_Med'] = df['<CLOSE>'].median()
    
    # List of time periods
    time_periods = ['1m', '3m', '6m', '12m', 'AT']

    # Create 90th percentile slope filters for each time period
    for period in time_periods:
        # Create MA Filter for High and Close
        df.loc[round(df[f'{period}_High'], 2) == round(df['<CLOSE>'], 2), f'{period}_MA_Filter'] = 1

        # Create MA Filter for Low and Close
        df.loc[round(df[f'{period}_Low'], 2) == round(df['<CLOSE>'], 2), f'{period}_MA_Filter'] = 1

        # Calculate Slope
        df[f'{period}_MA_slope'] = numpy_slope(df[f'{period}_Mean'])

        # Normalize Slope
        df[f'{period}_MA_slope_nm'] = (df[f'{period}_MA_slope'] - df[f'{period}_MA_slope'].mean()) / df[f'{period}_MA_slope'].std()
       
        # Create binary filter column on the percentile
        df[f'{period}_slope'] = (df[f'{period}_MA_slope_nm'] >= df[f'{period}_MA_slope_nm'].quantile(0.9)).astype(int)
    
    # MA Crosses: Where rounded till 3 is the same eg 0.6659 = 0.6663 = 0.666 etc
    # 1m vs 3m
    df.loc[round(df['1m_Mean'], 3) == round(df['3m_Mean'], 3), 'MA_1_3'] = 1

    # 1m vs 6m
    df.loc[round(df['1m_Mean'], 3) == round(df['6m_Mean'], 3), 'MA_1_6'] = 1

    # 1m vs 12m
    df.loc[round(df['1m_Mean'], 3) == round(df['12m_Mean'], 3), 'MA_1_12'] = 1

    # 1m vs AT
    df.loc[round(df['1m_Mean'], 3) == round(df['AT_Mean'], 3), 'MA_1_AT'] = 1

    # 3m vs 6m
    df.loc[round(df['3m_Mean'], 3) == round(df['6m_Mean'], 3), 'MA_3_6'] = 1

    # 3m vs 12m
    df.loc[round(df['3m_Mean'], 3) == round(df['12m_Mean'], 3), 'MA_3_12'] = 1

    # 3m vs AT
    df.loc[round(df['3m_Mean'], 3) == round(df['AT_Mean'], 3), 'MA_3_AT'] = 1

    # 6m vs 12m
    df.loc[round(df['6m_Mean'], 3) == round(df['12m_Mean'], 3), 'MA_6_12'] = 1

    # 6m vs AT
    df.loc[round(df['6m_Mean'], 3) == round(df['AT_Mean'], 3), 'MA_6_AT'] = 1

    # 12m vs AT
    df.loc[round(df['12m_Mean'], 3) == round(df['AT_Mean'], 3), 'MA_12_AT'] = 1

    # Optional: Fill NaN with 0 if needed
    ma_filter_columns = [
        'MA_1_3', 'MA_1_6', 'MA_1_12', 'MA_1_AT', 
        'MA_3_6', 'MA_3_12', 'MA_3_AT', 
        'MA_6_12', 'MA_6_AT', 
        'MA_12_AT'
    ]
    
    for col in ma_filter_columns:
        df[col] = df[col].fillna(0)

    return df

#Function 2: Filters to find the Trends. Where opps lie:
def Filter(df_c, filter_columns):
    # Create the filter condition
    filter_condition = ' | '.join([f'(df_c["{col}"]==1)' for col in filter_columns])
    
    # Create cumulative filter column
    df_c2 = df_c.copy()
    df_c2['Total_Filters'] = df_c2[filter_columns].sum(axis=1)
    
    # Apply the filter
    return df_c2[eval(filter_condition)]

#Function 3: Creating new dfs for sans-USD
def df_div(symbols):
    #Step 1: Create and save the data
    dfs = {}
    base_path = r"C:\Users\nitis\Documents\Forex\Data\New data"
    cols = ["<OPEN>","<HIGH>","<LOW>","<CLOSE>"]
    for sym in symbols:
        path = fr"{base_path}\{sym}.csv"
        df = pd.read_csv(path, sep='\t')
        # ensure required columns
        required = ["<DATE>","<OPEN>","<HIGH>","<LOW>","<CLOSE>"]
        # normalize
        df = df.copy()
        df["<DATE>"] = pd.to_datetime(df["<DATE>"])
        df = df.loc[:, required]  # keep exact column order        
        # If symbol is USDxxx (USD is base), invert the 4 price columns
        if sym.startswith("USD"):
            # avoid divide-by-zero; replace zeros with NaN first
            df[cols] = df[cols].replace({0: pd.NA}).astype(float)
            df[cols] = 1.0 / df[cols]
            # Optional: rename to represent inversion (not required)
            # e.g., USDCAD inverted now represents CADUSD prices
        dfs[sym] = df
    # Step 2: divide on intersection of dates to form crosses
    result = {}  # e.g., result["EURGBP"] -> DataFrame

    #Loop through all datasets
    for a, b in itertools.combinations(symbols, 2):
        L = dfs[a]
        R = dfs[b]
        # ensure index is datetime and named "<DATE>"
        for df in (L, R):
            if not pd.api.types.is_datetime64_any_dtype(df.index):
                # if <DATE> is a column, set it as index
                if "<DATE>" in df.columns:
                    df["<DATE>"] = pd.to_datetime(df["<DATE>"])
                    df.set_index("<DATE>", inplace=True)
                else:
                    raise RuntimeError("No datetime index or <DATE> column found in L/R")

        # now L and R have proper datetime index; do union/reindex then divide
        common_index = L.index.union(R.index)
        L = L.reindex(common_index)
        R = R.reindex(common_index)
        cross_df = L[cols].divide(R[cols])

        # reset index to get <DATE> column back
        cross_df = cross_df.reset_index().rename(columns={cross_df.columns[0]: "<DATE>"})

        # perform division while keeping the index
        cross_df = L[cols].divide(R[cols])

        # now reset index and ensure the column is named "<DATE>"
        cross_df = cross_df.reset_index()               # index -> first column
        if cross_df.columns[0] != "<DATE>":
            cross_df = cross_df.rename(columns={cross_df.columns[0]: "<DATE>"})

        # now you can safely select ordered columns
        cross_df = cross_df.loc[:, ["<DATE>"] + cols]

        base = a.replace("USD", "")
        quote = b.replace("USD", "")
        cross_name = f"{base}{quote}"

        result[cross_name] = cross_df

    return(result)

#Function 4: Add check on close
def close_check(df):
    first_close = df.iloc[0]['<CLOSE>']
    last_close = df.iloc[-1]['<CLOSE>']

    if last_close > first_close:
        label = 'higher'
    elif last_close < first_close:
        label = 'lower'
    else:
        label = 'equal'
        
    return label

#Analyse the data using above functions
def analyse(name, df):
    df2 = Run(df,name)
    df2 = df2.tail()
    df2['Close_check'] = close_check(df2)
    df2 = df2.tail(1)

    # Strong set: 4 filters
    # - Close is near AT and/or 12m H/L then time could be to reverse - Strong
    # - Rolling MA crosses comparisons for 12 with various and AT
    df_st = Filter(df2, ['AT_MA_Filter', '12m_MA_Filter'
                        ,'MA_6_AT','MA_12_AT'])
    
    df_st = df_st [['Currency', 'Total_Filters','AT_High','AT_Low','<CLOSE>','Close_check']].sort_values(by='Total_Filters', ascending=False).reset_index(drop=True)
    df_st = df_st.drop_duplicates(subset=['Currency']) #, 'Total_Filters'
    
    # Weak set: 11 filters
    # - Close is near other H/L then time could be to reverse - Weak
    # - Rolling MA crosses comparisons for 1 with various and 3/6
    # - Slopes for sudden jumps
    df_wk = Filter(df2, ['1m_MA_Filter','3m_MA_Filter','6m_MA_Filter'
                         ,'MA_1_3', 'MA_1_6', 'MA_1_12', 'MA_1_AT','MA_3_6'
                        ,'1m_slope','3m_slope','6m_slope'])
    
    df_wk = df_wk [['Currency', 'Total_Filters','AT_High','AT_Low','<CLOSE>','Close_check']].sort_values(by='Total_Filters', ascending=False).reset_index(drop=True) 
    df_wk = df_wk.drop_duplicates(subset=['Currency'])
    
    return df_st,df_wk

#Hurst-exponent for how quickly mean reversion
def get_hurst(series):
    series = series.dropna().values
    lags = range(2, 100) # The time scales we are checking
   
    # Standard deviation of differences scales as lag^H
    sigmas = [np.std(series[lag:] - series[:-lag]) for lag in lags]
    # Linear fit: log(sigma) = H * log(lag) + constant
    poly = np.polyfit(np.log(lags), np.log(sigmas), 1)
    
    return poly[0] # The slope IS the Hurst exponent

#MFDA analysis - superset of Hurst
def get_MFDFA(series):
    # 1. Generate or load your time series data
    # For this example, we use a random walk (integrated white noise)
    data = np.cumsum(series)

    # 2. Define the parameters for MFDFA
    # Select the scales (window sizes) - usually logarithmic ranging from 3-1000 as per convention
    lag = np.unique(np.logspace(0.5, 3, 20).astype(int))

    # Select the q-values (the "generalized" part)
    # q < 0 sensitive to small fluctuations; q > 0 sensitive to large ones as per convention
    q = np.linspace(-5, 5, 41)

    # The polynomial order for detrending (1 = linear, 2 = quadratic)
    order = 2 #as per convention

    # 3. Run the MFDFA
    # This returns the fluctuation function F_q(s)
    lag, dfa = MFDFA(data, lag=lag, q=q, order=order)

    # 4. Extract the Generalized Hurst Exponents (Hq)
    # We calculate the slope of log(F_q) vs log(lag) for each q
    num_columns = dfa.shape[1]
    Hq = np.zeros(len(q))
    for i in range(num_columns):
        # The log of DFA can produce -inf if fluctuations are 0
        y = np.log(dfa[:, i])
        x = np.log(lag)

        mask = np.isfinite(y) & np.isfinite(x)

        # Need at least 2 points to fit a line!
        if np.sum(mask) > 2:
            poly = np.polyfit(x[mask], y[mask], 1)
            Hq[i] = poly[0]
        else:
            Hq[i] = np.nan

    return Hq[q == 2][0] #Second moment again as per convention

# Hurst using Fractals Cycle
def calculate_hurst(prices: np.ndarray, min_window: int = 10) -> Tuple[float, float]:
    # Convert prices to log returns - for %ge returns check (as stocks esp are %ge driven not absolute) and ensuring min. is 0
    if len(prices) < min_window * 2:
        raise ValueError(f"Need at least {min_window * 2} data points")

    returns = np.diff(np.log(prices))
    n = len(returns)

    # Generate sub-period sizes (powers of 2 work well)
    max_k = int(np.floor(np.log2(n)))
    sizes = [2**i for i in range(int(np.log2(min_window)), max_k)]
    sizes = [s for s in sizes if s <= n // 2]

    if len(sizes) < 2:
        raise ValueError("Not enough data for R/S analysis")

    rs_values = []

    for size in sizes:
        # Number of sub-periods of this size
        num_periods = n // size
        rs_sum = 0.0
        valid_periods = 0

        for i in range(num_periods):
            # Extract sub-period
            start = i * size
            end = start + size
            subset = returns[start:end]

            # Calculate mean and deviations
            mean = np.mean(subset)
            deviations = subset - mean

            # Cumulative sum of deviations
            cumsum = np.cumsum(deviations)

            # Range of cumulative deviations
            range_val = np.max(cumsum) - np.min(cumsum)

            # Standard deviation
            std = np.std(subset, ddof=1)

            # Skip if std is too small (avoid division issues)
            if std > 1e-10:
                rs_sum += range_val / std
                valid_periods += 1

        if valid_periods > 0:
            rs_values.append((size, rs_sum / valid_periods))

    if len(rs_values) < 2:
        raise ValueError("Could not compute enough R/S values")

    # Linear regression of log(R/S) vs log(n)
    sizes_arr = np.array([v[0] for v in rs_values])
    rs_arr = np.array([v[1] for v in rs_values])

    log_sizes = np.log(sizes_arr)
    log_rs = np.log(rs_arr)

    # Slope is the Hurst exponent
    slope, intercept = np.polyfit(log_sizes, log_rs, 1)

    # Calculate R-squared for quality assessment
    predicted = slope * log_sizes + intercept
    ss_res = np.sum((log_rs - predicted) ** 2)
    ss_tot = np.sum((log_rs - np.mean(log_rs)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return slope, r_squared

#Rolling hurst checks
def rolling_hurst(series, window_size=500 #As the data is ~1000 so to get enough input but some files are much smaller
                  , max_window_fraction=0.15): #Same - ~500 rows of data so best to windows that have enough sub-samples
    hurst_values = []
    
    # We need enough data for at least one window
    if len(series) < window_size:
        raise ValueError(f"Series length ({len(series)}) must be >= window_size ({window_size})")

    # Slide the window across the data
    for i in range(window_size, len(series) + 1):
        sub_series = series[i - window_size : i]
        
        # 1. Get raw H and the underlying R/S data from the package
        # We use kind='price' assuming you are passing raw prices
        H_raw, c, data = compute_Hc(sub_series, kind='price', simplified=True)
        
        list_n = data[0]
        list_rs = data[1]
        
        # 2. Chop off the tail (The 1/10 Rule)
        limit = window_size * max_window_fraction
        valid_idx = [j for j, n in enumerate(list_n) if n <= limit]
        
        if len(valid_idx) < 2:
            continue # Not enough points left to fit a line
            
        f_n = np.log(np.array(list_n)[valid_idx])
        f_rs = np.log(np.array(list_rs)[valid_idx])
        
        # 3. Re-calculate the stable slope (Hurst)
        H_stable, intercept = np.polyfit(f_n, f_rs, 1)
        hurst_values.append(H_stable)

    # 4. Summarize the rolling statistics
    if not hurst_values:
        return np.nan, np.nan
        
    avg_h = np.mean(hurst_values)
    std_h = np.std(hurst_values)
    
    return avg_h, std_h
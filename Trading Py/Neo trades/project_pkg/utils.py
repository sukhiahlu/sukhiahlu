import os
import itertools
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from datetime import date
from typing import Tuple, Dict, List
from statsmodels.tsa.stattools import adfuller
from MFDFA import MFDFA
from hurst import compute_Hc

pd.set_option('display.max_columns', None)

# --- Configuration ---
BASE_PATH = os.getenv("FOREX_DATA_PATH", r"./data")


def numpy_slope(series: pd.Series, window: int = 10) -> pd.Series:
    """Calculates rolling linear regression slope."""
    def calc_slope(x):
        if len(x) < window:
            return np.nan
        coeffs = np.polyfit(range(len(x)), x, 1)
        return coeffs[0]
    
    return series.rolling(window=window).apply(calc_slope)


def Run(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Generates features, lookback filters, and normalized moving average slopes."""
    df = df.copy()
    df['Currency'] = name
    df['<DATE>'] = pd.to_datetime(df['<DATE>'], format='%Y.%m.%d')
    
    time_periods = ['1m', '3m', '6m', '12m', 'AT']
    for period in time_periods:
        df[f'{period}_MA_Filter'] = 0

    # Rolling Extremums and Means
    df['1m_High'] = df['<CLOSE>'].rolling(window=23).max()
    df['1m_Low'] = df['<CLOSE>'].rolling(window=23).min()
    df['1m_Mean'] = df['<CLOSE>'].rolling(window=23).mean()
        
    df['3m_High'] = df['<CLOSE>'].rolling(window=70).max()
    df['3m_Low'] = df['<CLOSE>'].rolling(window=70).min()
    df['3m_Mean'] = df['<CLOSE>'].rolling(window=70).mean()
    
    df['6m_High'] = df['<CLOSE>'].rolling(window=135).max()
    df['6m_Low'] = df['<CLOSE>'].rolling(window=135).min()
    df['6m_Mean'] = df['<CLOSE>'].rolling(window=135).mean()
    
    df['12m_High'] = df['<CLOSE>'].rolling(window=260).max()
    df['12m_Low'] = df['<CLOSE>'].rolling(window=260).min()
    df['12m_Mean'] = df['<CLOSE>'].rolling(window=260).mean()
        
    df['AT_High'] = df['<CLOSE>'].max()
    df['AT_Low'] = df['<CLOSE>'].min()
    df['AT_Mean'] = df['<CLOSE>'].mean()
    df['AT_Med'] = df['<CLOSE>'].median()

    # Boundary Touch and Slope Normalization
    for period in time_periods:
        is_near_high = np.isclose(df[f'{period}_High'], df['<CLOSE>'], rtol=1e-4)
        is_near_low = np.isclose(df[f'{period}_Low'], df['<CLOSE>'], rtol=1e-4)
        df.loc[is_near_high | is_near_low, f'{period}_MA_Filter'] = 1

        df[f'{period}_MA_slope'] = numpy_slope(df[f'{period}_Mean'])
        std_slope = df[f'{period}_MA_slope'].std()
        
        if std_slope > 0:
            df[f'{period}_MA_slope_nm'] = (df[f'{period}_MA_slope'] - df[f'{period}_MA_slope'].mean()) / std_slope
        else:
            df[f'{period}_MA_slope_nm'] = 0
       
        df[f'{period}_slope'] = (df[f'{period}_MA_slope_nm'] >= df[f'{period}_MA_slope_nm'].quantile(0.9)).astype(int)
    
    # Scale-Independent Moving Average Crosses (using 0.05% proximity band)
    ma_crosses = {
        'MA_1_3': ('1m_Mean', '3m_Mean'),
        'MA_1_6': ('1m_Mean', '6m_Mean'),
        'MA_1_12': ('1m_Mean', '12m_Mean'),
        'MA_1_AT': ('1m_Mean', 'AT_Mean'),
        'MA_3_6': ('3m_Mean', '6m_Mean'),
        'MA_3_12': ('3m_Mean', '12m_Mean'),
        'MA_3_AT': ('3m_Mean', 'AT_Mean'),
        'MA_6_12': ('6m_Mean', '12m_Mean'),
        'MA_6_AT': ('6m_Mean', 'AT_Mean'),
        'MA_12_AT': ('12m_Mean', 'AT_Mean')
    }

    for col, (ma_a, ma_b) in ma_crosses.items():
        df[col] = np.isclose(df[ma_a], df[ma_b], rtol=5e-4).astype(int)

    return df


def Filter(df_c: pd.DataFrame, filter_columns: List[str]) -> pd.DataFrame:
    """Filters rows matching any provided conditions and adds a sum column."""
    if df_c.empty:
        return df_c
    
    df_c2 = df_c.copy()
    df_c2['Total_Filters'] = df_c2[filter_columns].sum(axis=1)
    
    mask = (df_c2[filter_columns] == 1).any(axis=1)
    return df_c2[mask]


def df_div(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Ingests underlying files, handles USD base inversions, and computes cross-pairs."""
    dfs = {}
    cols = ["<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>"]
    required = ["<DATE>"] + cols
    
    for sym in symbols:
        path = os.path.join(BASE_PATH, f"{sym}.csv")
        if not os.path.exists(path):
            continue
            
        df = pd.read_csv(path, sep='\t')
        df["<DATE>"] = pd.to_datetime(df["<DATE>"])
        df = df[required].copy()
        
        if sym.startswith("USD"):
            df[cols] = df[cols].replace({0: np.nan}).astype(float)
            df[cols] = 1.0 / df[cols]
            
        df.set_index("<DATE>", inplace=True)
        dfs[sym] = df

    result = {}

    for a, b in itertools.combinations(dfs.keys(), 2):
        L = dfs[a]
        R = dfs[b]
        
        common_index = L.index.intersection(R.index)
        if common_index.empty:
            continue
            
        L_aligned = L.loc[common_index, cols]
        R_aligned = R.loc[common_index, cols]
        
        cross_df = L_aligned.divide(R_aligned).reset_index()
        cross_df.rename(columns={cross_df.columns[0]: "<DATE>"}, inplace=True)
        cross_df = cross_df.loc[:, ["<DATE>"] + cols]
        
        base = a.replace("USD", "")
        quote = b.replace("USD", "")
        result[f"{base}{quote}"] = cross_df

    return result


def close_check(df: pd.DataFrame) -> str:
    """Evaluates chronological directionality of the subset."""
    if df.empty:
        return 'equal'
    first_close = df.iloc[0]['<CLOSE>']
    last_close = df.iloc[-1]['<CLOSE>']

    if last_close > first_close:
        return 'higher'
    elif last_close < first_close:
        return 'lower'
    return 'equal'

def determine_direction_and_momentum(df_latest: pd.DataFrame) -> Tuple[str, str]:
    """Engine to parse true Direction and Momentum labels without Hurst,
    resolving lookback compression and MA crossover ambiguities.
    """
    if df_latest.empty:
        return "Unknown", "Unknown"
        
    current_close = df_latest['<CLOSE>'].values[0]
    
    # --- 1. FIX THE BOUNDARY BLIND SPOT ---
    # Instead of picking the min/max of everything, anchor to a reliable, fixed macro corridor.
    # We will use the 6-month (135 window) as our structural reference space.
    ceiling = df_latest['6m_High'].values[0]
    floor = df_latest['6m_Low'].values[0]
    
    total_range = ceiling - floor
    price_position = (current_close - floor) / total_range if total_range > 0 else 0.5
    
    if price_position > 0.85:
        direction = "High"
    elif price_position < 0.15:
        direction = "Low"
    else:
        direction = "Neutral"

    # --- 2. REGIME CLASSIFICATION BASED ON MA ALIGNMENT & SLOPES ---
    slope_cols = ['1m_slope', '3m_slope', '6m_slope', '12m_slope']
    crossover_cols = ['MA_1_3', 'MA_1_6', 'MA_1_12', 'MA_3_6', 'MA_3_12', 'MA_6_12']
    
    total_active_slopes = df_latest[slope_cols].sum(axis=1).values[0]
    total_active_crosses = df_latest[crossover_cols].sum(axis=1).values[0]
    
    # Check for true sequential trend alignment (e.g., 1m > 3m > 6m or vice versa)
    ma_1m = df_latest['1m_Mean'].values[0]
    ma_3m = df_latest['3m_Mean'].values[0]
    ma_6m = df_latest['6m_Mean'].values[0]
    
    is_trending_up = (ma_1m > ma_3m) and (ma_3m > ma_6m)
    is_trending_down = (ma_1m < ma_3m) and (ma_3m < ma_6m)
    ma_alignment = is_trending_up or is_trending_down

    # Trend Following: Slopes are active and MAs are cleanly stacked/fanning out (no crosses)
    if total_active_slopes >= 2 and total_active_crosses == 0 and ma_alignment:
        momentum = "Trend Following"
    # Mean Reversion: MAs are tangling/crossing, or price is extended to boundaries with flat slopes
    elif total_active_crosses >= 2 or (direction != "Neutral" and total_active_slopes == 0):
        momentum = "Mean Reversion"
    else:
        momentum = "Neutral"
        
    return direction, momentum

def analyse(name: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Executes structural sweeps, dividing into clean Macro and Micro sets."""
    df2 = Run(df, name)
    df_window = df2.tail(5).copy()
    label = close_check(df_window)
    
    df_latest = df_window.tail(1).copy()
    df_latest['Close_check'] = label

    # Dynamically derive the fixed structural dimensions
    direction, momentum = determine_direction_and_momentum(df_latest)
    df_latest['Direction'] = direction
    df_latest['Momentum'] = momentum
    
    # Calculate Total Filters across all active features to evaluate signal density
    filter_columns = [
        '1m_MA_Filter', '3m_MA_Filter', '6m_MA_Filter', '12m_MA_Filter', 'AT_MA_Filter',
        'MA_1_3', 'MA_1_6', 'MA_1_12', 'MA_1_AT', 'MA_3_6', 'MA_3_12', 'MA_6_12', 'MA_6_AT', 'MA_12_AT'
    ]
    df_latest['Total_Filters'] = df_latest[filter_columns].sum(axis=1)

    output_cols = ['Currency', 'Total_Filters', 'AT_High', 'AT_Low', '<CLOSE>', 'Close_check', 'Direction', 'Momentum']

    # --- Strong Set: Pure Macro Trend confirmation ---
    # Triggered when macro filters match a structural 'Trend Following' layout
    is_strong = (df_latest['Momentum'] == 'Trend Following') & (df_latest['Direction'] != 'Neutral')
    if is_strong.any():
        df_st = df_latest[output_cols].copy()
    else:
        df_st = pd.DataFrame(columns=output_cols)
    
    # --- Weak Set: Pure Overextended Mean Reversion ---
    # Triggered when micro crosses accumulate while price hits local limits
    # --- Weak Set: Pure Overextended Mean Reversion ---
    # FIX: Wrapped (df_latest['Total_Filters'] >= 3) and (df_latest['Momentum'] == 'Neutral') in explicit parentheses
    is_weak = (df_latest['Momentum'] == 'Mean Reversion') | ((df_latest['Total_Filters'] >= 3) & (df_latest['Momentum'] == 'Neutral'))
    if is_weak.any():
        df_wk = df_latest[output_cols].copy()
    else:
        df_wk = pd.DataFrame(columns=output_cols)
    
    return df_st, df_wk


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
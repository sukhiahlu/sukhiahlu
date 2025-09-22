# Import the libraries
import pyautogui
import time
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import os
import itertools

from datetime import date

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

#Checks
# df_wk['12m_MA_Filter'].value_counts()
# df_st['12m_MA_Filter'].value_counts()
# df_wk.tail()

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

        # reset <DATE> as column and keep exact column order
    #     cross_df = cross_df.reset_index().rename_axis(None)
    #     cross_df = cross_df[["<DATE>"] + cols]

        result[cross_name] = cross_df

    # print(result.keys())
    # print(result['EURAUD'])
    return(result)

#Analyse the data using above functions
def analyse(name, df):
    df2 = Run(df,name)
    df2 = df2.tail()

    # Strong set: 4 filters
    # - Close is near AT and/or 12m H/L then time could be to reverse - Strong
    # - Rolling MA crosses comparisons for 12 with various and AT
    df_st = Filter(df2, ['AT_MA_Filter', '12m_MA_Filter'
                        ,'MA_6_AT','MA_12_AT'])
    
    df_st = df_st [['Currency', 'Total_Filters']].sort_values(by='Total_Filters', ascending=False).reset_index(drop=True)
    df_st = df_st.drop_duplicates(subset=['Currency']) #, 'Total_Filters'
    
    # Weak set: 11 filters
    # - Close is near other H/L then time could be to reverse - Weak
    # - Rolling MA crosses comparisons for 1 with various and 3/6
    # - Slopes for sudden jumps
    df_wk = Filter(df2, ['1m_MA_Filter','3m_MA_Filter','6m_MA_Filter'
                         ,'MA_1_3', 'MA_1_6', 'MA_1_12', 'MA_1_AT','MA_3_6'
                        ,'1m_slope','3m_slope','6m_slope'])
    
    df_wk = df_wk [['Currency', 'Total_Filters']].sort_values(by='Total_Filters', ascending=False).reset_index(drop=True)   
    df_wk = df_wk.drop_duplicates(subset=['Currency'])
    
    return df_st,df_wk
import numpy as np
import pandas as pd
import warnings

# Suppress runtime warnings from NaNs during warmup
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================================
# MASTER SIGNAL VOTING ENGINE
# =========================================================================

def build_confluence_matrix(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Consumes raw Daily OHLCV data. Generates cross-timeframe indicators
    and aggregates them into an unweighted 3-column voting structure.
    """
    df = df_daily.copy().sort_values("<DATE>").reset_index(drop=True)
    df["<DATE>"] = pd.to_datetime(df["<DATE>"])
    
    # -----------------------------------------------------------------
    # A. DAILY TIMEFRAME INDICATORS
    # -----------------------------------------------------------------
    # Trend MAs
    for w in [10, 20, 50, 100, 200]:
        df[f"D_MA_{w}"] = df["<CLOSE>"].rolling(window=w).mean()
        
    # Daily MACD (12, 26, 9)
    df["D_EMA_12"] = df["<CLOSE>"].ewm(span=12, adjust=False).mean()
    df["D_EMA_26"] = df["<CLOSE>"].ewm(span=26, adjust=False).mean()
    df["D_MACD"] = df["D_EMA_12"] - df["D_EMA_26"]
    df["D_MACD_Signal"] = df["D_MACD"].ewm(span=9, adjust=False).mean()
    
    # Daily Oscillators (RSI 9 & 14, Stochastics)
    for w in [9, 14]:
        delta = df["<CLOSE>"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=w).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=w).mean()
        df[f"D_RSI_{w}"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
        
    low_14 = df["<LOW>"].rolling(window=14).min()
    high_14 = df["<HIGH>"].rolling(window=14).max()
    df["D_Stoch_K"] = ((df["<CLOSE>"] - low_14) / (high_14 - low_14 + 1e-9)) * 100
    df["D_Stoch_D"] = df["D_Stoch_K"].rolling(window=3).mean()
    
    # Daily Volume (OBV, CMF)
    df["D_OBV"] = (np.sign(df["<CLOSE>"].diff()).fillna(0) * df["<TICKVOL>"]).cumsum()
    df["D_OBV_EMA"] = df["D_OBV"].ewm(span=20, adjust=False).mean()
    
    mf_mult = ((df["<CLOSE>"] - df["<LOW>"]) - (df["<HIGH>"] - df["<CLOSE>"])) / (df["<HIGH>"] - df["<LOW>"] + 1e-9)
    df["D_CMF_20"] = (mf_mult * df["<TICKVOL>"]).rolling(window=20).sum() / (df["<TICKVOL>"].rolling(window=20).sum() + 1e-9)
    
    # Daily Volatility Structure (Bollinger Bands)
    df["D_BB_Mid"] = df["<CLOSE>"].rolling(window=20).mean()
    df["D_BB_Std"] = df["<CLOSE>"].rolling(window=20).std()
    df["D_BB_Upper"] = df["D_BB_Mid"] + (2 * df["D_BB_Std"])
    df["D_BB_Lower"] = df["D_BB_Mid"] - (2 * df["D_BB_Std"])
    
    # -----------------------------------------------------------------
    # B. SAFE WEEKLY RESAMPLING LAYER (ZERO LOOKAHEAD LEAKAGE)
    # -----------------------------------------------------------------
    df_temp = df.set_index("<DATE>")
    weekly = df_temp.resample("W").agg({
        "<OPEN>": "first", "<HIGH>": "max", "<LOW>": "min", "<CLOSE>": "last", "<TICKVOL>": "sum"
    }).dropna()
    
    # Weekly Trend
    for w in [20, 50, 200]:
        weekly[f"W_MA_{w}"] = weekly["<CLOSE>"].rolling(window=w).mean()
        
    weekly["W_EMA_12"] = weekly["<CLOSE>"].ewm(span=12, adjust=False).mean()
    weekly["W_EMA_26"] = weekly["<CLOSE>"].ewm(span=26, adjust=False).mean()
    weekly["W_MACD"] = weekly["W_EMA_12"] - weekly["W_EMA_26"]
    weekly["W_MACD_Signal"] = weekly["W_MACD"].ewm(span=9, adjust=False).mean()
    
    # Weekly Oscillators
    w_delta = weekly["<CLOSE>"].diff()
    w_gain = (w_delta.where(w_delta > 0, 0)).rolling(window=14).mean()
    w_loss = (-w_delta.where(w_delta < 0, 0)).rolling(window=14).mean()
    weekly["W_RSI_14"] = 100 - (100 / (1 + (w_gain / (w_loss + 1e-9))))
    
    w_low_14 = weekly["<LOW>"].rolling(window=14).min()
    w_high_14 = weekly["<HIGH>"].rolling(window=14).max()
    weekly["W_Stoch_K"] = ((weekly["<CLOSE>"] - w_low_14) / (w_high_14 - w_low_14 + 1e-9)) * 100
    weekly["W_Stoch_D"] = weekly["W_Stoch_K"].rolling(window=3).mean()
    
    # Weekly Volume & Structural Volatility
    weekly["W_OBV"] = (np.sign(weekly["<CLOSE>"].diff()).fillna(0) * weekly["<TICKVOL>"]).cumsum()
    weekly["W_OBV_EMA"] = weekly["W_OBV"].ewm(span=20, adjust=False).mean()
    
    w_mf_mult = ((weekly["<CLOSE>"] - weekly["<LOW>"]) - (weekly["<HIGH>"] - weekly["<CLOSE>"])) / (weekly["<HIGH>"] - weekly["<LOW>"] + 1e-9)
    weekly["W_CMF_20"] = (w_mf_mult * weekly["<TICKVOL>"]).rolling(window=20).sum() / (weekly["<TICKVOL>"].rolling(window=20).sum() + 1e-9)
    
    weekly["W_BB_Mid"] = weekly["<CLOSE>"].rolling(window=20).mean()
    weekly["W_BB_Std"] = weekly["<CLOSE>"].rolling(window=20).std()
    weekly["W_BB_Upper"] = weekly["W_BB_Mid"] + (2 * weekly["W_BB_Std"])
    weekly["W_BB_Lower"] = weekly["W_BB_Mid"] - (2 * weekly["W_BB_Std"])
    
    # Shift completed weekly data forward by 1 week to align safely
    weekly_signals = weekly.shift(1)
    weekly_signals["Year_Week"] = weekly_signals.index.to_period("W")
    
    df["Year_Week"] = df["<DATE>"].dt.to_period("W")
    df = df.merge(weekly_signals, on="Year_Week", how="left", suffixes=("", "_W"))
    df = df.drop(columns=["Year_Week"])
    
    # -----------------------------------------------------------------
    # C. VECTORIZED SIGNAL MAPPING (Outputting -1, 0, or 1)
    # -----------------------------------------------------------------
    signal_cols = []

    # --- 1. TREND MATRIX ---
    df["Sig_T_D10"]   = np.where(df["<CLOSE>"] > df["D_MA_10"],   1, -1)
    df["Sig_T_D20"]   = np.where(df["<CLOSE>"] > df["D_MA_20"],   1, -1)
    df["Sig_T_D50"]   = np.where(df["<CLOSE>"] > df["D_MA_50"],   1, -1)
    df["Sig_T_D100"]  = np.where(df["<CLOSE>"] > df["D_MA_100"],  1, -1)
    df["Sig_T_D200"]  = np.where(df["<CLOSE>"] > df["D_MA_200"],  1, -1)
    df["Sig_T_DMACD"] = np.where(df["D_MACD"]  > df["D_MACD_Signal"], 1, -1)
    
    df["Sig_T_W20"]   = np.where(df["<CLOSE>_W"] > df["W_MA_20"],   1, -1)
    df["Sig_T_W50"]   = np.where(df["<CLOSE>_W"] > df["W_MA_50"],   1, -1)
    df["Sig_T_W200"]  = np.where(df["<CLOSE>_W"] > df["W_MA_200"],  1, -1)
    df["Sig_T_WMACD"] = np.where(df["W_MACD"]   > df["W_MACD_Signal"], 1, -1)
    
    signal_cols.extend(["Sig_T_D10", "Sig_T_D20", "Sig_T_D50", "Sig_T_D100", "Sig_T_D200", 
                        "Sig_T_DMACD", "Sig_T_W20", "Sig_T_W50", "Sig_T_W200", "Sig_T_WMACD"])

    # --- 2. MOMENTUM MATRIX ---
    df["Sig_M_RSI14_D"] = np.where(df["D_RSI_14"] < 30, 1, np.where(df["D_RSI_14"] > 70, -1, 0))
    df["Sig_M_RSI9_D"]  = np.where(df["D_RSI_9"] < 25, 1, np.where(df["D_RSI_9"] > 75, -1, 0))
    df["Sig_M_Stoch_D"] = np.where((df["D_Stoch_K"] < 20) & (df["D_Stoch_K"] > df["D_Stoch_D"]), 1, 
                                   np.where((df["D_Stoch_K"] > 80) & (df["D_Stoch_K"] < df["D_Stoch_D"]), -1, 0))
    
    df["Sig_M_RSI14_W"] = np.where(df["W_RSI_14"] < 35, 1, np.where(df["W_RSI_14"] > 65, -1, 0))
    df["Sig_M_Stoch_W"] = np.where((df["W_Stoch_K"] < 25) & (df["W_Stoch_K"] > df["W_Stoch_D"]), 1,
                                   np.where((df["W_Stoch_K"] > 75) & (df["W_Stoch_K"] < df["W_Stoch_D"]), -1, 0))
    
    signal_cols.extend(["Sig_M_RSI14_D", "Sig_M_RSI9_D", "Sig_M_Stoch_D", "Sig_M_RSI14_W", "Sig_M_Stoch_W"])

    # --- 3. VOLUME MATRIX ---
    df["Sig_V_OBV_D"] = np.where(df["D_OBV"] > df["D_OBV_EMA"], 1, -1)
    df["Sig_V_CMF_D"] = np.where(df["D_CMF_20"] > 0.05, 1, np.where(df["D_CMF_20"] < -0.05, -1, 0))
    df["Sig_V_OBV_W"] = np.where(df["W_OBV"] > df["W_OBV_EMA"], 1, -1)
    df["Sig_V_CMF_W"] = np.where(df["W_CMF_20"] > 0.05, 1, np.where(df["W_CMF_20"] < -0.05, -1, 0))
    
    signal_cols.extend(["Sig_V_OBV_D", "Sig_V_CMF_D", "Sig_V_OBV_W", "Sig_V_CMF_W"])

    # --- 4. VOLATILITY MATRIX ---
    df["Sig_S_BB_D"] = np.where(df["<CLOSE>"] <= df["D_BB_Lower"], 1, np.where(df["<CLOSE>"] >= df["D_BB_Upper"], -1, 0))
    df["Sig_S_BB_W"] = np.where(df["<CLOSE>_W"] <= df["W_BB_Lower"], 1, np.where(df["<CLOSE>_W"] >= df["W_BB_Upper"], -1, 0))
    
    signal_cols.extend(["Sig_S_BB_D", "Sig_S_BB_W"])

    # -----------------------------------------------------------------
    # D. UNWEIGHTED CONFLUENCE AGGREGATION
    # -----------------------------------------------------------------
    # Stack the signals matrix to perform simple vectorized summing
    matrix_signals = df[signal_cols].values
    
    df["COUNT_LONG"]  = np.sum(matrix_signals == 1, axis=1)
    df["COUNT_HOLD"]  = np.sum(matrix_signals == 0, axis=1)
    df["COUNT_SHORT"] = np.sum(matrix_signals == -1, axis=1)
    
    # Dynamic Warmup Drop
    return df.dropna(subset=["D_MA_200", "W_MA_200"]).reset_index(drop=True)[
        ["<DATE>", "<CLOSE>", "COUNT_LONG", "COUNT_HOLD", "COUNT_SHORT"]
    ]
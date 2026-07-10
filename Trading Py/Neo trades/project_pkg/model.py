import numpy as np
import pandas as pd
import warnings
import datetime
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore", category=UserWarning)

# =====================================================================
# 0. GLOBAL FEATURE ENCYCLOPEDIA (MASTER DAILY DEFINITIONS)
# =====================================================================
FEATURE_ENCYCLOPEDIA = {
    "returns": "Log price change over previous session - close momentum",
    "volatility_5": "5-day rolling standard dev - short-term asset risk",
    "HL_spread": "Intraday high-to-low spread - volatility/liquidity stress",
    "CO_spread": "Intraday close-to-open - overnight vs. intraday gap risk",
    "dist_MA_5": "% distance from 5-day ma - short-term mean-reversion",
    "dist_MA_20": "% distance from 20-day ma - medium-term trend extension",
    "volume_ratio": "Current volume vs. 20-day avg - institution participation",
    "RSI": "Relative Strength Index - overextended buyers or sellers.",
    "Stochastic_14": "14-day Stochastic Oscillator - close & recent high-low range",
    "NTR_5": "Normalized True Range over 5 days - structural volatility",
    "ROC_10": "Rate of Change over 10 days - velocity of macro price shifts",
    "vol_acceleration": "Rate of change in short-term volatility - regime shifts",
    "price_zscore_50": "50-day rolling price Z-Score - extremeness of current price.",
    "volume_spread_ratio": "Volume ratio * high-low spread - high-volume liquidity traps."
}

# =====================================================================
# 1. CORE MACHINE LEARNING ENGINE & TECHNICAL INDICATORS
# =====================================================================

def calculate_technical_indicators(df):
    """Calculates 14 normalized, scale-invariant technical indicators daily."""
    df = df.copy()
    
    # 1. Base Returns and Volatility
    df["returns"] = np.log(df["<CLOSE>"] / df["<CLOSE>"].shift(1))
    df["volatility_5"] = df["returns"].rolling(window=5).std()
    
    # 2. Scale-Invariant Price Action Spreads
    df["HL_spread"] = (df["<HIGH>"] - df["<LOW>"]) / df["<CLOSE>"]
    df["CO_spread"] = (df["<CLOSE>"] - df["<OPEN>"]) / df["<OPEN>"]    
    
    # 3. Trend Proximity
    ma_5 = df["<CLOSE>"].rolling(window=5).mean()
    ma_20 = df["<CLOSE>"].rolling(window=20).mean()
    df["dist_MA_5"] = (df["<CLOSE>"] - ma_5) / ma_5
    df["dist_MA_20"] = (df["<CLOSE>"] - ma_20) / ma_20
    
    # 4. Liquidity Dynamics
    df["volume_ratio"] = df["<TICKVOL>"] / (df["<TICKVOL>"].rolling(window=20).mean() + 1e-9)
    df["volume_ratio"] = df["volume_ratio"].fillna(1.0).replace([np.inf, -np.inf], 1.0)
    
    # 5. Bounded Momentum: RSI
    delta = df["<CLOSE>"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # 6. Bounded Structural Position: Stochastic Oscillator
    lowest_low_14 = df["<LOW>"].rolling(window=14).min()
    highest_high_14 = df["<HIGH>"].rolling(window=14).max()
    df["Stochastic_14"] = ((df["<CLOSE>"] - lowest_low_14) / (highest_high_14 - lowest_low_14 + 1e-9)) * 100

    # 7. Scale-Invariant Volatility: NTR
    prev_close = df["<CLOSE>"].shift(1)
    tr1 = df["<HIGH>"] - df["<LOW>"]
    tr2 = (df["<HIGH>"] - prev_close).abs()
    tr3 = (df["<LOW>"] - prev_close).abs()
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    df["NTR_5"] = true_range.rolling(window=5).mean() / df["<CLOSE>"]

    # 8. Acceleration Momentum: Rate of Change (ROC 10)
    df["ROC_10"] = ((df["<CLOSE>"] - df["<CLOSE>"].shift(10)) / df["<CLOSE>"].shift(10)) * 100
    
    # --- ADVANCED REGIME FEATURES ---
    # 9. Volatility Acceleration
    df["vol_acceleration"] = df["volatility_5"].diff() / (df["volatility_5"].shift(1) + 1e-9)
    df["vol_acceleration"] = df["vol_acceleration"].fillna(0.0).replace([np.inf, -np.inf], 0.0)

    # 10. Extreme Price Deviations (Rolling 50-day Z-Score)
    ma_50 = df["<CLOSE>"].rolling(window=50).mean()
    std_50 = df["<CLOSE>"].rolling(window=50).std()
    df["price_zscore_50"] = (df["<CLOSE>"] - ma_50) / (std_50 + 1e-9)
    df["price_zscore_50"] = df["price_zscore_50"].fillna(0.0)

    # 11. Volume-to-Spread Discrepancy
    df["volume_spread_ratio"] = df["volume_ratio"] * df["HL_spread"]
    
    # 12. Realized Volatility Smoothing for Target Normalization
    df["realized_vol_20"] = df["returns"].rolling(window=20).std()  # Removed .bfill() to prevent data leakage
    df["realized_vol_20_smoothed"] = df["realized_vol_20"].ewm(span=5, min_periods=1).mean()
    
    return df

# =====================================================================
# 2. RISK PERFORMANCE EVALUATION METRICS
# =====================================================================

def calculate_sharpe_ratio(returns):
    if len(returns) < 2 or np.std(returns) == 0: return 0.0
    return (np.mean(returns) / np.std(returns)) * np.sqrt(252)

def calculate_sortino_ratio(returns):
    if len(returns) < 2: return 0.0
    returns = np.array(returns)
    downside = returns[returns < 0]
    if len(downside) == 0 or np.std(downside) == 0: return 0.0
    return (np.mean(returns) / np.std(downside)) * np.sqrt(252)

def calculate_max_drawdown(returns):
    if len(returns) == 0: return 0.0
    equity = np.exp(np.cumsum(returns))
    return np.min((equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity))

# =====================================================================
# 3. SYNTHETIC FLAVOR GENERATORS (DAILY CONSTRAINED STRESS TESTING)
# =====================================================================

def reconstruct_ohlc_safely(df, new_close, volatility_expansion=1.0):
    df = df.copy()
    raw_co_ratio = df["<OPEN>"] / df["<CLOSE>"]
    raw_h_ratio = df["<HIGH>"] / np.maximum(df["<OPEN>"], df["<CLOSE>"])
    raw_l_ratio = df["<LOW>"] / np.minimum(df["<OPEN>"], df["<CLOSE>"])
    
    expanded_h_ratio = 1.0 + ((raw_h_ratio - 1.0) * volatility_expansion)
    expanded_l_ratio = 1.0 - ((1.0 - raw_l_ratio) * volatility_expansion)
    
    df["<CLOSE>"] = new_close
    df["<OPEN>"] = df["<CLOSE>"] * raw_co_ratio
    df["<HIGH>"] = np.maximum(df["<OPEN>"], df["<CLOSE>"]) * expanded_h_ratio
    df["<LOW>"] = np.minimum(df["<OPEN>"], df["<CLOSE>"]) * expanded_l_ratio
    return df

def generate_flavor_1_liquidity_black_hole(df, start_idx=400, window=15, severity=0.15, volume_spike=4.0):
    df_syn = df.copy()
    close_array = df_syn["<CLOSE>"].values.copy()
    open_array = df_syn["<OPEN>"].values.copy()
    high_array = df_syn["<HIGH>"].values.copy()
    low_array = df_syn["<LOW>"].values.copy()
    
    prev_close = close_array[start_idx - 1]
    open_array[start_idx] = prev_close * (1.0 - severity)
    close_array[start_idx] = open_array[start_idx] * 0.97 
    high_array[start_idx] = prev_close * 1.05  
    low_array[start_idx] = close_array[start_idx] * 0.90
    df_syn.loc[start_idx, "<TICKVOL>"] = int(df_syn.loc[start_idx, "<TICKVOL>"] * volume_spike)

    for idx, t in enumerate(range(start_idx + 1, min(start_idx + window, len(df)))):
        recovery_factor = 1.0 + (0.05 * np.log(idx + 1))  
        close_array[t] = close_array[t] * recovery_factor
        open_array[t] = open_array[t] * recovery_factor
        high_array[t] = high_array[t] * recovery_factor * 1.03
        low_array[t] = low_array[t] * recovery_factor * 0.97
        df_syn.loc[t, "<TICKVOL>"] = int(df_syn.loc[t, "<TICKVOL>"] * 1.5)

    df_syn["<OPEN>"] = open_array; df_syn["<HIGH>"] = high_array; df_syn["<LOW>"] = low_array; df_syn["<CLOSE>"] = close_array
    return df_syn

def generate_flavor_2_structural_bear_market(df, start_idx=300, daily_drift=0.004, noise_scale=0.012):
    df_syn = df.copy()
    close_array = df_syn["<CLOSE>"].values.copy()
    for t in range(start_idx, len(df)):
        bull_trap = 0.04 if np.random.rand() > 0.95 else 0.0
        shock_return = -daily_drift + (noise_scale * np.random.normal(0, 1)) + bull_trap
        close_array[t] = close_array[t-1] * np.exp(shock_return)
    return reconstruct_ohlc_safely(df_syn, close_array, volatility_expansion=1.5)

def generate_flavor_3_high_volatility_grind(df, speed_of_reversion=0.15, volatility=0.02):
    df_syn = df.copy()
    close_array = df_syn["<CLOSE>"].values.copy()
    anchor_price = close_array[0]
    for t in range(1, len(df)):
        dx = speed_of_reversion * (np.log(anchor_price) - np.log(close_array[t-1])) + volatility * np.random.normal(0, 1)
        close_array[t] = close_array[t-1] * np.exp(dx)
        df_syn.loc[t, "<TICKVOL>"] = int(df_syn.loc[t, "<TICKVOL>"] * 1.5)
    return reconstruct_ohlc_safely(df_syn, close_array, volatility_expansion=1.3)

def generate_flavor_4_the_gold_rush(df, start_idx=500, peak_idx=750, acceleration=1.4, bubble_pop_rate=0.4):
    df_syn = df.copy()
    close_array = df_syn["<CLOSE>"].values.copy()
    for t in range(start_idx, peak_idx):
        progress = (t - start_idx) / (peak_idx - start_idx)
        hype_boost = 0.05 * (progress ** acceleration)
        close_array[t] = close_array[t] * (1.0 + hype_boost)
    for idx, t in enumerate(range(peak_idx, len(df))):
        crash_decay = np.exp(-bubble_pop_rate * idx)
        close_array[t] = close_array[peak_idx] * (0.4 + 0.6 * crash_decay)
    return reconstruct_ohlc_safely(df_syn, close_array, volatility_expansion=1.8)

# =====================================================================
# 4. DAILY WALK-FORWARD REGRESSION VALIDATION ENGINE
# =====================================================================

def run_walk_forward_validation_tuned(df, train_window, retrain_interval=100):
    """Executes walk-forward validation across the history."""
    today_str = datetime.datetime.now().strftime("%Y.%m.%d")
    df = df[df["<DATE>"] != today_str].copy().sort_values("<DATE>").reset_index(drop=True)
    
    df = calculate_technical_indicators(df)
    df["future_raw_returns"] = df["returns"].shift(-1)
    df["future_vol_adjusted_target"] = df["future_raw_returns"] / (df["realized_vol_20"] + 1e-9)
    
    feature_cols = list(FEATURE_ENCYCLOPEDIA.keys())
    df = df.dropna(subset=feature_cols + ["future_vol_adjusted_target"]).reset_index(drop=True)

    out_of_sample_preds = []
    actuals_all = []
    strategy_returns = []
    benchmark_returns = []
    importance_accumulator = np.zeros(len(feature_cols))
    
    embargo_window = 15  # Fixed: Widened to 15 to cut out overlapping feature serial correlation
    prev_pos = 0.0
    friction_bps = 0.00015  
    
    CONVICTION_THRESHOLD = 0.05
    LEVERAGE_MULTIPLIER = 1.0

    for end_train_idx in range(train_window, len(df), retrain_interval):
        X_train_df = df.iloc[0 : end_train_idx - embargo_window].copy()
        if len(X_train_df) < 100: continue
            
        end_test_idx = min(end_train_idx + retrain_interval, len(df))
        test_block = df.iloc[end_train_idx : end_test_idx].copy()
        if len(test_block) == 0: break

        train_mean = X_train_df["future_vol_adjusted_target"].mean()
        train_std = X_train_df["future_vol_adjusted_target"].std() + 1e-9
        X_train_df["target"] = (X_train_df["future_vol_adjusted_target"] - train_mean) / train_std

        model = LGBMRegressor(
            objective="regression", n_estimators=50, learning_rate=0.05, 
            max_depth=4, min_child_samples=20, reg_alpha=5.0, reg_lambda=5.0,
            random_state=42, verbose=-1, colsample_bytree=0.7
        )
        model.fit(X_train_df[feature_cols], X_train_df["target"])
        importance_accumulator += model.booster_.feature_importance(importance_type="gain")

        p_z = model.predict(test_block[feature_cols])
        pred_log_returns = (p_z * train_std + train_mean) * test_block["realized_vol_20_smoothed"]
        actual_log_returns = test_block["future_raw_returns"]
        
        out_of_sample_preds.extend(pred_log_returns)
        actuals_all.extend(actual_log_returns)
        
        for p, act_l, smooth_vol in zip(pred_log_returns, actual_log_returns, test_block["realized_vol_20_smoothed"]):
            if abs(p) < (smooth_vol * CONVICTION_THRESHOLD):
                pos = 0.0
            else:
                pos = np.clip((p / (smooth_vol + 1e-9)) * LEVERAGE_MULTIPLIER, -1.0, 1.0)
            
            turnover = abs(pos - prev_pos)
            trade_drag = friction_bps * turnover
            r_strat = act_l * pos - trade_drag
            
            strategy_returns.append(r_strat)
            benchmark_returns.append(act_l)
            prev_pos = pos

    mae = mean_absolute_error(actuals_all, out_of_sample_preds) if actuals_all else 0.0
    sharpe = calculate_sharpe_ratio(strategy_returns)
    sortino = calculate_sortino_ratio(strategy_returns)
    max_dd = calculate_max_drawdown(strategy_returns)
    b_sharpe = calculate_sharpe_ratio(benchmark_returns)
    
    total_gain = np.sum(importance_accumulator) if np.sum(importance_accumulator) > 0 else 1
    feature_importance_dict = dict(zip(feature_cols, (importance_accumulator / total_gain) * 100))
    ranked_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    return mae, sharpe, sortino, max_dd, b_sharpe, ranked_features

# =====================================================================
# 5. LIVE PRODUCTION ENVIRONMENT INTERACTIVE SCENARIO ENGINE
# =====================================================================

def predict_next_day_scenarios(df, train_window=300):
    """
    Fits the regression model to historical data, simulates 6 discrete
    intraday development scenarios, and prints a decision matrix including 
    allocation units, volatility-adjusted Take Profit (TP), and Stop Loss (SL) targets.
    Includes the multi-metric weekly health matrix and structural regime cockpit.
    """
    feature_cols = list(FEATURE_ENCYCLOPEDIA.keys())
    
    # 1. Clean and separate complete history
    today_str = datetime.datetime.now().strftime("%Y.%m.%d")
    df_clean = df[df["<DATE>"] != today_str].copy().sort_values("<DATE>").reset_index(drop=True)
    
    latest_close = df_clean.iloc[-1]["<CLOSE>"]
    recent_vol = df_clean["<TICKVOL>"].tail(5).mean()
    base_vol = df_clean["<TICKVOL>"].tail(20).mean() + 1e-9
    
    # --- ADDED: WEEKLY HIGHER-LEVEL INDICATORS ---
    # [METRIC 1] Structural Trend: 200-Day Simple Moving Average
    sma_200 = df_clean["<CLOSE>"].tail(200).mean()
    trend_status = "ABOVE (Bullish Structure)" if latest_close > sma_200 else "BELOW (Bearish Structure)"
    
    # [METRIC 2] Macro Momentum: 50-Day Exponential Moving Average
    ema_50 = df_clean["<CLOSE>"].ewm(span=50, adjust=False).mean().iloc[-1]
    mom_status = "ABOVE (Macro Momentum Intact)" if latest_close > ema_50 else "BELOW (Short-Term Macro Fatigue)"
    
    # [METRIC 3] Institution Volume: Current rolling 5-day volume vs last 20 days
    vol_pct_change = ((recent_vol - base_vol) / base_vol) * 100
    vol_status = f"{vol_pct_change:+.1f}% Spike vs 20-Day Avg"
    
    # [METRIC 4] Regime Volatility: True Range Over Last 5 Days Normalized vs 20-Day Distribution
    highs = df_clean["<HIGH>"].tail(20).values
    lows = df_clean["<LOW>"].tail(20).values
    closes = df_clean["<CLOSE>"].tail(21).values
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - closes[:-1]), np.abs(lows - closes[:-1])))
    atr_5_now = np.mean(tr[-5:]) / latest_close
    atr_history = [np.mean(tr[i:i+5]) / closes[i+5] for i in range(15)]
    pct_rank = (sum(1 for x in atr_history if atr_5_now > x) / max(len(atr_history), 1)) * 100
    vol_env = f"{pct_rank:.1f}th Percentile (Historical Scale)"

    print("\n" + "="*136)
    print("                                           WEEKLY MACRO HEALTH CHECK MATRIX                                             ")
    print("="*136)
    print(f" [STRUCTURAL TREND]     Close vs 200-Day MA (40-Week Proxy)  : {trend_status}")
    print(f" [MACRO MOMENTUM]       Close vs 50-Day EMA (10-Week Proxy)  : {mom_status}")
    print(f" [INSTITUTION VOLUME]   Recent 5-Day Volume Magnitude        : {vol_status}")
    print(f" [REGIME VOLATILITY]    5-Day ATR Historical Percentile      : {vol_env}")
    print("="*136)

    # 2. Extract Base Features & Cutoff Boundaries
    df_features = calculate_technical_indicators(df_clean)
    latest_known_row = df_features.dropna(subset=feature_cols).iloc[-1]
    
    cutoff_date = latest_known_row["<DATE>"]
    atr_5 = latest_known_row["NTR_5"] * latest_close
    
    # 3. Train the pipeline up to the current cutoff point
    train_df = df_features.dropna(subset=feature_cols + ["returns"]).reset_index(drop=True)
    train_df["future_raw_returns"] = train_df["returns"].shift(-1)
    train_df["future_vol_adjusted_target"] = train_df["future_raw_returns"] / (train_df["realized_vol_20"] + 1e-9)
    train_df = train_df.dropna(subset=["future_vol_adjusted_target"])
    
    X_train_df = train_df.tail(int(train_window)).copy()
    train_mean = X_train_df["future_vol_adjusted_target"].mean()
    train_std = X_train_df["future_vol_adjusted_target"].std() + 1e-9
    X_train_df["target"] = (X_train_df["future_vol_adjusted_target"] - train_mean) / train_std

    model = LGBMRegressor(
        objective="regression", n_estimators=50, learning_rate=0.05, 
        max_depth=4, min_child_samples=20, reg_alpha=5.0, reg_lambda=5.0,
        random_state=42, verbose=-1, colsample_bytree=0.7
    )
    model.fit(X_train_df[feature_cols], X_train_df["target"])
    
    # 4. Formulate the 6 Intraday Structural Scenarios
    sim_open = latest_close
    tickvol_ref = latest_known_row["<TICKVOL>"]
    
    scenarios = {
        "1. Pure Positive Breakout":       {"C": sim_open + (0.6 * atr_5), "H": sim_open + (0.7 * atr_5), "L": sim_open - (0.1 * atr_5)},
        "2. Positive + Upper Wick Drop":   {"C": sim_open + (0.1 * atr_5), "H": sim_open + (0.8 * atr_5), "L": sim_open - (0.1 * atr_5)},
        "3. Pure Negative Breakdown":      {"C": sim_open - (0.6 * atr_5), "H": sim_open + (0.1 * atr_5), "L": sim_open - (0.7 * atr_5)},
        "4. Negative + Lower Wick Sweep":  {"C": sim_open - (0.1 * atr_5), "H": sim_open + (0.1 * atr_5), "L": sim_open - (0.8 * atr_5)},
        "5. Compressed Neutral Chop":      {"C": sim_open + (0.0 * atr_5), "H": sim_open + (0.1 * atr_5), "L": sim_open - (0.1 * atr_5)},
        "6. High-Vol Indecision Cross":    {"C": sim_open + (0.0 * atr_5), "H": sim_open + (0.7 * atr_5), "L": sim_open - (0.7 * atr_5)}
    }
    
    print("\n" + "="*130)
    print("                                   LIVE DAILY DEVELOPMENT SCENARIO MATRIX WITH RISK TARGETS                                   ")
    print("="*130)
    print(f" Reference Price Close: {latest_close:.5f}  │  Current Daily Volatility Context (ATR_5): {atr_5:.5f}")
    print(f" Last Closed Session:   {cutoff_date}")
    print("─" * 130)
    print(f" {'STRUCTURAL DEVELOPMENT SCENARIO':<32} │ {'SIM CLOSE':<10} │ {'MODEL VECTOR':<12} │ {'TACTICAL VECTOR':<18} │ {'TAKE PROFIT':<12} │ {'STOP LOSS'}")
    print("─" * 130)
    
    for name, levels in scenarios.items():
        sim_row = pd.DataFrame([{
            "<DATE>": "TOMORROW_SIM", "<OPEN>": sim_open, "<HIGH>": levels["H"], "<LOW>": levels["L"], "<CLOSE>": levels["C"], "<TICKVOL>": tickvol_ref
        }])
        
        # Scenario execution note: Remember our structural lookahead fix should be implemented when ready!
        extended_df = pd.concat([df_clean, sim_row], ignore_index=True)
        extended_features = calculate_technical_indicators(extended_df)
        
        sim_vector = extended_features.iloc[[-1]]
        smooth_vol = sim_vector["realized_vol_20_smoothed"].values[0]
        
        raw_pred_z = model.predict(sim_vector[feature_cols])[0]
        pred_log_return = (raw_pred_z * train_std + train_mean) * smooth_vol
        nominal_pct = (np.exp(pred_log_return) - 1) * 100
        
        if abs(pred_log_return) < (smooth_vol * 0.05):
            action = "STANDBY (No Edge)"
            tp_str = "  N/A"
            sl_str = "  N/A"
        else:
            alloc_unit = np.clip((pred_log_return / smooth_vol) * 1.0, -1.0, 1.0)
            if alloc_unit > 0:
                action = f"BUY LONG  {alloc_unit:+.2f}x"
                tp_str = f"{levels['C'] + (1.5 * atr_5):<12.5f}"
                sl_str = f"{levels['C'] - (1.0 * atr_5):.5f}"
            else:
                action = f"SELL SHORT {alloc_unit:.2f}x"
                tp_str = f"{levels['C'] - (1.5 * atr_5):<12.5f}"
                sl_str = f"{levels['C'] + (1.0 * atr_5):.5f}"
            
        print(f" {name:<32} │ {levels['C']:<10.5f} │ {nominal_pct:+.4f}%    │ {action:<18} │ {tp_str} │ {sl_str}")
        
    print("="*130)
    
    # --- ADDED: DYNAMIC 4-METRIC DATA-DRIVEN REGIME COCKPIT ---
    print("\n" + " " * 42 + "DYNAMIC 4-METRIC DATA-DRIVEN REGIME COCKPIT")
    print("-" * 130)
    
    hist_sma200 = df_clean["<CLOSE>"].rolling(window=200).mean()
    hist_sma200_std = df_clean["<CLOSE>"].rolling(window=200).std() + 1e-9
    hist_trend_z = (df_clean["<CLOSE>"] - hist_sma200) / hist_sma200_std
    
    hist_ema50 = df_clean["<CLOSE>"].ewm(span=50, adjust=False).mean()
    hist_mom_dist = (df_clean["<CLOSE>"] - hist_ema50) / (hist_ema50 + 1e-9)
    
    hist_vol5 = df_clean["<TICKVOL>"].rolling(window=5).mean()
    hist_vol20 = df_clean["<TICKVOL>"].rolling(window=20).mean() + 1e-9
    hist_vol_spike = (hist_vol5 - hist_vol20) / hist_vol20
    
    hist_highs = df_clean["<HIGH>"].values
    hist_lows = df_clean["<LOW>"].values
    hist_closes = df_clean["<CLOSE>"].shift(1).fillna(df_clean["<OPEN>"]).values
    tr_all = np.maximum(hist_highs - hist_lows, np.maximum(np.abs(hist_highs - hist_closes), np.abs(hist_lows - hist_closes)))
    tr_series = pd.Series(tr_all, index=df_clean.index)
    hist_atr_ratio = tr_series.rolling(5).mean() / (tr_series.rolling(20).mean() + 1e-9)
    
    current_trend_z = (latest_close - sma_200) / (df_clean["<CLOSE>"].tail(200).std() + 1e-9)
    current_mom_dist = (latest_close - ema_50) / (ema_50 + 1e-9)
    current_vol_spike = (recent_vol - base_vol) / base_vol
    current_atr_ratio = atr_5_now / (np.mean(tr_all[-20:]) / latest_close + 1e-9)
    
    t_trend = hist_trend_z.dropna().median()
    t_mom   = hist_mom_dist.dropna().median()
    t_vol   = hist_vol_spike.dropna().median()
    t_atr   = hist_atr_ratio.dropna().median()
    
    b_trend = current_trend_z > t_trend
    b_mom   = current_mom_dist > t_mom
    b_vol   = current_vol_spike > t_vol
    b_atr   = current_atr_ratio > t_atr
    
    print(f" [METRIC VALUES]  Trend Z: {current_trend_z:+.2f} (Med: {t_trend:+.2f}) | Mom Dist: {current_mom_dist:+.4f} (Med: {t_mom:+.4f})")
    print(f"                  Vol Spike: {current_vol_spike*100:+.1f}% (Med: {t_vol*100:+.1f}%) | ATR Ratio: {current_atr_ratio:.2f} (Med: {t_atr:.2f})")
    print("-" * 130)
    
    if b_trend and b_mom and b_vol and b_atr:
        print(" [STATE: INSTITUTIONAL BREAKOUT EXPANSION]")
        print("   -> Profile : Price is structurally bullish, short-term momentum is accelerating, volume and volatility are both spiking.")
        print("   -> Bias    : High momentum exposure environment. Be careful utilizing rigid mean-reversion rules here.")
    elif b_trend and b_mom and not b_vol and not b_atr:
        print(" [STATE: MATURE INSTITUTIONAL GRIND]")
        print("   -> Profile : Elevated trend and momentum, but trading volume and true range are drying up.")
        print("   -> Bias    : Reduce global position sizes. This is a low-liquidity grinding environment prone to false breakouts.")
    elif not b_trend and not b_mom and b_vol and b_atr:
        print(" [STATE: CAPITULATION / PANIC LIQUIDATION]")
        print("   -> Profile : Price is broken below macro trend and short-term momentum handles, but volume and volatility are structurally maxed.")
        print("   -> Bias    : High tactical edge for Mean Reversion on extreme exhaustion sweeps once lower wick scenarios test out.")
    elif not b_trend and not b_mom and not b_vol and not b_atr:
        print(" [STATE: APATHY SECTOR / EQUILIBRIUM]")
        print("   -> Profile : Below average trend, momentum, volume, and volatility. The asset is entirely dormant.")
        print("   -> Bias    : Maintain minimal standard allocations. Rely on tight-bound parameters until institutional volume returns.")
    elif (b_trend or b_mom) and (b_vol and not b_atr):
        print(" [STATE: LIQUIDITY DISTRIBUTION / INSIDER SELLING]")
        print("   -> Profile : Price looks technically strong or flat, volume is spiking heavily, but actual price volatility is being compressed.")
        print("   -> Bias    : Major players are absorbing buy orders without letting price move up. Exercise extreme caution on long trends.")
    elif (not b_trend and b_mom) and (not b_vol and b_atr):
        print(" [STATE: MEAN REVERSION SQUEEZE]")
        print("   -> Profile : Long-term trend is bearish, but short-term momentum has triggered a sharp volatility squeeze on thin volume.")
        print("   -> Bias    : Prime short-reversion setup. Look for tactical opportunities at the upper limits of tomorrow's simulations.")
    else:
        print(" [STATE: TRANSITIONAL REGIME MIX]")
        print("   -> Profile : Mixed macro signals (conflicting trend/momentum directions or detached volume/volatility parameters).")
        print("   -> Bias    : Standard baseline allocations. Let localized technical indicators sort out micro-edge variations.")
    print("-" * 130 + "\n")
    
    # --- AUTOMATED ENVIRONMENTAL RISK STRESS SUITE ---
    print("Calculating historical macro-durability metrics for engine audit...")
    df_f1 = generate_flavor_1_liquidity_black_hole(df_clean, start_idx=int(len(df_clean)*0.4), window=10, severity=0.15)
    df_f2 = generate_flavor_2_structural_bear_market(df_clean, start_idx=int(len(df_clean)*0.35), daily_drift=0.004)
    df_f3 = generate_flavor_3_high_volatility_grind(df_clean, speed_of_reversion=0.2, volatility=0.025)
    df_f4 = generate_flavor_4_the_gold_rush(df_clean, start_idx=int(len(df_clean)*0.45), peak_idx=int(len(df_clean)*0.7), acceleration=1.6)
    
    _, base_sh, base_so, base_dd, b_sh, _ = run_walk_forward_validation_tuned(df_clean, train_window=train_window)
    _, f1_sh, f1_so, f1_dd, _, _ = run_walk_forward_validation_tuned(df_f1, train_window=train_window)
    _, f2_sh, f2_so, f2_dd, _, _ = run_walk_forward_validation_tuned(df_f2, train_window=train_window)
    _, f3_sh, f3_so, f3_dd, _, _ = run_walk_forward_validation_tuned(df_f3, train_window=train_window)
    _, f4_sh, f4_so, f4_dd, _, _ = run_walk_forward_validation_tuned(df_f4, train_window=train_window)

    print("\n=====================================================================")
    print("             DAILY MODEL REGIME RISK & DURABILITY MATRIX             ")
    print("=====================================================================")
    print(f" REGIME ENVIRONMENT               │ STRAT SHARPE │ SORTINO │ MAX DD  │ BENCH SHARPE")
    print("──────────────────────────────────┼──────────────┼─────────┼─────────┼─────────────")
    print(f" Raw Historical Baseline Data     │    {base_sh:5.2f}     │  {base_so:5.2f}  │ {base_dd:6.1%} │    {b_sh:5.2f}")
    print(f" Flavor 1: Liquidity Black Hole   │    {f1_sh:5.2f}     │  {f1_so:5.2f}  │ {f1_dd:6.1%} │     --")
    print(f" Flavor 2: Structural Bear Market │    {f2_sh:5.2f}     │  {f2_so:5.2f}  │ {f2_dd:6.1%} │     --")
    print(f" Flavor 3: High-Volatility Grind  │    {f3_sh:5.2f}     │  {f3_so:5.2f}  │ {f3_dd:6.1%} │     --")
    print(f" Flavor 4: The Gold Rush (Bubble) │    {f4_sh:5.2f}     │  {f4_so:5.2f}  │ {f4_dd:6.1%} │     --")
    print("=====================================================================\n")

# =====================================================================
# 6. RUNTIME SIMULATION
# =====================================================================

if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=1000, freq="D")
    
    returns_vector = np.random.normal(0.0005, 0.012, size=1000)
    close = 100 * np.exp(np.cumsum(returns_vector))
    open_p = close * (1.0 + np.random.normal(0, 0.003, size=1000))
    high = np.maximum(open_p, close) * (1.0 + np.random.exponential(0.004, size=1000))
    low = np.minimum(open_p, close) * (1.0 - np.random.exponential(0.004, size=1000))
    tickvol = np.random.randint(1000, 5000, size=1000)

    mock_df = pd.DataFrame({
        "<DATE>": dates, "<OPEN>": open_p, "<HIGH>": high, "<LOW>": low, "<CLOSE>": close, "<TICKVOL>": tickvol
    })

    predict_next_day_scenarios(mock_df, train_window=300)
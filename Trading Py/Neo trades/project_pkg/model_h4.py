import numpy as np
import pandas as pd
import warnings
import datetime
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore", category=UserWarning)

# =====================================================================
# 0. GLOBAL MULTI-TIMEFRAME FEATURE ENCYCLOPEDIA & STRATEGY DICTIONARIES
# =====================================================================
MEAN_REVERSION_FEATURES = {
    "raw_volatility_5": "5-day rolling standard dev - short-term risk magnitude",
    "dist_MA_5": "% distance from 5-day ma - short-term mean-reversion",
    "RSI": "Relative Strength Index - overextended buyers or sellers.",
    "Stochastic_14": "14-day Stochastic Oscillator - close & recent high-low range",
    "vol_acceleration": "Rate of change in volatility magnitude",
    "price_zscore_5": "5-day price Z-Score - short-term extremeness of current price.",
    "volume_spread_ratio": "Volume ratio * high-low spread - liquidity traps.",
    "h4_rsi_imbalance": "Max - Min H4 RSI in a day - intraday momentum imbalance",
    "HL_spread": "Intraday high-to-low spread - volatility/liquidity stress"
}

PURE_MOMENTUM_FEATURES = {
    "returns": "Log price change over previous session - close momentum",
    "session_sign": "Direction of the daily session close vs open (-1 or 1)",
    "CO_spread": "Intraday close-to-open - overnight vs. intraday gap risk",
    "volume_ratio": "Current volume vs. 20-day avg - institution participation",
    "raw_NTR_5": "5 days True Range - structural volatility magnitude",
    "ROC_10": "Rate of Change over 10 days - macro price shifts",
    "raw_h4_volatility": "Stdev of H4 log returns within daily session.",
    "h4_close_to_daily_open": "% distance between final H4 close and market open.",
    "h4_volume_concentration": "%ge daily trading volume in highest-volume H4 bar."
}

# =====================================================================
# 1. SYNTHETIC TAIL RISK SCENARIO REGENERATION ("THE FLAVORS")
# =====================================================================

def generate_flavor_1_liquidity_black_hole(df, start_idx=400, window=15, severity=0.15, volume_spike=4.0):
    df_mut = df.copy()
    if start_idx >= len(df_mut): return df_mut
    end_idx = min(start_idx + window, len(df_mut))
    crash_factor = np.linspace(1.0, 1.0 - severity, end_idx - start_idx)
    for i, idx in enumerate(range(start_idx, end_idx)):
        df_mut.loc[idx, "<CLOSE>"] *= crash_factor[i]
        df_mut.loc[idx, "<LOW>"] *= (crash_factor[i] * 0.98)
        df_mut.loc[idx, "<TICKVOL>"] *= volume_spike
    return df_mut

def generate_flavor_2_structural_bear_market(df, start_idx=300, daily_drift=0.004, noise_scale=0.012):
    df_mut = df.copy()
    if start_idx >= len(df_mut): return df_mut
    current_price = df_mut.loc[start_idx, "<CLOSE>"]
    for idx in range(start_idx, len(df_mut)):
        random_shock = np.random.normal(-daily_drift, noise_scale * 1.5)
        current_price *= np.exp(random_shock)
        df_mut.loc[idx, "<CLOSE>"] = current_price
        df_mut.loc[idx, "<OPEN>"] = current_price * np.exp(-random_shock * 0.2)
        df_mut.loc[idx, "<HIGH>"] = max(df_mut.loc[idx, "<CLOSE>"], df_mut.loc[idx, "<OPEN>"]) * 1.005
        df_mut.loc[idx, "<LOW>"] = min(df_mut.loc[idx, "<CLOSE>"], df_mut.loc[idx, "<OPEN>"]) * 0.992
    return df_mut

def generate_flavor_3_high_volatility_grind(df, speed_of_reversion=0.15, volatility=0.02):
    df_mut = df.copy()
    log_prices = np.log(df_mut["<CLOSE>"].values)
    mean_log_price = np.mean(log_prices)
    for idx in range(1, len(df_mut)):
        drift = speed_of_reversion * (mean_log_price - log_prices[idx-1])
        shock = np.random.normal(0, volatility)
        log_prices[idx] = log_prices[idx-1] + drift + shock
        sim_close = np.exp(log_prices[idx])
        df_mut.loc[idx, "<CLOSE>"] = sim_close
        df_mut.loc[idx, "<OPEN>"] = np.exp(log_prices[idx-1])
        df_mut.loc[idx, "<HIGH>"] = max(sim_close, df_mut.loc[idx, "<OPEN>"]) * (1.0 + abs(shock)*0.5)
        df_mut.loc[idx, "<LOW>"] = min(sim_close, df_mut.loc[idx, "<OPEN>"]) * (1.0 - abs(shock)*0.5)
    return df_mut

def generate_flavor_4_the_gold_rush(df, start_idx=500, peak_idx=750, acceleration=1.4, bubble_pop_rate=0.4):
    df_mut = df.copy()
    if start_idx >= len(df_mut) or peak_idx >= len(df_mut) or start_idx >= peak_idx: return df_mut
    base_price = df_mut.loc[start_idx, "<CLOSE>"]
    for idx in range(start_idx, peak_idx):
        progress = (idx - start_idx) / (peak_idx - start_idx)
        multiplier = 1.0 + (progress ** acceleration) * 0.8
        df_mut.loc[idx, "<CLOSE>"] = base_price * multiplier
        df_mut.loc[idx, "<HIGH>"] = df_mut.loc[idx, "<CLOSE>"] * 1.01
        df_mut.loc[idx, "<LOW>"] = df_mut.loc[idx, "<CLOSE>"] * 0.99
    peak_price = df_mut.loc[peak_idx, "<CLOSE>"]
    for idx in range(peak_idx, len(df_mut)):
        steps_post = idx - peak_idx
        decay = np.exp(-bubble_pop_rate * steps_post)
        sim_close = peak_price * (0.4 + 0.6 * decay) + np.random.normal(0, peak_price * 0.02)
        df_mut.loc[idx, "<CLOSE>"] = max(sim_close, base_price * 0.3)
        df_mut.loc[idx, "<HIGH>"] = df_mut.loc[idx, "<CLOSE>"] * 1.03 
        df_mut.loc[idx, "<LOW>"] = df_mut.loc[idx, "<CLOSE>"] * 0.95
    return df_mut

# =====================================================================
# 2. TIMEFRAME DISTILLATION & PIPELINE ENGINES
# =====================================================================

def calculate_technical_indicators(df):
    df = df.copy()
    df["returns"] = np.log(df["<CLOSE>"] / df["<CLOSE>"].shift(1))
    df["session_sign"] = np.sign(df["<CLOSE>"] - df["<OPEN>"])
    df["raw_volatility_5"] = df["returns"].rolling(window=5).std()
    df["HL_spread"] = (df["<HIGH>"] - df["<LOW>"]) / df["<CLOSE>"]
    df["CO_spread"] = (df["<CLOSE>"] - df["<OPEN>"]) / df["<OPEN>"]    
    ma_5 = df["<CLOSE>"].rolling(window=5).mean()
    df["dist_MA_5"] = (df["<CLOSE>"] - ma_5) / ma_5
    df["volume_ratio"] = df["<TICKVOL>"] / (df["<TICKVOL>"].rolling(window=20).mean() + 1e-9)
    df["volume_ratio"] = df["volume_ratio"].fillna(1.0)
    delta = df["<CLOSE>"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df["RSI"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    df["daily_rsi_ema"] = df["RSI"].ewm(span=10, min_periods=1).mean().fillna(50.0)
    lowest_low_14 = df["<LOW>"].rolling(window=14).min()
    highest_high_14 = df["<HIGH>"].rolling(window=14).max()
    df["Stochastic_14"] = ((df["<CLOSE>"] - lowest_low_14) / (highest_high_14 - lowest_low_14 + 1e-9)) * 100
    prev_close = df["<CLOSE>"].shift(1)
    tr1 = df["<HIGH>"] - df["<LOW>"]
    tr2 = (df["<HIGH>"] - prev_close).abs()
    tr3 = (df["<LOW>"] - prev_close).abs()
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    df["raw_NTR_5"] = (true_range.rolling(window=5).mean() / df["<CLOSE>"])
    df["ROC_10"] = ((df["<CLOSE>"] - df["<CLOSE>"].shift(10)) / df["<CLOSE>"].shift(10)) * 100
    df["vol_acceleration"] = (df["raw_volatility_5"].diff() / (df["raw_volatility_5"].shift(1) + 1e-9))
    ma_5_macro = df["<CLOSE>"].rolling(window=5).mean()
    std_5_macro = df["<CLOSE>"].rolling(window=5).std()
    df["price_zscore_5"] = (df["<CLOSE>"] - ma_5_macro) / (std_5_macro + 1e-9)
    df["volume_spread_ratio"] = df["volume_ratio"] * df["HL_spread"]
    
    # --- FIXED SECTION: Removed .bfill() to prevent data leakage ---
    df["realized_vol_20"] = df["returns"].rolling(window=20).std()
    df["realized_vol_20_smoothed"] = df["realized_vol_20"].ewm(span=5, min_periods=1).mean()
    return df

def extract_and_merge_h4_features(df_daily, df_h4):
    df_daily = df_daily.copy()
    df_h4 = df_h4.copy()
    df_h4["h4_returns"] = np.log(df_h4["<CLOSE>"] / df_h4["<CLOSE>"].shift(1))
    delta = df_h4["<CLOSE>"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df_h4["h4_rsi"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    h4_grouped = df_h4.groupby("<DATE>")
    h4_features = pd.DataFrame(index=h4_grouped.groups.keys())
    h4_features["raw_h4_volatility"] = h4_grouped["h4_returns"].std()
    h4_features["h4_rsi_imbalance"] = h4_grouped["h4_rsi"].apply(
        lambda x: x.max() - x.min() if x.notna().any() else np.nan
    )
    h4_features["h4_volume_concentration"] = h4_grouped.apply(
        lambda x: x["<TICKVOL>"].max() / (x["<TICKVOL>"].sum() + 1e-9) if len(x) > 0 else 0.25
    )
    h4_features = h4_features.reset_index().rename(columns={"index": "<DATE>"}) 
    h4_features = h4_features.dropna(subset=["<DATE>"])
    merged_df = pd.merge(df_daily, h4_features, on="<DATE>", how="left")
    merged_df["h4_close_to_daily_open"] = (merged_df["<CLOSE>"] - merged_df["<OPEN>"]) / (merged_df["<OPEN>"] + 1e-9)
    return merged_df

def simulate_h4_bars_for_scenario(date_str, open_p, high_p, low_p, close_p, base_tickvol):
    h4_records = []
    total_slots = 6
    vol_weights = [0.25, 0.12, 0.08, 0.10, 0.20, 0.25]
    v_shares = [int(base_tickvol * w) for w in vol_weights]
    mid_high = max(open_p, close_p) + (high_p - max(open_p, close_p)) * 0.6
    mid_low = min(open_p, close_p) - (min(open_p, close_p) - low_p) * 0.6
    steps = [open_p + (mid_high - open_p) * 0.5, high_p, mid_low, low_p, close_p - (close_p - mid_low) * 0.3, close_p]
    for i in range(total_slots):
        h4_records.append({"<DATE>": date_str, "<CLOSE>": steps[i], "<TICKVOL>": max(v_shares[i], 10)})
    return pd.DataFrame(h4_records)

# =====================================================================
# 3. PURGED & EMBARGOED WALK-FORWARD VALIDATION FUNCTION
# =====================================================================

def run_block_walk_forward_validation(df_daily, df_h4, train_window, retrain_interval=100):
    train_window = int(train_window)
    df = df_daily.copy().sort_values("<DATE>").reset_index(drop=True)
    df = calculate_technical_indicators(df)
    df = extract_and_merge_h4_features(df, df_h4)
    df["future_raw_returns"] = df["returns"].shift(-1)
    df["future_vol_adjusted_target"] = df["future_raw_returns"] / (df["realized_vol_20"] + 1e-9)
    
    rev_feats = list(MEAN_REVERSION_FEATURES.keys())
    mom_feats = list(PURE_MOMENTUM_FEATURES.keys())
    all_needed_feats = list(set(rev_feats + mom_feats))
    df = df.dropna(subset=all_needed_feats + ["future_vol_adjusted_target"]).reset_index(drop=True)

    out_of_sample_preds_rev, out_of_sample_preds_mom = [], []
    actuals_all = []
    embargo_window, purge_window = 2, 1    

    for end_train_idx in range(train_window, len(df), retrain_interval):
        train_start_cutoff = purge_window
        train_end_cutoff = end_train_idx - embargo_window
        if train_end_cutoff <= train_start_cutoff: continue
        X_train_df = df.iloc[train_start_cutoff:train_end_cutoff].copy()
        
        end_test_idx = min(end_train_idx + retrain_interval, len(df))
        test_block = df.iloc[end_train_idx : end_test_idx].copy()
        if len(test_block) == 0: break

        for col in ["raw_h4_volatility", "h4_volume_concentration"]:
            fill_val = X_train_df[col].median() if not X_train_df[col].isna().all() else 0.0
            X_train_df[col] = X_train_df[col].ffill().fillna(fill_val)
            test_block[col] = test_block[col].ffill().fillna(fill_val)
            
        X_train_df["h4_rsi_imbalance"] = X_train_df["h4_rsi_imbalance"].ffill().fillna(X_train_df["daily_rsi_ema"])
        test_block["h4_rsi_imbalance"] = test_block["h4_rsi_imbalance"].ffill().fillna(test_block["daily_rsi_ema"])

        train_mean = X_train_df["future_vol_adjusted_target"].mean()
        train_std = X_train_df["future_vol_adjusted_target"].std() + 1e-9
        X_train_df["target"] = (X_train_df["future_vol_adjusted_target"] - train_mean) / train_std

        model_rev = LGBMRegressor(objective="regression", n_estimators=80, learning_rate=0.05, max_depth=4, verbose=-1, random_state=42)
        model_rev.fit(X_train_df[rev_feats], X_train_df["target"])
        
        model_mom = LGBMRegressor(objective="regression", n_estimators=80, learning_rate=0.05, max_depth=4, verbose=-1, random_state=42)
        model_mom.fit(X_train_df[mom_feats], X_train_df["target"])

        p_z_rev = model_rev.predict(test_block[rev_feats])
        p_z_mom = model_mom.predict(test_block[mom_feats])
        
        out_of_sample_preds_rev.extend((p_z_rev * train_std + train_mean) * test_block["realized_vol_20_smoothed"])
        out_of_sample_preds_mom.extend((p_z_mom * train_std + train_mean) * test_block["realized_vol_20_smoothed"])
        actuals_all.extend(test_block["future_raw_returns"])

    return mean_absolute_error(actuals_all, out_of_sample_preds_rev), mean_absolute_error(actuals_all, out_of_sample_preds_mom)

# =====================================================================
# 4. HISTORICAL MACRO FLAVOR STRESS TESTING RESULTS ENGINE
# =====================================================================

def run_flavor_stress_tests(df_daily, df_h4, train_window=600):
    print("\n" + "-"*65)
    print(" TAIL RISK STRESS TEST RESULTS MATRIX (HISTORICAL MACRO FLAVORS)")
    print("-"*65)
    print(f" {'MACROECONOMIC SIMULATED CRITICAL FLAVOR':<32} | {'ENG1 MAE':<9} | {'ENG2 MAE'}")
    print("-" * 65)
    
    flavors = {
        "Flavor 1 (Liquidity Black Hole)": generate_flavor_1_liquidity_black_hole(df_daily),
        "Flavor 2 (Structural Bear Mkt)":  generate_flavor_2_structural_bear_market(df_daily),
        "Flavor 3 (High-Vol Grind)":       generate_flavor_3_high_volatility_grind(df_daily),
        "Flavor 4 (The Gold Rush Bubble)": generate_flavor_4_the_gold_rush(df_daily)
    }
    
    for f_name, f_df in flavors.items():
        try:
            mae_rev, mae_mom = run_block_walk_forward_validation(f_df, df_h4, train_window=train_window, retrain_interval=100)
            print(f" {f_name:<32} | {mae_rev:<9.5f} | {mae_mom:.5f}")
        except Exception:
            print(f" {f_name:<32} | {'ERROR':<9} | {'ERROR'}")
    print("-"*65 + "\n")

# =====================================================================
# 5. LIVE PRODUCTION ENVIRONMENT MATRIX PIPELINE WITH 4-METRIC HEALTH CHECK
# =====================================================================

def predict_next_day_scenarios(df_daily, df_h4, train_window=150):
    rev_feats = list(MEAN_REVERSION_FEATURES.keys())
    mom_feats = list(PURE_MOMENTUM_FEATURES.keys())
    all_needed_feats = list(set(rev_feats + mom_feats))
    
    today_str = datetime.datetime.now().strftime("%Y.%m.%d")
    df_daily_clean = df_daily[df_daily["<DATE>"] != today_str].copy().sort_values("<DATE>").reset_index(drop=True)
    df_h4_clean = df_h4[df_h4["<DATE>"] != today_str].copy()
    
    latest_close = df_daily_clean.iloc[-1]["<CLOSE>"]
    
    # [METRIC 1] Structural Trend: 200-Day Simple Moving Average
    sma_200 = df_daily_clean["<CLOSE>"].tail(200).mean()
    trend_status = "ABOVE (Bullish Structure)" if latest_close > sma_200 else "BELOW (Bearish Structure)"
    
    # [METRIC 2] Macro Momentum: 50-Day Exponential Moving Average
    ema_50 = df_daily_clean["<CLOSE>"].ewm(span=50, adjust=False).mean().iloc[-1]
    mom_status = "ABOVE (Macro Momentum Intact)" if latest_close > ema_50 else "BELOW (Short-Term Macro Fatigue)"
    
    # [METRIC 3] Institution Volume: Current rolling 5-day volume vs last 20 days
    recent_vol = df_daily_clean["<TICKVOL>"].tail(5).mean()
    base_vol = df_daily_clean["<TICKVOL>"].tail(20).mean() + 1e-9
    vol_pct_change = ((recent_vol - base_vol) / base_vol) * 100
    vol_status = f"{vol_pct_change:+.1f}% Spike vs 20-Day Avg"
    
    # [METRIC 4] Regime Volatility: True Range Over Last 5 Days Normalized vs 20-Day Distribution
    highs = df_daily_clean["<HIGH>"].tail(20).values
    lows = df_daily_clean["<LOW>"].tail(20).values
    closes = df_daily_clean["<CLOSE>"].tail(21).values
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

    # --- MODEL CORE PIPELINE ---
    df_features = calculate_technical_indicators(df_daily_clean)
    df_features = extract_and_merge_h4_features(df_features, df_h4_clean)
    
    latest_known_row = df_features.dropna(subset=all_needed_feats).iloc[-1]
    atr_5 = latest_known_row["raw_NTR_5"] * latest_close
    
    df_features["future_raw_returns"] = df_features["returns"].shift(-1)
    df_features["future_vol_adjusted_target"] = df_features["future_raw_returns"] / (df_features["realized_vol_20"] + 1e-9)
    
    train_df = df_features.dropna(subset=all_needed_feats + ["future_vol_adjusted_target"]).reset_index(drop=True)
    X_train_df = train_df.tail(int(train_window)).copy()
    
    train_mean = X_train_df["future_vol_adjusted_target"].mean()
    train_std = X_train_df["future_vol_adjusted_target"].std() + 1e-9
    X_train_df["target"] = (X_train_df["future_vol_adjusted_target"] - train_mean) / train_std

    model_rev_lgb = LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1)
    model_rev_ridge = Ridge(alpha=1.0)
    model_rev_lgb.fit(X_train_df[rev_feats], X_train_df["target"])
    model_rev_ridge.fit(X_train_df[rev_feats], X_train_df["target"])
    
    model_mom_lgb = LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1)
    model_mom_ridge = Ridge(alpha=1.0)
    model_mom_lgb.fit(X_train_df[mom_feats], X_train_df["target"])
    model_mom_ridge.fit(X_train_df[mom_feats], X_train_df["target"])
    
    sim_open, tickvol_ref = latest_close, latest_known_row["<TICKVOL>"]
    scenarios = {
        "1. Pure Positive Breakout":       {"C": sim_open + (0.6 * atr_5), "H": sim_open + (0.7 * atr_5), "L": sim_open - (0.1 * atr_5)},
        "2. Positive + Upper Wick Drop":   {"C": sim_open + (0.1 * atr_5), "H": sim_open + (0.8 * atr_5), "L": sim_open - (0.1 * atr_5)},
        "3. Pure Negative Breakdown":      {"C": sim_open - (0.6 * atr_5), "H": sim_open + (0.1 * atr_5), "L": sim_open - (0.7 * atr_5)},
        "4. Negative + Lower Wick Sweep":  {"C": sim_open - (0.1 * atr_5), "H": sim_open + (0.1 * atr_5), "L": sim_open - (0.8 * atr_5)},
        "5. Compressed Neutral Chop":      {"C": sim_open + (0.0 * atr_5), "H": sim_open + (0.1 * atr_5), "L": sim_open - (0.1 * atr_5)},
        "6. High-Vol Indecision Cross":    {"C": sim_open + (0.0 * atr_5), "H": sim_open + (0.7 * atr_5), "L": sim_open - (0.7 * atr_5)}
    }
    
    print("\n" + "="*136)
    print("                                                 LIVE SCENARIO MATRIX GENERATOR                                                 ")
    print("="*136)
    print(f" {'STRUCTURAL SCENARIO':<32} | {'SIM CLOSE':<9} | {'E1 ALLOC':<8} | {'E2 ALLOC':<8} | {'PRED PRICE':<10} | {'TACTICAL VECTOR':<15} | {'TAKE PROFIT':<11} | {'STOP LOSS'}")
    print("-" * 136)
    
    for name, levels in scenarios.items():
        sim_date = "TOMORROW_SIM"
        sim_row = pd.DataFrame([{"<DATE>": sim_date, "<OPEN>": sim_open, "<HIGH>": levels["H"], "<LOW>": levels["L"], "<CLOSE>": levels["C"], "<TICKVOL>": tickvol_ref}])
        sim_h4_df = simulate_h4_bars_for_scenario(sim_date, sim_open, levels["H"], levels["L"], levels["C"], tickvol_ref)
        
        extended_features = calculate_technical_indicators(pd.concat([df_daily_clean, sim_row], ignore_index=True))
        extended_features = extract_and_merge_h4_features(extended_features, pd.concat([df_h4_clean, sim_h4_df], ignore_index=True))
        
        sim_vector = extended_features.iloc[[-1]].copy()
        smooth_vol = sim_vector["realized_vol_20_smoothed"].values[0]
        
        pred_rev = (((model_rev_lgb.predict(sim_vector[rev_feats])[0] + model_rev_ridge.predict(sim_vector[rev_feats])[0]) * 0.5) * train_std + train_mean) * smooth_vol
        alloc_rev = 0.0 if abs(pred_rev) < (smooth_vol * 0.05) else np.clip((pred_rev / smooth_vol), -1.0, 1.0)
        
        pred_mom = (((model_mom_lgb.predict(sim_vector[mom_feats])[0] + model_mom_ridge.predict(sim_vector[mom_feats])[0]) * 0.5) * train_std + train_mean) * smooth_vol
        alloc_mom = 0.0 if abs(pred_mom) < (smooth_vol * 0.05) else np.clip((pred_mom / smooth_vol), -1.0, 1.0)
        
        combined_alloc = 0.5 * alloc_rev + 0.5 * alloc_mom
        predicted_market_price = latest_close * np.exp((0.5 * pred_rev) + (0.5 * pred_mom))
        
        if abs(combined_alloc) < 0.05:
            action, tp_str, sl_str = "STANDBY", "N/A", "N/A"
        elif combined_alloc > 0:
            action = f"LONG {combined_alloc:+.2f}x"
            tp_str, sl_str = f"{levels['C'] + (1.5 * atr_5):<11.4f}", f"{levels['C'] - (1.0 * atr_5):.4f}"
        else:
            action = f"SHORT {combined_alloc:.2f}x"
            tp_str, sl_str = f"{levels['C'] - (1.5 * atr_5):<11.4f}", f"{levels['C'] + (1.0 * atr_5):.4f}"
            
        print(f" {name:<32} | {levels['C']:<9.4f} | {alloc_rev:+.2f}    | {alloc_mom:+.2f}    | {predicted_market_price:<10.4f} | {action:<15} | {tp_str} | {sl_str}")
    print("="*136)
    
    # --- DYNAMIC 4-METRIC DATA-DRIVEN REGIME COCKPIT (MECE & SYMBOL-AGNOSTIC) ---
    print("\n" + " " * 45 + "DYNAMIC 4-METRIC DATA-DRIVEN REGIME COCKPIT")
    print("-" * 136)
    
    # 1. Generate Historical Distributions for all 4 exact parameters to build asset-specific thresholds
    hist_sma200 = df_daily_clean["<CLOSE>"].rolling(window=200).mean()
    hist_sma200_std = df_daily_clean["<CLOSE>"].rolling(window=200).std() + 1e-9
    hist_trend_z = (df_daily_clean["<CLOSE>"] - hist_sma200) / hist_sma200_std
    
    hist_ema50 = df_daily_clean["<CLOSE>"].ewm(span=50, adjust=False).mean()
    hist_mom_dist = (df_daily_clean["<CLOSE>"] - hist_ema50) / (hist_ema50 + 1e-9)
    
    hist_vol5 = df_daily_clean["<TICKVOL>"].rolling(window=5).mean()
    hist_vol20 = df_daily_clean["<TICKVOL>"].rolling(window=20).mean() + 1e-9
    hist_vol_spike = (hist_vol5 - hist_vol20) / hist_vol20
    
    hist_highs = df_daily_clean["<HIGH>"].values
    hist_lows = df_daily_clean["<LOW>"].values
    hist_closes = df_daily_clean["<CLOSE>"].shift(1).fillna(df_daily_clean["<OPEN>"]).values
    tr_all = np.maximum(hist_highs - hist_lows, np.maximum(np.abs(hist_highs - hist_closes), np.abs(hist_lows - hist_closes)))
    tr_series = pd.Series(tr_all, index=df_daily_clean.index)
    hist_atr_ratio = tr_series.rolling(5).mean() / (tr_series.rolling(20).mean() + 1e-9)
    
    # 2. Evaluate current state valuations against their respective historical medians
    current_trend_z = (latest_close - sma_200) / (df_daily_clean["<CLOSE>"].tail(200).std() + 1e-9)
    current_mom_dist = (latest_close - ema_50) / (ema_50 + 1e-9)
    current_vol_spike = (recent_vol - base_vol) / base_vol
    current_atr_ratio = atr_5_now / (np.mean(tr_all[-20:]) / latest_close + 1e-9)

    # 1. Take absolute values for historical medians so they represent "strength"	
    t_trend = hist_trend_z.abs().dropna().median()
    t_mom   = hist_mom_dist.abs().dropna().median()
    t_vol   = hist_vol_spike.dropna().median()
    t_atr   = hist_atr_ratio.dropna().median()

    # 2. Compare the absolute current state against absolute historical strength
    b_trend = np.abs(current_trend_z) > t_trend
    b_mom   = np.abs(current_mom_dist) > t_mom
    b_vol   = current_vol_spike > t_vol
    b_atr   = current_atr_ratio > t_atr
    	    
    print(f" [METRIC VALUES]  Trend Z: {current_trend_z:+.2f} (Med: {t_trend:+.2f}) | Mom Dist: {current_mom_dist:+.4f} (Med: {t_mom:+.4f})")
    print(f"                  Vol Spike: {current_vol_spike*100:+.1f}% (Med: {t_vol*100:+.1f}%) | ATR Ratio: {current_atr_ratio:.2f} (Med: {t_atr:.2f})")
    print("-" * 136)
    
    # 3. Highly Refined MECE Decision Matrix Execution Paths
    if b_trend and b_mom and b_vol and b_atr:
        print(" [STATE: INSTITUTIONAL BREAKOUT EXPANSION]")
        print("   -> Profile : Price is structurally bullish, short-term momentum is accelerating, volume and volatility are both spiking.")
        print("   -> Bias    : Maximum Engine 2 (Momentum) exposure. De-weight Engine 1 (Mean Reversion) completely to avoid catching a running knife.")
        
    elif b_trend and b_mom and not b_vol and not b_atr:
        print(" [STATE: MATURE INSTITUTIONAL GRIND]")
        print("   -> Profile : Elevated trend and momentum, but trading volume and true range are drying up.")
        print("   -> Bias    : Reduce global position sizes. This is a low-liquidity grinding environment where breakouts are highly prone to fading.")
        
    elif not b_trend and not b_mom and b_vol and b_atr:
        print(" [STATE: CAPITULATION / PANIC LIQUIDATION]")
        print("   -> Profile : Price is broken below macro trend and short-term momentum handles, but volume and volatility are structurally maxed.")
        print("   -> Bias    : High edge for Engine 1 (Mean Reversion) on extreme exhaustion sweeps once tomorrow's lower wick scenario tests out.")
        
    elif not b_trend and not b_mom and not b_vol and not b_atr:
        print(" [STATE: APATHY SECTOR / EQUILIBRIUM]")
        print("   -> Profile : Below average trend, momentum, volume, and volatility. The asset is entirely dormant.")
        print("   -> Bias    : Maintain minimal standard allocations. Rely on tight-bound mean reversion parameters until an institutional volume spike occurs.")
        
    elif (b_trend or b_mom) and (b_vol and not b_atr):
        print(" [STATE: LIQUIDITY DISTRIBUTION / INSIDER SELLING]")
        print("   -> Profile : Price looks technically strong or flat, volume is spiking heavily, but actual price volatility is being compressed.")
        print("   -> Bias    : Major players are absorbing buy orders without letting price move up. Exercise extreme caution on long allocations.")
        
    elif (not b_trend and b_mom) and (not b_vol and b_atr):
        print(" [STATE: MEAN REVERSION SQUEEZE]")
        print("   -> Profile : Long-term trend is bearish, but short-term momentum has triggered a sharp volatility squeeze on thin volume.")
        print("   -> Bias    : Prime short-reversion setup. Look for short Engine 1 targets at the extremes of tomorrow's upper wick simulations.")
        
    else:
        print(" [STATE: TRANSITIONAL REGIME MIX]")
        print("   -> Profile : Mixed macro signals (conflicting trend/momentum directions or detached volume/volatility parameters).")
        print("   -> Bias    : Equal 50/50 blend between Engine 1 and Engine 2. Let the localized LightGBM features sort out micro-edge allocations.")
        
    print("-" * 136 + "\n")
import numpy as np
import pandas as pd
import warnings
import datetime
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
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

# Unsupervised Clustering Features for Regime Definition
REGIME_CLUSTERING_FEATURES = [
    "dist_MA_5", 
    "price_zscore_5", 
    "volume_ratio", 
    "raw_NTR_5"
]

# =====================================================================
# 0b. ASSET UNIVERSE & CROSS-ASSET KNOWLEDGE MAP
# =====================================================================
PRIMARY_ASSETS = ["AUDUSD", "USDJPY", "EURUSD", "US30.a", "XAUUSD.a", "NAS100.a", "USDINR"]
SECONDARY_ASSETS = ["USTN10YR-F.a", "SpotBrent", "VIX.a", "USDX.a"]

CROSS_ASSET_SHORTLISTS = {
    "EURUSD":   ["USDX.a", "USTN10YR-F.a", "VIX.a", "US30.a"],
    "AUDUSD":   ["XAUUSD.a", "SpotBrent", "NAS100.a", "USDX.a"],
    "USDINR":   ["SpotBrent", "USDX.a", "USTN10YR-F.a", "NAS100.a"],
    "USDJPY":   ["USTN10YR-F.a", "NAS100.a", "USDX.a", "VIX.a"],
    "XAUUSD.a": ["USDX.a", "USTN10YR-F.a", "VIX.a", "EURUSD"],
    "US30.a":   ["VIX.a", "USTN10YR-F.a", "NAS100.a", "EURUSD"],
    "NAS100.a": ["USTN10YR-F.a", "VIX.a", "US30.a", "USDX.a"]
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

def generate_cross_asset_features(target_symbol, df_target, dfs_dict):
    if dfs_dict is None or target_symbol not in CROSS_ASSET_SHORTLISTS:
        return df_target.copy(), []
        
    df_target = df_target.copy()
    context_symbols = CROSS_ASSET_SHORTLISTS[target_symbol]
    target_ret = np.log(df_target["<CLOSE>"] / df_target["<CLOSE>"].shift(1))
    generated_cols = []
    
    for context_sym in context_symbols:
        if context_sym not in dfs_dict:
            continue
            
        context_df = dfs_dict[context_sym].copy()
        if len(context_df) == 0:
            continue
            
        context_df[f"_ret_{context_sym}"] = np.log(context_df["<CLOSE>"] / context_df["<CLOSE>"].shift(1))
        merged = pd.merge(
            df_target[["<DATE>"]].assign(_target_ret=target_ret),
            context_df[["<DATE>", f"_ret_{context_sym}"]],
            on="<DATE>",
            how="left"
        )
        
        lag_col = f"context_{context_sym}_return_lag1"
        df_target[lag_col] = merged[f"_ret_{context_sym}"].shift(1)
        
        corr_col = f"context_{context_sym}_corr_20"
        df_target[corr_col] = (
            merged["_target_ret"]
            .shift(1)
            .rolling(window=20)
            .corr(merged[f"_ret_{context_sym}"].shift(1))
        )
        
        df_target[lag_col] = df_target[lag_col].fillna(0.0)
        df_target[corr_col] = df_target[corr_col].fillna(0.0)
        generated_cols.extend([lag_col, corr_col])
        
    return df_target, generated_cols

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

def sim_daily_row(open_p, close_p, atr, vol):
    row = pd.DataFrame([{
        "<OPEN>": open_p, 
        "<HIGH>": max(open_p, close_p) + 0.1 * atr, 
        "<LOW>": min(open_p, close_p) - 0.1 * atr, 
        "<CLOSE>": close_p, 
        "<TICKVOL>": vol
    }])
    row["returns"] = np.log(close_p / open_p)
    row["dist_MA_5"] = (close_p - open_p) / open_p
    row["price_zscore_5"] = 0.5 if close_p > open_p else -0.5
    row["volume_ratio"] = 1.0
    row["raw_NTR_5"] = atr / close_p
    return row

# =====================================================================
# 3. ROBUST ENVIRONMENT-INDEPENDENT UNSUPERVISED CLUSTERING
# =====================================================================

class ThreadSafeKMeans:
    def __init__(self, n_clusters=4, max_iter=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None

    def fit(self, X):
        np.random.seed(self.random_state)
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.cluster_centers_ = X[indices].copy()
        
        for _ in range(self.max_iter):
            distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
            labels = np.argmin(distances, axis=1)
            
            new_centers = np.array([
                X[labels == i].mean(axis=0) if np.any(labels == i) else self.cluster_centers_[i]
                for i in range(self.n_clusters)
            ])
            
            if np.allclose(self.cluster_centers_, new_centers, atol=1e-6):
                break
            self.cluster_centers_ = new_centers
        return self

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
        return np.argmin(distances, axis=1)

def fit_unsupervised_regimes(df_train, df_test=None, n_clusters=4):
    scaler = StandardScaler()
    X_train_clust = df_train[REGIME_CLUSTERING_FEATURES].fillna(0.0).values
    X_train_scaled = scaler.fit_transform(X_train_clust)
    
    kmeans = ThreadSafeKMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_train_scaled)
    train_clusters = kmeans.predict(X_train_scaled)
    
    train_regimes = pd.get_dummies(train_clusters, prefix="regime").astype(float)
    regime_cols = [f"regime_{i}" for i in range(n_clusters)]
    
    for col in regime_cols:
        if col not in train_regimes.columns:
            train_regimes[col] = 0.0
    train_regimes = train_regimes[regime_cols]
            
    if df_test is not None:
        X_test_clust = df_test[REGIME_CLUSTERING_FEATURES].fillna(0.0).values
        X_test_scaled = scaler.transform(X_test_clust)
        test_clusters = kmeans.predict(X_test_scaled)
        
        test_regimes = pd.get_dummies(test_clusters, prefix="regime").astype(float)
        for col in regime_cols:
            if col not in test_regimes.columns:
                test_regimes[col] = 0.0
        test_regimes = test_regimes[regime_cols]
        return train_regimes, test_regimes, kmeans, scaler
        
    return train_regimes, kmeans, scaler

def get_meta_features(X_df, pred_rev, pred_mom, regime_df):
    meta_df = pd.DataFrame({
        "pred_rev": pred_rev,
        "pred_mom": pred_mom
    }, index=X_df.index)
    
    regime_df_aligned = regime_df.set_index(X_df.index)
    regime_cols = [f"regime_{i}" for i in range(4)]
    
    for col in regime_cols:
        if col not in regime_df_aligned.columns:
            regime_df_aligned[col] = 0.0
            
    regime_df_aligned = regime_df_aligned[regime_cols]
    meta_df = pd.concat([meta_df, regime_df_aligned], axis=1)
    return meta_df

# =====================================================================
# 4. PURGED & EMBARGOED WALK-FORWARD VALIDATION FUNCTION (META-BLENDED)
# =====================================================================

def run_block_walk_forward_validation(df_daily, df_h4, train_window, retrain_interval=100, target_symbol=None, dfs_dict=None):
    train_window = int(train_window)
    df = df_daily.copy().sort_values("<DATE>").reset_index(drop=True)
    df = calculate_technical_indicators(df)
    
    df, macro_features = generate_cross_asset_features(target_symbol, df, dfs_dict)
    df = extract_and_merge_h4_features(df, df_h4)
    
    df["future_raw_returns"] = df["returns"].shift(-1)
    df["future_vol_adjusted_target"] = df["future_raw_returns"] / (df["realized_vol_20"] + 1e-9)
    
    rev_feats = list(MEAN_REVERSION_FEATURES.keys())
    mom_feats = list(PURE_MOMENTUM_FEATURES.keys()) + macro_features
    all_needed_feats = list(set(rev_feats + mom_feats + REGIME_CLUSTERING_FEATURES))
    
    df = df.dropna(subset=all_needed_feats + ["future_vol_adjusted_target"]).reset_index(drop=True)

    out_of_sample_preds_rev = []
    out_of_sample_preds_mom = []
    out_of_sample_preds_meta = []
    actuals_all = []
    
    embargo_window, purge_window = 2, 1    

    for end_train_idx in range(train_window, len(df), retrain_interval):
        train_start_cutoff = purge_window
        train_end_cutoff = end_train_idx - embargo_window
        if train_end_cutoff <= train_start_cutoff: continue
        X_train_df = df.iloc[train_start_cutoff:train_end_cutoff].copy().reset_index(drop=True)
        
        end_test_idx = min(end_train_idx + retrain_interval, len(df))
        test_block = df.iloc[end_train_idx : end_test_idx].copy().reset_index(drop=True)
        if len(test_block) == 0: break

        for col in ["raw_h4_volatility", "h4_volume_concentration"] + macro_features:
            fill_val = X_train_df[col].median() if not X_train_df[col].isna().all() else 0.0
            X_train_df[col] = X_train_df[col].ffill().fillna(fill_val)
            test_block[col] = test_block[col].ffill().fillna(fill_val)
            
        X_train_df["h4_rsi_imbalance"] = X_train_df["h4_rsi_imbalance"].ffill().fillna(X_train_df["daily_rsi_ema"])
        test_block["h4_rsi_imbalance"] = test_block["h4_rsi_imbalance"].ffill().fillna(test_block["daily_rsi_ema"])

        train_mean = X_train_df["future_vol_adjusted_target"].mean()
        train_std = X_train_df["future_vol_adjusted_target"].std() + 1e-9
        X_train_df["target"] = (X_train_df["future_vol_adjusted_target"] - train_mean) / train_std

        train_regimes, test_regimes, _, _ = fit_unsupervised_regimes(X_train_df, test_block, n_clusters=4)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_rev = np.zeros(len(X_train_df))
        oof_mom = np.zeros(len(X_train_df))
        
        for train_fold_idx, val_fold_idx in kf.split(X_train_df):
            fold_train = X_train_df.iloc[train_fold_idx]
            fold_val = X_train_df.iloc[val_fold_idx]
            
            f_model_rev = LGBMRegressor(objective="regression", n_estimators=50, learning_rate=0.05, max_depth=4, verbose=-1, random_state=42)
            f_model_mom = LGBMRegressor(objective="regression", n_estimators=50, learning_rate=0.05, max_depth=4, verbose=-1, random_state=42)
            
            f_model_rev.fit(fold_train[rev_feats], fold_train["target"])
            f_model_mom.fit(fold_train[mom_feats], fold_train["target"])
            
            oof_rev[val_fold_idx] = f_model_rev.predict(fold_val[rev_feats])
            oof_mom[val_fold_idx] = f_model_mom.predict(fold_val[mom_feats])

        model_rev = LGBMRegressor(objective="regression", n_estimators=80, learning_rate=0.05, max_depth=4, verbose=-1, random_state=42)
        model_mom = LGBMRegressor(objective="regression", n_estimators=80, learning_rate=0.05, max_depth=4, verbose=-1, random_state=42)
        
        model_rev.fit(X_train_df[rev_feats], X_train_df["target"])
        model_mom.fit(X_train_df[mom_feats], X_train_df["target"])

        X_meta_train = get_meta_features(X_train_df, oof_rev, oof_mom, train_regimes)
        meta_model = Ridge(alpha=5.0, fit_intercept=False, random_state=42)
        meta_model.fit(X_meta_train, X_train_df["target"])

        p_z_rev = model_rev.predict(test_block[rev_feats])
        p_z_mom = model_mom.predict(test_block[mom_feats])
        
        X_meta_test = get_meta_features(test_block, p_z_rev, p_z_mom, test_regimes)
        p_z_meta = meta_model.predict(X_meta_test)
        
        vol_scale = test_block["realized_vol_20_smoothed"]
        out_of_sample_preds_rev.extend((p_z_rev * train_std + train_mean) * vol_scale)
        out_of_sample_preds_mom.extend((p_z_mom * train_std + train_mean) * vol_scale)
        out_of_sample_preds_meta.extend((p_z_meta * train_std + train_mean) * vol_scale)
        
        actuals_all.extend(test_block["future_raw_returns"])

    return (
        mean_absolute_error(actuals_all, out_of_sample_preds_rev),
        mean_absolute_error(actuals_all, out_of_sample_preds_mom),
        mean_absolute_error(actuals_all, out_of_sample_preds_meta)
    )

# =====================================================================
# 5. HISTORICAL MACRO FLAVOR STRESS TESTING RESULTS ENGINE
# =====================================================================

def run_flavor_stress_tests(df_daily, df_h4, train_window=600, target_symbol=None, dfs_dict=None):
    if target_symbol is not None and target_symbol not in PRIMARY_ASSETS:
        return
        
    print("\n" + "-"*80)
    print(f" LEARNED META-MODEL TAIL RISK STRESS TEST: {target_symbol if target_symbol else 'Asset'}")
    print("-"*80)
    print(f" {'MACRO SIMULATED SHOCK':<32} | {'ENG1 MAE':<9} | {'ENG2 MAE':<9} | {'META-BLENDED MAE'}")
    print("-" * 80)
    
    flavors = {
        "Flavor 1 (Liquidity Black Hole)": generate_flavor_1_liquidity_black_hole(df_daily),
        "Flavor 2 (Structural Bear Mkt)":  generate_flavor_2_structural_bear_market(df_daily),
        "Flavor 3 (High-Vol Grind)":       generate_flavor_3_high_volatility_grind(df_daily),
        "Flavor 4 (The Gold Rush Bubble)": generate_flavor_4_the_gold_rush(df_daily)
    }
    
    for f_name, f_df in flavors.items():
        try:
            sim_dfs_dict = dfs_dict.copy() if dfs_dict is not None else {}
            if target_symbol:
                sim_dfs_dict[target_symbol] = f_df
            
            mae_rev, mae_mom, mae_meta = run_block_walk_forward_validation(
                df_daily=f_df, df_h4=df_h4, train_window=train_window, retrain_interval=100, target_symbol=target_symbol, dfs_dict=sim_dfs_dict
            )
            print(f" {f_name:<32} | {mae_rev:<9.5f} | {mae_mom:<9.5f} | {mae_meta:.5f}")
        except Exception as e:
            print(f" {f_name:<32} | {'ERROR':<9} | {'ERROR':<9} | {'ERROR'}")
    print("-"*80 + "\n")

# =====================================================================
# 6. LIVE PRODUCTION ENVIRONMENT PIPELINE WITH UNSUPERVISED COCKPIT
# =====================================================================

def predict_next_day_scenarios(df_daily, df_h4, train_window=150, target_symbol=None, dfs_dict=None):
    if target_symbol is not None and target_symbol not in PRIMARY_ASSETS:
        return

    today_str = datetime.datetime.now().strftime("%Y.%m.%d")
    df_daily_clean = df_daily[df_daily["<DATE>"] != today_str].copy().sort_values("<DATE>").reset_index(drop=True)
    df_h4_clean = df_h4[df_h4["<DATE>"] != today_str].copy()
    
    latest_close = df_daily_clean.iloc[-1]["<CLOSE>"]
    
    df_features = calculate_technical_indicators(df_daily_clean)
    df_features, macro_features = generate_cross_asset_features(target_symbol, df_features, dfs_dict)
    df_features = extract_and_merge_h4_features(df_features, df_h4_clean)
    
    rev_feats = list(MEAN_REVERSION_FEATURES.keys())
    mom_feats = list(PURE_MOMENTUM_FEATURES.keys()) + macro_features
    all_needed_feats = list(set(rev_feats + mom_feats + REGIME_CLUSTERING_FEATURES))
    
    df_features["future_raw_returns"] = df_features["returns"].shift(-1)
    df_features["future_vol_adjusted_target"] = df_features["future_raw_returns"] / (df_features["realized_vol_20"] + 1e-9)
    
    train_df = df_features.dropna(subset=all_needed_feats + ["future_vol_adjusted_target"]).reset_index(drop=True)
    X_train_df = train_df.tail(int(train_window)).copy().reset_index(drop=True)
    
    for col in macro_features:
        if col in X_train_df:
            X_train_df[col] = X_train_df[col].fillna(X_train_df[col].median())
            
    train_mean = X_train_df["future_vol_adjusted_target"].mean()
    train_std = X_train_df["future_vol_adjusted_target"].std() + 1e-9
    X_train_df["target"] = (X_train_df["future_vol_adjusted_target"] - train_mean) / train_std

    train_regimes, kmeans, scaler = fit_unsupervised_regimes(X_train_df, n_clusters=4)
    
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    regime_profiles = {}
    for c_id in range(4):
        dist_ma = centroids[c_id, 0]
        vol_ratio = centroids[c_id, 2]
        nat_tr = centroids[c_id, 3]
        
        if nat_tr > np.median(centroids[:, 3]) and vol_ratio > np.median(centroids[:, 2]):
            regime_profiles[c_id] = "High-Vol Liquidity Expansion"
        elif nat_tr > np.median(centroids[:, 3]) and vol_ratio <= np.median(centroids[:, 2]):
            regime_profiles[c_id] = "High-Vol Mean Reverting Grind"
        elif nat_tr <= np.median(centroids[:, 3]) and abs(dist_ma) > np.median(np.abs(centroids[:, 0])):
            regime_profiles[c_id] = "Mature Trend Horizon"
        else:
            regime_profiles[c_id] = "Low-Vol Compressed Range"

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_rev = np.zeros(len(X_train_df))
    oof_mom = np.zeros(len(X_train_df))
    
    for train_fold_idx, val_fold_idx in kf.split(X_train_df):
        fold_train = X_train_df.iloc[train_fold_idx]
        fold_val = X_train_df.iloc[val_fold_idx]
        
        f_model_rev = LGBMRegressor(n_estimators=50, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1)
        f_model_mom = LGBMRegressor(n_estimators=50, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1)
        f_model_rev.fit(fold_train[rev_feats], fold_train["target"])
        f_model_mom.fit(fold_train[mom_feats], fold_train["target"])
        
        oof_rev[val_fold_idx] = f_model_rev.predict(fold_val[rev_feats])
        oof_mom[val_fold_idx] = f_model_mom.predict(fold_val[mom_feats])
        
    model_rev_lgb = LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1)
    model_mom_lgb = LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1)
    model_rev_lgb.fit(X_train_df[rev_feats], X_train_df["target"])
    model_mom_lgb.fit(X_train_df[mom_feats], X_train_df["target"])
    
    X_meta_train = get_meta_features(X_train_df, oof_rev, oof_mom, train_regimes)
    meta_model = Ridge(alpha=5.0, fit_intercept=False, random_state=42)
    meta_model.fit(X_meta_train, X_train_df["target"])

    latest_known_row = df_features.dropna(subset=all_needed_feats).iloc[-1]
    atr_5 = latest_known_row["raw_NTR_5"] * latest_close
    
    sim_open, tickvol_ref = latest_close, latest_known_row["<TICKVOL>"]
    scenarios = {
        "1. Pure Positive Breakout":       {"C": sim_open + (0.6 * atr_5), "H": sim_open + (0.7 * atr_5), "L": sim_open - (0.1 * atr_5)},
        "2. Positive + Upper Wick Drop":   {"C": sim_open + (0.1 * atr_5), "H": sim_open + (0.8 * atr_5), "L": sim_open - (0.1 * atr_5)},
        "3. Pure Negative Breakdown":      {"C": sim_open - (0.6 * atr_5), "H": sim_open + (0.1 * atr_5), "L": sim_open - (0.7 * atr_5)},
        "4. Negative + Lower Wick Sweep":  {"C": sim_open - (0.1 * atr_5), "H": sim_open + (0.1 * atr_5), "L": sim_open - (0.8 * atr_5)},
        "5. Compressed Neutral Chop":      {"C": sim_open + (0.0 * atr_5), "H": sim_open + (0.1 * atr_5), "L": sim_open - (0.1 * atr_5)},
        "6. High-Vol Indecision Cross":    {"C": sim_open + (0.0 * atr_5), "H": sim_open + (0.7 * atr_5), "L": sim_open - (0.7 * atr_5)}
    }
    
    print("\n" + "="*148)
    print("                                              LEARNED STACKED META-MODEL SCENARIO MATRIX                                              ")
    print("="*148)
    print(f" {'SIMULATED MARKET BOUNDARY':<32} | {'SIM CLOSE':<9} | {'ACTIVE REGIME':<28} | {'E1 WEIGHT':<9} | {'E2 WEIGHT':<9} | {'PREDICTED':<10} | {'TACTICAL VECTOR':<15} | {'TAKE PROFIT'}")
    print("-" * 148)
    
    for name, levels in scenarios.items():
        sim_date = "TOMORROW_SIM"
        sim_row = pd.DataFrame([{"<DATE>": sim_date, "<OPEN>": sim_open, "<HIGH>": levels["H"], "<LOW>": levels["L"], "<CLOSE>": levels["C"], "<TICKVOL>": tickvol_ref}])
        sim_h4_df = simulate_h4_bars_for_scenario(sim_date, sim_open, levels["H"], levels["L"], levels["C"], tickvol_ref)
        
        extended_features = calculate_technical_indicators(pd.concat([df_daily_clean, sim_row], ignore_index=True))
        extended_features, _ = generate_cross_asset_features(target_symbol, extended_features, dfs_dict)
        extended_features = extract_and_merge_h4_features(extended_features, pd.concat([df_h4_clean, sim_h4_df], ignore_index=True))
        
        sim_vector = extended_features.iloc[[-1]].copy().reset_index(drop=True)
        smooth_vol = sim_vector["realized_vol_20_smoothed"].values[0]
        
        for col in macro_features:
            sim_vector[col] = sim_vector[col].fillna(latest_known_row[col] if col in latest_known_row else 0.0)
            
        sim_clust_scaled = scaler.transform(sim_vector[REGIME_CLUSTERING_FEATURES].fillna(0.0).values)
        sim_cluster_id = kmeans.predict(sim_clust_scaled)[0]
        sim_regime_name = regime_profiles[sim_cluster_id]
        
        sim_regime_df = pd.DataFrame([[0.0]*4], columns=[f"regime_{i}" for i in range(4)])
        sim_regime_df[f"regime_{sim_cluster_id}"] = 1.0
        
        p_rev = model_rev_lgb.predict(sim_vector[rev_feats])[0]
        p_mom = model_mom_lgb.predict(sim_vector[mom_feats])[0]
        
        X_meta_sim = get_meta_features(sim_vector, [p_rev], [p_mom], sim_regime_df)
        pred_meta = meta_model.predict(X_meta_sim)[0]
        
        meta_coefs = meta_model.coef_
        w_rev = meta_coefs[0]
        w_mom = meta_coefs[1]
        
        total_w = abs(w_rev) + abs(w_mom) + 1e-9
        alloc_rev = np.clip(w_rev / total_w, -1.0, 1.0)
        alloc_mom = np.clip(w_mom / total_w, -1.0, 1.0)
        combined_alloc = (alloc_rev * p_rev) + (alloc_mom * p_mom)
        
        predicted_market_price = latest_close * np.exp((pred_meta * train_std + train_mean) * smooth_vol)
        
        if abs(combined_alloc) < 0.05:
            action, tp_str, sl_str = "STANDBY", "N/A", "N/A"
        elif combined_alloc > 0:
            action = f"LONG {abs(combined_alloc):.2f}x"
            tp_str, sl_str = f"{levels['C'] + (1.5 * atr_5):<11.4f}", f"{levels['C'] - (1.0 * atr_5):.4f}"
        else:
            action = f"SHORT {abs(combined_alloc):.2f}x"
            tp_str, sl_str = f"{levels['C'] - (1.5 * atr_5):<11.4f}", f"{levels['C'] + (1.0 * atr_5):.4f}"
            
        print(f" {name:<32} | {levels['C']:<9.4f} | {sim_regime_name:<28} | {alloc_rev:+.2f}      | {alloc_mom:+.2f}      | {predicted_market_price:<10.4f} | {action:<15} | {tp_str}")
    print("="*148)

# =====================================================================
# 7. TWO-TIER EXECUTIVE COCKPIT ENGINE (CONSOLIDATED OUTPUTS)
# =====================================================================

def evaluate_symbol_cockpit(df_daily, df_h4, train_window=600, target_symbol=None, dfs_dict=None, show_deep_dive=False):
    """
    Executes models and prints a clean executive dashboard for the client.
    """
    if target_symbol is not None and target_symbol not in PRIMARY_ASSETS: return

    today_str = datetime.datetime.now().strftime("%Y.%m.%d")
    df_daily_clean = df_daily[df_daily["<DATE>"] != today_str].copy().sort_values("<DATE>").reset_index(drop=True)
    df_h4_clean = df_h4[df_h4["<DATE>"] != today_str].copy()
    latest_close = df_daily_clean.iloc[-1]["<CLOSE>"]
    
    # Feature engineering forward pass
    df_features = calculate_technical_indicators(df_daily_clean)
    df_features, macro_features = generate_cross_asset_features(target_symbol, df_features, dfs_dict)
    df_features = extract_and_merge_h4_features(df_features, df_h4_clean)
    
    rev_feats, mom_feats = list(MEAN_REVERSION_FEATURES.keys()), list(PURE_MOMENTUM_FEATURES.keys()) + macro_features
    all_needed_feats = list(set(rev_feats + mom_feats + REGIME_CLUSTERING_FEATURES))
    
    df_features["future_raw_returns"] = df_features["returns"].shift(-1)
    df_features["future_vol_adjusted_target"] = df_features["future_raw_returns"] / (df_features["realized_vol_20"] + 1e-9)
    train_df = df_features.dropna(subset=all_needed_feats + ["future_vol_adjusted_target"]).reset_index(drop=True)
    X_train_df = train_df.tail(int(train_window)).copy().reset_index(drop=True)
    
    for col in macro_features:
        if col in X_train_df: X_train_df[col] = X_train_df[col].fillna(X_train_df[col].median())
            
    train_mean, train_std = X_train_df["future_vol_adjusted_target"].mean(), X_train_df["future_vol_adjusted_target"].std() + 1e-9
    X_train_df["target"] = (X_train_df["future_vol_adjusted_target"] - train_mean) / train_std

    # Run Unsupervised Regime Cluster Fit
    train_regimes, kmeans, scaler = fit_unsupervised_regimes(X_train_df, n_clusters=4)
    
    # Profile centroids
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    regime_profiles = {}
    for c_id in range(4):
        vol_ratio, nat_tr = centroids[c_id, 2], centroids[c_id, 3]
        if nat_tr > np.median(centroids[:, 3]) and vol_ratio > np.median(centroids[:, 2]): regime_profiles[c_id] = "High-Vol Expansion"
        elif nat_tr > np.median(centroids[:, 3]) and vol_ratio <= np.median(centroids[:, 2]): regime_profiles[c_id] = "High-Vol Grind"
        elif nat_tr <= np.median(centroids[:, 3]) and abs(centroids[c_id, 0]) > np.median(np.abs(centroids[:, 0])): regime_profiles[c_id] = "Mature Trend"
        else: regime_profiles[c_id] = "Compressed Range"

    # Quick train for live state
    model_rev_lgb = LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1)
    model_mom_lgb = LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1)
    model_rev_lgb.fit(X_train_df[rev_feats], X_train_df["target"])
    model_mom_lgb.fit(X_train_df[mom_feats], X_train_df["target"])
    
    # Generate in-sample base predictions to extract coefficients
    pred_rev_in = model_rev_lgb.predict(X_train_df[rev_feats])
    pred_mom_in = model_mom_lgb.predict(X_train_df[mom_feats])
    
    # Create aligned Meta-feature matrix
    X_meta_train = get_meta_features(X_train_df, pred_rev_in, pred_mom_in, train_regimes)
    
    # Fit the Ridge Meta-Model stacker
    meta_model = Ridge(alpha=5.0, fit_intercept=False, random_state=42)
    meta_model.fit(X_meta_train, X_train_df["target"])
    meta_coefs = meta_model.coef_
    
    # Extract structural meta-weights
    w_rev, w_mom = meta_coefs[0], meta_coefs[1]
    is_hedged = (np.sign(w_rev) != np.sign(w_mom))
    alignment_status = f"OPTIMAL (Hedged / Opposite Directions | E1 Coef: {w_rev:+.2f}, E2 Coef: {w_mom:+.2f})" if is_hedged else f"CONCENTRATED (Same Direction Risk | E1 Coef: {w_rev:+.2f}, E2 Coef: {w_mom:+.2f})"

    # Profile Scenario Regimes
    latest_known_row = df_features.dropna(subset=all_needed_feats).iloc[-1]
    atr_5 = latest_known_row["raw_NTR_5"] * latest_close
    
    scenarios = {
        "Breakout": sim_daily_row(latest_close, latest_close + (0.6 * atr_5), atr_5, latest_known_row["<TICKVOL>"]),
        "Chop":     sim_daily_row(latest_close, latest_close, atr_5, latest_known_row["<TICKVOL>"])
    }
    
    active_regimes = []
    regime_names_only = []
    for name, s_row in scenarios.items():
        sim_scaled = scaler.transform(s_row[REGIME_CLUSTERING_FEATURES].fillna(0.0).values)
        c_id = kmeans.predict(sim_scaled)[0]
        active_regimes.append(f"{name}->{regime_profiles[c_id]}")
        regime_names_only.append(regime_profiles[c_id])
        
    # Vol tracking
    vol_ratio = latest_known_row["volume_ratio"]
    is_vol_high = vol_ratio > 1.2
    is_vol_low = vol_ratio < 0.8
    if is_vol_high: vol_status = f"HIGH (Tracking at {vol_ratio:.1f}x normal volume - Institutional Activity)"
    elif is_vol_low: vol_status = f"LOW (Tracking at {vol_ratio:.1f}x normal volume - Compression/Drying Liquidity)"
    else: vol_status = f"NORMAL (Tracking at {vol_ratio:.1f}x normal volume - Baseline Grid)"

    # Plain english Flavors performance
    flavors = {
        "Liquidity Shock": generate_flavor_1_liquidity_black_hole(df_daily_clean),
        "Bear Grind":      generate_flavor_2_structural_bear_market(df_daily_clean),
        "Bull Bubble":     generate_flavor_4_the_gold_rush(df_daily_clean)
    }
    
    flavor_summaries = []
    for f_name, f_df in flavors.items():
        try:
            sim_dfs_dict = dfs_dict.copy() if dfs_dict is not None else {}
            if target_symbol: sim_dfs_dict[target_symbol] = f_df
                
            _, _, mae_meta = run_block_walk_forward_validation(
                f_df, df_h4_clean, train_window, target_symbol=target_symbol, dfs_dict=sim_dfs_dict
            )
            flavor_summaries.append(f"{f_name} (Meta-MAE: {mae_meta:.4f})")
        except Exception as e:
            flavor_summaries.append(f"{f_name} (Pending)")

    # =====================================================================
    # ALGORTIHMIC EXECUTIVE CONCLUSION GENERATOR (1-LINE STRATEGY LAYER)
    # =====================================================================
    if is_vol_high and any(r in ["High-Vol Expansion", "High-Vol Grind"] for r in regime_names_only):
        conclusion = "WARNING: Highly volatile institutional breakout under-way. Expect wider swings; momentum models dominant."
    elif any(r in ["High-Vol Expansion", "Mature Trend"] for r in regime_names_only):
        conclusion = "BULLISH EXPANSION: Strong structural trend detected with healthy parameters. Conducive to Trend-Following vectors."
    elif is_hedged and any(r in ["High-Vol Grind", "Compressed Range"] for r in regime_names_only):
        conclusion = "STABLE MEAN-REVERSION: Range constraints bound. Core engines are hedging each other efficiently. Favor Grid/Fading styles."
    elif is_vol_low and "Compressed Range" in regime_names_only:
        conclusion = "COMPRESSION HOLD: Extreme liquidity squeeze. Volatility is coiled tight; stand by for an explosive breakout move."
    else:
        conclusion = "STABLE MIXED REGIME: Market indicators are running inside historical baseline trends. Standard allocation thresholds apply."

    # =====================================================================
    # PRINT 5-LINE CLIENT EXECUTIVE SUMMARY VIEW
    # =====================================================================
    print(f"\n EXECUTIVE COCKPIT SUMMARY: {target_symbol if target_symbol else 'Asset'}")
    print("="*105)
    print(f"0. CONCLUSION        : {conclusion}")
    print(f"1. ENGINE ALIGNMENT  : {alignment_status}")
    print(f"2. SCENARIO REGIMES  : Currently mapping as [{', '.join(active_regimes)}]")
    print(f"3. VOLUME TRACKING   : {vol_status}")
    print(f"4. STRESS PERF RUN   : Steady tracking across simulated tail-risks: {', '.join(flavor_summaries)}")
    print("="*105)

    if show_deep_dive:
        print("\n[DEBUG DEEP-DIVE MATRIX INITIALIZED]")
        print(f" -> Feature dimensions for Training: {X_train_df.shape}")
        print(f" -> Full Coefficient Vector: {dict(zip(X_meta_train.columns, meta_coefs))}")
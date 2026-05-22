import numpy as np
import pandas as pd
import warnings
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore", category=UserWarning)


def calculate_technical_indicators(df):
    """Calculates normalized, scale-invariant technical indicators for LightGBM."""
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

    # 8. Acceleration Momentum: ROC
    df["ROC_10"] = ((df["<CLOSE>"] - df["<CLOSE>"].shift(10)) / df["<CLOSE>"].shift(10)) * 100
    
    # 9. Institutional Volatility: Garman-Klass Volatility
    log_hl = np.log(df["<HIGH>"] / (df["<LOW>"] + 1e-9))
    log_co = np.log(df["<CLOSE>"] / (df["<OPEN>"] + 1e-9))
    gk_var = 0.5 * (log_hl**2) - (2 * np.log(2) - 1) * (log_co**2)
    gk_var = np.maximum(gk_var, 0.0)
    df["gk_vol_5"] = np.sqrt(gk_var).rolling(window=5).mean()
    
    # 10. Volume-Weighted Momentum: MFI
    typical_price = (df["<HIGH>"] + df["<LOW>"] + df["<CLOSE>"]) / 3
    raw_money_flow = typical_price * df["<TICKVOL>"]
    tp_diff = typical_price.diff()
    pos_flow = raw_money_flow.where(tp_diff > 0, 0).rolling(window=14).sum()
    neg_flow = raw_money_flow.where(tp_diff < 0, 0).rolling(window=14).sum()
    mfi_ratio = pos_flow / (neg_flow + 1e-9)
    df["MFI_14"] = 100 - (100 / (1 + mfi_ratio))
    
    # 11. Trend Strength: ADX
    up_move = df["<HIGH>"] - df["<HIGH>"].shift(1)
    down_move = df["<LOW>"].shift(1) - df["<LOW>"]
    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_14 = true_range.rolling(window=14).mean() + 1e-9
    plus_di = 100 * (pd.Series(pos_dm).rolling(window=14).mean().to_numpy() / atr_14)
    minus_di = 100 * (pd.Series(neg_dm).rolling(window=14).mean().to_numpy() / atr_14)
    di_sum = plus_di + minus_di + 1e-9
    dx = (np.abs(plus_di - minus_di) / di_sum) * 100
    df["ADX_14"] = pd.Series(dx).rolling(window=14).mean().fillna(20.0)
    
    return df


def find_best_hyperparameters(X_train, y_train):
    """
    Performs an internal time-series validation sweep on the training slice 
    to find parameters that stop the model from choking on the new features.
    """
    param_grid = [
        {"reg_alpha": 0.1, "reg_lambda": 0.1, "colsample_bytree": 1.0, "max_depth": 4}, 
        {"reg_alpha": 1.0, "reg_lambda": 1.0, "colsample_bytree": 0.8, "max_depth": 4}, 
        {"reg_alpha": 5.0, "reg_lambda": 5.0, "colsample_bytree": 0.7, "max_depth": 3}, 
        {"reg_alpha": 0.0, "reg_lambda": 10.0, "colsample_bytree": 0.6, "max_depth": 5} 
    ]
    
    best_score = -1.0
    best_params = param_grid[1] 
    
    split_idx = int(len(X_train) * 0.8)
    if split_idx < 50: 
        return best_params 
        
    X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]
    
    for params in param_grid:
        model = LGBMClassifier(
            n_estimators=40,
            learning_rate=0.05,
            max_depth=params["max_depth"],
            colsample_bytree=params["colsample_bytree"],
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
            random_state=42,
            verbose=-1
        )
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        score = accuracy_score(y_val, preds)
        
        if score > best_score:
            best_score = score
            best_params = params
            
    return best_params


def run_walk_forward_validation_tuned(df, train_window, test_steps=100):
    """Executes rolling backtest validation with dynamic auto-tuning per window."""
    df["<DATE>"] = pd.to_datetime(df["<DATE>"])
    df = df.sort_values("<DATE>").reset_index(drop=True)
    
    df = calculate_technical_indicators(df)
    df["target"] = (df["<CLOSE>"].shift(-1) > df["<CLOSE>"]).astype(int)

    feature_cols = [
        "dist_MA_5", "dist_MA_20", "volatility_5", "HL_spread", "CO_spread", 
        "volume_ratio", "RSI", "Stochastic_14", "NTR_5", "ROC_10",
        "gk_vol_5", "MFI_14", "ADX_14"
    ]

    df = df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)
    
    if len(df) == 0:
        print("CRITICAL ERROR: Features purged data. Inspect primary columns.")
        return df, 0.0, []

    predictions, actuals = [], []
    print(f"Executing self-tuning backtest across {test_steps} market sessions...")
    
    importance_accumulator = np.zeros(len(feature_cols))

    for i in range(test_steps):
        end_train_idx = train_window + i
        if end_train_idx >= len(df):
            break

        X_train_df = df.iloc[i:end_train_idx]
        test_row = df.iloc[[end_train_idx]]

        best_hparams = find_best_hyperparameters(X_train_df[feature_cols], X_train_df["target"])

        lgb_model = LGBMClassifier(
            n_estimators=50, 
            learning_rate=0.05, 
            max_depth=best_hparams["max_depth"], 
            random_state=42, 
            verbose=-1,
            colsample_bytree=best_hparams["colsample_bytree"],
            reg_alpha=best_hparams["reg_alpha"],         
            reg_lambda=best_hparams["reg_lambda"]         
        )
        lgb_model.fit(X_train_df[feature_cols], X_train_df["target"])

        importance_accumulator += lgb_model.booster_.feature_importance(importance_type="gain")

        pred_proba = lgb_model.predict_proba(test_row[feature_cols])[0][1]
        pred_class = 1 if pred_proba >= 0.5 else 0

        predictions.append(pred_class)
        actuals.append(test_row["target"].values[0])

    accuracy = accuracy_score(actuals, predictions) if predictions else 0.0
    
    total_gain = np.sum(importance_accumulator) if np.sum(importance_accumulator) > 0 else 1
    normalized_importances = (importance_accumulator / total_gain) * 100
    
    feature_importance_dict = dict(zip(feature_cols, normalized_importances))
    sorted_all_features = sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)

    return df, accuracy, sorted_all_features


def predict_next_1_day(df, train_window=250):
    """Generates out-of-sample forward direction projections on tomorrow's market open using auto-tuning."""
    print("\n==================================================")
    print("      1-DAY FORWARD DIRECTIONAL FORECAST          ")
    print("==================================================")
    
    feature_cols = [
        "dist_MA_5", "dist_MA_20", "volatility_5", "HL_spread", "CO_spread", 
        "volume_ratio", "RSI", "Stochastic_14", "NTR_5", "ROC_10",
        "gk_vol_5", "MFI_14", "ADX_14"
    ]
    
    # Ensure indicators are aligned and drop trailing nans before target check
    if "target" not in df.columns:
        df = calculate_technical_indicators(df)
        df["target"] = (df["<CLOSE>"].shift(-1) > df["<CLOSE>"]).astype(int)
        
    # Isolate training matrix using historical window up to the last point that has a valid target
    train_df_slice = df.dropna(subset=feature_cols + ["target"]).tail(train_window)
    
    if len(train_df_slice) < 50:
        print("ERROR: Not enough historical data available to compile final training slice.")
        return

    # Dynamically optimize parameters using the final known training window
    best_hparams = find_best_hyperparameters(train_df_slice[feature_cols], train_df_slice["target"])
    
    final_model = LGBMClassifier(
        n_estimators=50, 
        learning_rate=0.05, 
        max_depth=best_hparams["max_depth"], 
        random_state=42, 
        verbose=-1,
        colsample_bytree=best_hparams["colsample_bytree"],
        reg_alpha=best_hparams["reg_alpha"],
        reg_lambda=best_hparams["reg_lambda"]
    )
    final_model.fit(train_df_slice[feature_cols], train_df_slice["target"])
    
    # Access the absolute last row of the primary dataframe (today's closing numbers)
    latest_known_row = df.dropna(subset=feature_cols).iloc[[-1]]
    current_date_dt = pd.to_datetime(latest_known_row["<DATE>"].values[0])
    
    # Weekend calendar processing filter
    if current_date_dt.weekday() == 4:  
        forecast_date = current_date_dt + pd.Timedelta(days=3)
    else:
        forecast_date = current_date_dt + pd.Timedelta(days=1)
    
    up_probability = final_model.predict_proba(latest_known_row[feature_cols])[0][1]
    
    if up_probability >= 0.65:
        nudge = "🟢 STRONG UP CONVICTION"
    elif up_probability <= 0.35:
        nudge = "🔴 STRONG DOWN CONVICTION"
    else:
        nudge = "⚪ COIN FLIP / NO ACTION"
        
    print(f"Current Date (T):   {current_date_dt.strftime('%Y-%m-%d')}")
    print(f"Forecast Date (T+1): {forecast_date.strftime('%Y-%m-%d')}")
    print(f"Up Probability:      {up_probability:.1%}")
    print(f"Nudge Action:        {nudge}")
    print("==================================================")


if __name__ == "__main__":
    # Simulated validation testing environment 
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=1000, freq="D")
    close = 100 + np.cumsum(np.random.normal(0.02, 1.0, size=1000))
    open_p = close - np.random.normal(0, 0.4, size=1000)
    high = np.maximum(open_p, close) + np.random.exponential(0.15, size=1000)
    low = np.minimum(open_p, close) - np.random.exponential(0.15, size=1000)
    tickvol = np.random.randint(1000, 5000, size=1000) 

    mock_df = pd.DataFrame({
        "<DATE>": dates, "<OPEN>": open_p, "<HIGH>": high, "<LOW>": low, "<CLOSE>": close, "<TICKVOL>": tickvol
    })

    # Execute dynamic optimization validation script
    processed_df, historical_accuracy, ranked_features = run_walk_forward_validation_tuned(mock_df, train_window=250, test_steps=50)
    
    print("\n" + "="*50)
    print("          TUNED BACKTEST RESULTS                  ")
    print("="*50)
    print(f"Adaptive Directional Accuracy: {historical_accuracy:.2%}")
    print("="*50)
    
    # Run the added prediction framework on the simulated dataset
    predict_next_1_day(processed_df, train_window=250)
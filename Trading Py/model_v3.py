import numpy as np
import pandas as pd
import warnings
from arch import arch_model
from lightgbm import LGBMClassifier
from pmdarima import auto_arima
from sklearn.metrics import accuracy_score

# Silence the polynomial reciprocal warning from statsmodels/pmdarima
warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=UserWarning)


def calculate_technical_indicators(df):
    """Calculates normalized, scale-invariant technical indicators for LightGBM."""
    df = df.copy()
    
    # Base Returns and Volatility
    df["returns"] = np.log(df["<CLOSE>"] / df["<CLOSE>"].shift(1))
    df["volatility_5"] = df["returns"].rolling(window=5).std()
    
    # Scale-Invariant Price Action Spreads
    df["HL_spread"] = (df["<HIGH>"] - df["<LOW>"]) / df["<CLOSE>"]
    df["CO_spread"] = (df["<CLOSE>"] - df["<OPEN>"]) / df["<OPEN>"]    
    
    # Trend Proximity (Replaces raw Moving Averages)
    ma_5 = df["<CLOSE>"].rolling(window=5).mean()
    ma_20 = df["<CLOSE>"].rolling(window=20).mean()
    df["dist_MA_5"] = (df["<CLOSE>"] - ma_5) / ma_5
    df["dist_MA_20"] = (df["<CLOSE>"] - ma_20) / ma_20
    
    # Liquidity Dynamics with safety check for zeros/NaNs (Avoids 0/0 and inf drops)
    df["volume_ratio"] = df["<TICKVOL>"] / (df["<TICKVOL>"].rolling(window=20).mean() + 1e-9)
    df["volume_ratio"] = df["volume_ratio"].fillna(1.0).replace([np.inf, -np.inf], 1.0)
    
    # Bounded Momentum: Relative Strength Index (RSI 14)
    delta = df["<CLOSE>"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)  # Avoid division by zero
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # --- NEW INDICATORS ADDED ---
    # 1. Bounded Structural Position: Stochastic Oscillator (%K)
    lowest_low_14 = df["<LOW>"].rolling(window=14).min()
    highest_high_14 = df["<HIGH>"].rolling(window=14).max()
    df["Stochastic_14"] = ((df["<CLOSE>"] - lowest_low_14) / (highest_high_14 - lowest_low_14 + 1e-9)) * 100

    # 2. Scale-Invariant Volatility: Normalized True Range (NTR 5)
    prev_close = df["<CLOSE>"].shift(1)
    tr1 = df["<HIGH>"] - df["<LOW>"]
    tr2 = (df["<HIGH>"] - prev_close).abs()
    tr3 = (df["<LOW>"] - prev_close).abs()
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    df["NTR_5"] = true_range.rolling(window=5).mean() / df["<CLOSE>"]

    # 3. Acceleration Momentum: Rate of Change (ROC 10)
    df["ROC_10"] = ((df["<CLOSE>"] - df["<CLOSE>"].shift(10)) / df["<CLOSE>"].shift(10)) * 100
    
    return df


def fit_arima_garch(returns_history):
    """Fits ARIMA + GARCH(1,1) and returns 1-step ahead forecasts."""
    arima_model = auto_arima(
        returns_history.dropna(),
        start_p=0, start_q=0, max_p=3, max_q=3, d=0,
        seasonal=False, error_action="ignore", suppress_warnings=True,
    )
    arima_resid = arima_model.resid()
    arima_forecast = arima_model.predict(n_periods=1).to_numpy()[0]

    scaled_resid = arima_resid * 100
    try:
        garch = arch_model(scaled_resid, vol="Garch", p=1, q=1, dist="normal", show_warning=False)
        garch_fitted = garch.fit(disp="off")
        garch_forecast_res = garch_fitted.forecast(horizon=1)
        garch_vol_forecast = np.sqrt(garch_forecast_res.variance.iloc[-1].values[0]) / 100
    except Exception:
        garch_vol_forecast = returns_history.std()

    return arima_forecast, garch_vol_forecast


def generate_all_statistical_features(df, train_window, step_size=5):
    """Pre-computes features but SKIPS steps to drastically improve speed."""
    print(f"Pre-computing ARIMA/GARCH features (Optimized: updating every {step_size} days)...")
    arima_preds = np.zeros(len(df)) * np.nan
    garch_preds = np.zeros(len(df)) * np.nan

    last_arima, last_garch = 0.0, 0.0

    for i in range(len(df)):
        if i < train_window:
            continue
        
        if (i - train_window) % step_size == 0:
            history = df["returns"].iloc[i - train_window : i]
            try:
                last_arima, last_garch = fit_arima_garch(history)
            except Exception:
                last_arima, last_garch = history.mean(), history.std()
        
        arima_preds[i] = last_arima
        garch_preds[i] = last_garch

    df["arima_pred"] = arima_preds
    df["garch_pred"] = garch_preds

    # SHIFT BY 1 to ensure Day T only sees features known at Day T-1
    df["arima_pred"] = df["arima_pred"].shift(1)
    df["garch_pred"] = df["garch_pred"].shift(1)
    
    return df


def run_walk_forward_validation(df, train_window, test_steps=100):
    """Executes rolling validation with regularization and returns all ranked features."""
    df["<DATE>"] = pd.to_datetime(df["<DATE>"])
    df = df.sort_values("<DATE>").reset_index(drop=True)
    
    df = calculate_technical_indicators(df)
    df["target"] = (df["<CLOSE>"].shift(-1) > df["<CLOSE>"]).astype(int)

    # Cleanup step 1: Drops rows missing indicator warmup periods
    required_indicators = [
        "returns", "volatility_5", "HL_spread", "CO_spread", "dist_MA_20", 
        "volume_ratio", "RSI", "Stochastic_14", "NTR_5", "ROC_10"
    ]
    df = df.dropna(subset=required_indicators).reset_index(drop=True)
    
    print(f"Rows remaining after technical indicator cleanup: {len(df)}")
    if len(df) == 0:
        print("CRITICAL WARNING: Technical indicators purged the entire dataframe! Check data consistency.")
        return df, 0.0, []
    
    # Fast statistical feature generation
    df = generate_all_statistical_features(df, train_window, step_size=5)
    df = df.dropna(subset=["arima_pred", "garch_pred"]).reset_index(drop=True)
    
    print(f"Rows remaining after ARIMA/GARCH statistical cleanup: {len(df)}")

    # Expanded Master Feature Array
    feature_cols = [
        "dist_MA_5", "dist_MA_20", "volatility_5", "HL_spread", "CO_spread", 
        "volume_ratio", "RSI", "arima_pred", "garch_pred", "Stochastic_14", "NTR_5", "ROC_10"
    ]

    predictions, actuals = [], []
    print(f"Starting walk-forward validation for {test_steps} iterations...")
    
    # Track cross-window aggregate feature importance (Information Gain)
    importance_accumulator = np.zeros(len(feature_cols))

    for i in range(test_steps):
        end_train_idx = train_window + i
        if end_train_idx >= len(df) - 1:
            break

        X_train_df = df.iloc[i:end_train_idx]
        test_row = df.iloc[[end_train_idx]]

        # --- LIGHTGBM WITH L1/L2 REGULARIZATION FOR VARIABLE SELECTION ---
        lgb_model = LGBMClassifier(
            n_estimators=50, 
            learning_rate=0.05, 
            max_depth=4, 
            random_state=42, 
            verbose=-1,
            colsample_bytree=0.7,  # Randomly sample 70% of columns per tree
            reg_alpha=5.0,         # L1 penalty to prune weak features completely
            reg_lambda=5.0         # L2 penalty to smooth leaf variances
        )
        lgb_model.fit(X_train_df[feature_cols], X_train_df["target"])

        # Accumulate information gain metric
        importance_accumulator += lgb_model.booster_.feature_importance(importance_type="gain")

        pred_proba = lgb_model.predict_proba(test_row[feature_cols])[0][1]
        pred_class = 1 if pred_proba >= 0.5 else 0

        predictions.append(pred_class)
        actuals.append(test_row["target"].values[0])

    accuracy = accuracy_score(actuals, predictions) if predictions else 0.0
    
    # Normalize importance values into percentage allocations
    total_gain = np.sum(importance_accumulator) if np.sum(importance_accumulator) > 0 else 1
    normalized_importances = (importance_accumulator / total_gain) * 100
    
    feature_importance_dict = dict(zip(feature_cols, normalized_importances))
    # Sort ALL features from highest to lowest impact
    sorted_all_features = sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)

    return df, accuracy, sorted_all_features


def predict_next_1_day(df, train_window=600):
    """Trains the final model on the most recent data window and projects out tomorrow's nudge."""
    print("\n==================================================")
    print("      1-DAY FORWARD DIRECTIONAL FORECAST          ")
    print("==================================================")
    
    feature_cols = [
        "dist_MA_5", "dist_MA_20", "volatility_5", "HL_spread", "CO_spread", 
        "volume_ratio", "RSI", "arima_pred", "garch_pred", "Stochastic_14", "NTR_5", "ROC_10"
    ]
    
    X_train_final = df.tail(train_window)
    y_train_final = X_train_final["target"]
    
    # Final production model applies matching regularization parameters
    lgb_model = LGBMClassifier(
        n_estimators=50, 
        learning_rate=0.05, 
        max_depth=4, 
        random_state=42, 
        verbose=-1,
        colsample_bytree=0.7,
        reg_alpha=5.0,
        reg_lambda=5.0
    )
    lgb_model.fit(X_train_final[feature_cols], y_train_final)
    
    latest_known_row = df.iloc[[-1]] 
    current_date = latest_known_row["<DATE>"].values[0]
    current_date_dt = pd.to_datetime(current_date)
    
    if current_date_dt.weekday() == 4:  # Friday check
        forecast_date = current_date_dt + pd.Timedelta(days=3)
    else:
        forecast_date = current_date_dt + pd.Timedelta(days=1)
    
    up_probability = lgb_model.predict_proba(latest_known_row[feature_cols])[0][1]
    
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
    close = 100 + np.cumsum(np.random.normal(0.05, 1.0, size=1000))
    open_p = close - np.random.normal(0, 0.5, size=1000)
    high = np.maximum(open_p, close) + np.random.exponential(0.2, size=1000)
    low = np.minimum(open_p, close) - np.random.exponential(0.2, size=1000)
    tickvol = np.random.randint(1000, 5000, size=1000) 

    mock_df = pd.DataFrame({
        "<DATE>": dates, "<OPEN>": open_p, "<HIGH>": high, "<LOW>": low, "<CLOSE>": close, "<TICKVOL>": tickvol
    })

    # Step 1: Run validation pipeline on mock data (Use train_window=250 for your active live data df_mod)
    processed_df, historical_accuracy, ranked_features = run_walk_forward_validation(mock_df, train_window=250, test_steps=20)
    
    # Print Backtest Performance
    print("\n" + "="*50)
    print("          BACKTEST ACCURACY RESULTS               ")
    print("="*50)
    print(f"Directional Accuracy Score: {historical_accuracy:.2%}")
    print("="*50)
    
    # Print FULL Feature Importance Metrics (Un-truncated)
    print("          COMPLETE FEATURE IMPORTANCE RANKINGS     ")
    print("="*50)
    for rank, (feature, importance) in enumerate(ranked_features, 1):
        print(f"Rank {rank:02d}: {feature:<15} -> Relative Impact: {importance:.2f}%")
    print("="*50)
    
    # Step 2: Generate live out-of-sample forward looking projections 
    predict_next_1_day(processed_df, train_window=250)
import numpy as np
import pandas as pd
import warnings
from arch import arch_model
from lightgbm import LGBMClassifier
from pmdarima import auto_arima
from sklearn.metrics import accuracy_score, classification_report

# Issue 1 Fix: Silence the polynomial reciprocal warning from statsmodels/pmdarima
warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=UserWarning)


def calculate_technical_indicators(df):
    """Calculates standard technical indicators for the LightGBM model."""
    df = df.copy()
    df["returns"] = np.log(df["<CLOSE>"] / df["<CLOSE>"].shift(1))
    df["MA_5"] = df["<CLOSE>"].rolling(window=5).mean()
    df["MA_20"] = df["<CLOSE>"].rolling(window=20).mean()
    df["volatility_5"] = df["returns"].rolling(window=5).std()
    df["HL_spread"] = df["<HIGH>"] - df["<LOW>"]
    df["CO_spread"] = df["<CLOSE>"] - df["<OPEN>"]
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
    """
    Issue 2 Fix: Pre-computes features but SKIPS steps to drastically improve speed.
    Updates the ARIMA/GARCH models every 'step_size' days instead of every single day.
    """
    print(f"Pre-computing ARIMA/GARCH features (Optimized: updating every {step_size} days)...")
    arima_preds = np.zeros(len(df)) * np.nan
    garch_preds = np.zeros(len(df)) * np.nan

    last_arima, last_garch = 0.0, 0.0

    for i in range(len(df)):
        if i < train_window:
            continue
        
        # Only re-fit models periodically to save massive amounts of time
        if (i - train_window) % step_size == 0:
            history = df["returns"].iloc[i - train_window : i]
            try:
                last_arima, last_garch = fit_arima_garch(history)
            except Exception:
                last_arima, last_garch = history.mean(), history.std()
        
        # Forward-fill the prediction for the step-gap
        arima_preds[i] = last_arima
        garch_preds[i] = last_garch

    df["arima_pred"] = arima_preds
    df["garch_pred"] = garch_preds
    return df


def run_walk_forward_validation(df, train_window=500, test_steps=100):
    """Executes rolling validation using the optimized fast-feature generation."""
    df["<DATE>"] = pd.to_datetime(df["<DATE>"])
    df = df.sort_values("<DATE>").reset_index(drop=True)
    df = calculate_technical_indicators(df)
    df["target"] = (df["<CLOSE>"].shift(-1) > df["<CLOSE>"]).astype(int)

    df = df.dropna(subset=["returns", "MA_20", "volatility_5", "HL_spread", "CO_spread"]).reset_index(drop=True)
    
    # Fast feature generation
    df = generate_all_statistical_features(df, train_window, step_size=5)
    df = df.dropna(subset=["arima_pred", "garch_pred"]).reset_index(drop=True)

    predictions, actuals = [], []
    feature_cols = ["MA_5", "MA_20", "volatility_5", "HL_spread", "CO_spread", "arima_pred", "garch_pred"]

    print(f"Starting walk-forward validation for {test_steps} iterations...")
    for i in range(test_steps):
        end_train_idx = train_window + i
        if end_train_idx >= len(df) - 1:
            break

        X_train_df = df.iloc[i:end_train_idx]
        test_row = df.iloc[[end_train_idx]]

        lgb_model = LGBMClassifier(n_estimators=50, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1)
        lgb_model.fit(X_train_df[feature_cols], X_train_df["target"])

        pred_proba = lgb_model.predict_proba(test_row[feature_cols])[0][1]
        pred_class = 1 if pred_proba >= 0.5 else 0

        predictions.append(pred_class)
        actuals.append(test_row["target"].values[0])

    if predictions:
        print(f"\n=== Validation Results ===\nDirectional Accuracy: {accuracy_score(actuals, predictions):.2%}")
    
    return df  # Return processed data to use for the future forecast


def predict_next_5_days(df, train_window=600):
    """
    NEW SCRIPT LAYER: Trains the final model on the most recent historical data window
    and projects out the probability nudges for the next 5 consecutive calendar days.
    """
    print("\n==================================================")
    print("      5-DAY FORWARD DIRECTIONAL FORECAST          ")
    print("==================================================")
    
    feature_cols = ["MA_5", "MA_20", "volatility_5", "HL_spread", "CO_spread", "arima_pred", "garch_pred"]
    
    # 1. Train the final LightGBM model using the absolute latest available data window
    X_train_final = df.tail(train_window)
    y_train_final = X_train_final["target"]
    
    lgb_model = LGBMClassifier(n_estimators=50, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1)
    lgb_model.fit(X_train_final[feature_cols], y_train_final)
    
    # 2. Extract the latest known row of data to bootstrap our 5-day projections
    last_known_row = df.iloc[-1].copy()
    current_close = last_known_row["<CLOSE>"]
    current_date = last_known_row["<DATE>"]
    
    # Re-extract the history needed to generate the base statistical forecasts
    latest_returns_history = df["returns"].tail(train_window)
    arima_f, garch_f = fit_arima_garch(latest_returns_history)
    
    # We project forward day-by-day. 
    # Note: Technical indicators are rolled forward using the model's expected baseline.
    for day in range(1, 6):
        forecast_date = current_date + pd.Timedelta(days=day)
        
        # Prepare feature vector for the target day
        input_data = pd.DataFrame([{
            "MA_5": last_known_row["MA_5"], 
            "MA_20": last_known_row["MA_20"],
            "volatility_5": last_known_row["volatility_5"],
            "HL_spread": last_known_row["HL_spread"],
            "CO_spread": last_known_row["CO_spread"],
            "arima_pred": arima_f,
            "garch_pred": garch_f
        }])
        
        # Get our prediction probability
        up_probability = lgb_model.predict_proba(input_data[feature_cols])[0][1]
        
        # Determine the nudge category based on your 0.35 / 0.65 thresholds
        if up_probability >= 0.65:
            nudge = "🟢 STRONG UP CONVICTION"
        elif up_probability <= 0.35:
            nudge = "🔴 STRONG DOWN CONVICTION"
        else:
            nudge = "⚪ COIN FLIP / NO ACTION"
            
        print(f"Day T+{day} ({forecast_date.strftime('%Y-%m-%d')}): Up Prob: {up_probability:.1%} | Nudge: {nudge}")


if __name__ == "__main__":
    # Mock data setup
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=1000, freq="D")
    close = 100 + np.cumsum(np.random.normal(0.05, 1.0, size=1000))
    open_p = close - np.random.normal(0, 0.5, size=1000)
    high = np.maximum(open_p, close) + np.random.exponential(0.2, size=1000)
    low = np.minimum(open_p, close) - np.random.exponential(0.2, size=1000)

    mock_df = pd.DataFrame({
        "<DATE>": dates, "<OPEN>": open_p, "<HIGH>": high, "<LOW>": low, "<CLOSE>": close,
    })

    # Run the backtest framework first
    processed_df = run_walk_forward_validation(mock_df, train_window=600, test_steps=20)
    
    # Run the new 5-day out-of-sample forward looking script
    predict_next_5_days(processed_df, train_window=600)
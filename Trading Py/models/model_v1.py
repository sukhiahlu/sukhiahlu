import numpy as np
import pandas as pd
from arch import arch_model
from lightgbm import LGBMClassifier
from pmdarima import auto_arima
from sklearn.metrics import accuracy_score, classification_report


def calculate_technical_indicators(df):
    """Calculates standard technical indicators for the LightGBM model."""
    df = df.copy()

    # Log Returns
    df["returns"] = np.log(df["<CLOSE>"] / df["<CLOSE>"].shift(1))

    # Moving Averages
    df["MA_5"] = df["<CLOSE>"].rolling(window=5).mean()
    df["MA_20"] = df["<CLOSE>"].rolling(window=20).mean()

    # Volatility (Standard Deviation)
    df["volatility_5"] = df["returns"].rolling(window=5).std()

    # Daily Range / Spread
    df["HL_spread"] = df["<HIGH>"] - df["<LOW>"]
    df["CO_spread"] = df["<CLOSE>"] - df["<OPEN>"]

    return df


def fit_arima_garch(returns_history):
    """Fits an automated ARIMA model and uses its residuals to fit a GARCH(1,1) model.

    Returns the 1-step ahead mean and volatility forecast.
    """
    # 1. ARIMA Step (auto-fit to find optimal p, d, q)
    # Using stationary log returns
    arima_model = auto_arima(
        returns_history.dropna(),
        start_p=0,
        start_q=0,
        max_p=3,
        max_q=3,
        d=0,
        seasonal=False,
        error_action="ignore",
        suppress_warnings=True,
    )
    arima_resid = arima_model.resid()
    arima_forecast = arima_model.predict(n_periods=1)[0]

    # 2. GARCH Step (on ARIMA residuals)
    # Scale returns slightly if values are too small to prevent optimizer convergence issues
    garch = arch_model(
        arima_resid, vol="Garch", p=1, q=1, dist="normal", show_warning=False
    )
    garch_fitted = garch.fit(disp="off")

    # 1-step ahead conditional volatility forecast
    garch_forecast_res = garch_fitted.forecast(horizon=1)
    garch_vol_forecast = np.sqrt(
        garch_forecast_res.variance.iloc[-1].values[0]
    )

    return arima_forecast, garch_vol_forecast


def run_walk_forward_validation(df, train_window=500, test_steps=100):
    """Executes the rolling walk-forward framework.

    Parameters:
    - df: The input DataFrame with columns: <DATE>, <OPEN>, <HIGH>, <LOW>, <CLOSE>
    - train_window: Number of days used to train the initial models (e.g.,
    500)
    - test_steps: Number of times (X) to roll forward and re-evaluate
    """
    # Ensure correct sorting and datetime formatting
    df["<DATE>"] = pd.to_datetime(df["<DATE>"])
    df = df.sort_values("<DATE>").reset_index(drop=True)

    # Feature Engineering
    df = calculate_technical_indicators(df)

    # Target variable: 1 if next day's close is higher than today's close, else 0
    df["target"] = (df["<CLOSE>"].shift(-1) > df["<CLOSE>"]).astype(int)

    # Initialize empty placeholders for statistical features
    df["arima_pred"] = np.nan
    df["garch_pred"] = np.nan

    # Drop early NaNs caused by indicators
    df = df.dropna(
        subset=["returns", "MA_20", "volatility_5", "HL_spread", "CO_spread"]
    ).reset_index(drop=True)

    predictions = []
    actuals = []

    print(f"Starting walk-forward validation for {test_steps} iterations...")

    # Rolling Backtest Loop
    for i in range(test_steps):
        # Define current dynamic slicing indices
        start_idx = i
        end_train_idx = start_idx + train_window

        # Check if we have hit the edge of the dataset
        if end_train_idx >= len(df) - 1:
            print("Reached the end of the available dataset.")
            break

        # Slice historical training data
        train_df = df.iloc[start_idx:end_train_idx].copy()
        test_row = df.iloc[[end_train_idx]].copy()

        # --- STEP 1: ARIMA + GARCH ---
        try:
            # We pass the log returns history to generate the next day's forecast
            arima_f, garch_f = fit_arima_garch(train_df["returns"])
            df.loc[end_train_idx, "arima_pred"] = arima_f
            df.loc[end_train_idx, "garch_pred"] = garch_f
        except Exception as e:
            # Fallback to 0 / last known rolling mean if optimizer occasionally fails to converge
            df.loc[end_train_idx, "arima_pred"] = train_df["returns"].mean()
            df.loc[end_train_idx, "garch_pred"] = train_df[
                "volatility_5"
            ].iloc[-1]

        # Update test row slice with the freshly generated features
        test_row = df.iloc[[end_train_idx]]

        # --- STEP 2: LightGBM ---
        # Re-slice training matrix to include only populated rows up to this point
        # (We skip the rows before train_window that lack ARIMA/GARCH historical signals)
        valid_train = df.iloc[train_window:end_train_idx].dropna(
            subset=["arima_pred", "garch_pred"]
        )

        if len(valid_train) < 50:  # Warmup buffer check
            continue

        feature_cols = [
            "MA_5",
            "MA_20",
            "volatility_5",
            "HL_spread",
            "CO_spread",
            "arima_pred",
            "garch_pred",
        ]

        X_train = valid_train[feature_cols]
        y_train = valid_train["target"]

        X_test = test_row[feature_cols]
        y_test = test_row["target"].values[0]

        # Fit LightGBM Classifier
        lgb_model = LGBMClassifier(
            n_estimators=50,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            verbose=-1,
        )
        lgb_model.fit(X_train, y_train)

        # Predict
        pred = lgb_model.predict(X_test)[0]

        predictions.append(pred)
        actuals.append(y_test)

    # Evaluation summary
    if predictions:
        print("\n=== Validation Results ===")
        print(f"Total Iterations Evaluated: {len(predictions)}")
        print(f"Directional Accuracy: {accuracy_score(actuals, predictions):.2%}")
        print("\nClassification Report:")
        print(classification_report(actuals, predictions))
    else:
        print("Not enough iterations completed.")


if __name__ == "__main__":
    # --- Example Usage ---
    # Replace this mock block with your real data ingestion:
    # df = pd.read_csv("your_price_data.csv")

    print("Generating synthetic trading data for demonstration...")
    np.random.seed(42)
    dates = pd.date_range(start="2022-01-01", periods=1000, freq="D")
    close = 100 + np.cumsum(np.random.normal(0.1, 1.5, size=1000))
    open_p = close - np.random.normal(0, 1, size=1000)
    high = np.maximum(open_p, close) + np.random.exponential(0.5, size=1000)
    low = np.minimum(open_p, close) - np.random.exponential(0.5, size=1000)

    mock_df = pd.DataFrame(
        {
            "<DATE>": dates,
            "<OPEN>": open_p,
            "<HIGH>": high,
            "<LOW>": low,
            "<CLOSE>": close,
        }
    )

    # Run validation: Trains on 600 days, rolls forward 100 times to validate
    run_walk_forward_validation(mock_df, train_window=600, test_steps=100)
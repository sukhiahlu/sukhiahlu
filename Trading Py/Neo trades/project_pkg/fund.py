import numpy as np
import pandas as pd
from fredapi import Fred
import sys
import requests

# =====================================================================
# ENGINE INITIALIZATION & CONFIGURATION
# =====================================================================
FRED_API_KEY = "1f3befec93b6c68383d27f3b58cca038"
if not FRED_API_KEY or FRED_API_KEY == "YOUR_FREE_FRED_API_KEY_HERE":
    print("\n[!] ERROR: Valid FRED API Key required.")
    sys.exit()

fred = Fred(api_key=FRED_API_KEY)

# World Bank Country ISO-3 Codes Mapping
CCY_TO_ISO3 = {
    "USD": "USA", "EUR": "EMU", "GBP": "GBR", "JPY": "JPN",
    "AUD": "AUS", "CAD": "CAN", "CHF": "CHE", "CNY": "CHN", "INR": "IND"
}

# World Bank API Indicator Codes
WB_INDICATORS = {
    "GDP_Growth": "NY.GDP.MKTP.KD.ZG",       # GDP growth (annual %)
    "Current_Account": "BN.CAB.XOKA.GD.ZS",  # Current account balance (% of GDP)
    "Inflation": "FP.CPI.TOTL.ZG"            # Inflation, consumer prices (annual %)
}

# Note: Current Account uses dummy tickers to force the pipeline to use World Bank's uniform % of GDP format.
series_map = {
    "USD": {"Nominal_2Y": "GS2", "Inflation": "CPIAUCSL", "GDP_Growth": "A191RL1Q225SBEA", "Current_Account": "INVALID_FRED_CA_DUMMY"},
    "EUR": {"Nominal_2Y": "IR3TIB01EZA156N", "Inflation": "CP0000EZ19M086NEST", "GDP_Growth": "INVALID_FRED_EUR_GDP_DUMMY", "Current_Account": "INVALID_FRED_CA_DUMMY"},
    "GBP": {"Nominal_2Y": "IRLTLT01GBM156N", "Inflation": "GBRCPIALLMINMEI", "GDP_Growth": "UKNGDPRPCHGQS", "Current_Account": "INVALID_FRED_CA_DUMMY"},
    "JPY": {"Nominal_2Y": "IRLTLT01JPM156N", "Inflation": "JPNCPIALLMINM", "GDP_Growth": "JPNNGDPPCHQS", "Current_Account": "INVALID_FRED_CA_DUMMY"},
    "AUD": {"Nominal_2Y": "IRLTLT01AUM156N", "Inflation": "AUSCPIALLQINM", "GDP_Growth": "NGDPRPCHAUSQS", "Current_Account": "INVALID_FRED_CA_DUMMY"},
    "CAD": {"Nominal_2Y": "IR3TIB01CAM156N", "Inflation": "CANCPIALLMINMEI", "GDP_Growth": "CANGDPRPCHGQS", "Current_Account": "INVALID_FRED_CA_DUMMY"},
    "CHF": {"Nominal_2Y": "IRLTLT01CHM156N", "Inflation": "CHFCPIALLMINMEI", "GDP_Growth": "CHECOSGR01GTQ", "Current_Account": "INVALID_FRED_CA_DUMMY"},
    "CNY": {"Nominal_2Y": "INTDSRCNM193N", "Inflation": "CHNCPIALLMINMEI", "GDP_Growth": "CHINGDPRPCHGQS", "Current_Account": "INVALID_FRED_CA_DUMMY"},
    "INR": {"Nominal_2Y": "IRSTCI01INA156N", "Inflation": "INRCPIALLMINMEI", "GDP_Growth": "INDNGDPRPCHGQS", "Current_Account": "INVALID_FRED_CA_DUMMY"}
}

fallback_defaults = {
    "USD": {"Nominal_2Y": 4.15, "Inflation": 2.6, "GDP_Growth": 2.1, "Current_Account": -3.0},
    "EUR": {"Nominal_2Y": 3.00, "Inflation": 2.2, "GDP_Growth": 1.0, "Current_Account": 2.5},
    "GBP": {"Nominal_2Y": 3.85, "Inflation": 2.4, "GDP_Growth": 0.8, "Current_Account": -2.0},
    "JPY": {"Nominal_2Y": 0.35, "Inflation": 2.1, "GDP_Growth": 0.5, "Current_Account": 3.5},
    "AUD": {"Nominal_2Y": 3.90, "Inflation": 2.8, "GDP_Growth": 1.8, "Current_Account": 1.2},
    "CAD": {"Nominal_2Y": 3.65, "Inflation": 2.5, "GDP_Growth": 1.2, "Current_Account": -0.5},
    "CHF": {"Nominal_2Y": 1.10, "Inflation": 1.2, "GDP_Growth": 1.1, "Current_Account": 7.0},
    "CNY": {"Nominal_2Y": 1.95, "Inflation": 0.5, "GDP_Growth": 4.8, "Current_Account": 1.5},
    "INR": {"Nominal_2Y": 6.75, "Inflation": 4.8, "GDP_Growth": 6.5, "Current_Account": -1.2}
}

def fetch_world_bank_value(iso3: str, metric: str) -> float:
    """Helper to pull the absolute most recent data point from the World Bank API."""
    if metric not in WB_INDICATORS:
        return None
    indicator = WB_INDICATORS[metric]
    url = f"http://api.worldbank.org/v2/country/{iso3}/indicator/{indicator}?format=json&per_page=5"
    try:
        response = requests.get(url, timeout=10).json()
        if len(response) > 1 and isinstance(response[1], list):
            for entry in response[1]:
                if entry.get("value") is not None:
                    return float(entry["value"])
    except Exception:
        pass
    return None

def generate_daily_fred_scorecard():
    output_vals = {}
    audit_trail = {}
    
    print("[+] Executing 3-Step Pipeline (With Strict 2-Decimal Precision Rounding)...")
    
    for ccy, tickers in series_map.items():
        ccy_vals = {}
        ccy_audit = {}
        iso3 = CCY_TO_ISO3[ccy]
        
        metrics_list = ["Nominal_2Y", "Inflation", "GDP_Growth", "Current_Account"]
        
        for metric in metrics_list:
            # -----------------------------------------------------------------
            # STEP 1: FRED INGESTION
            # -----------------------------------------------------------------
            try:
                if metric == "Inflation":
                    series = fred.get_series(tickers["Inflation"], start_date="2020-01-01").dropna()
                    if len(series) >= 13:
                        yoy_series = ((series / series.shift(12)) - 1.0) * 100.0
                        val = yoy_series.dropna().iloc[-1]
                        status = "FRED"
                    else:
                        status = "FALLBACK"
                else:
                    series = fred.get_series(tickers[metric], start_date="2021-01-01").dropna()
                    if not series.empty:
                        val = series.iloc[-1]
                        status = "FRED"
                    else:
                        status = "FALLBACK"
            except Exception:
                status = "FALLBACK"
                
            # -----------------------------------------------------------------
            # STEP 2: APPLY DEFAULT STATIC FALLBACK IF FRED IS EMPTY
            # -----------------------------------------------------------------
            if status == "FALLBACK":
                val = fallback_defaults[ccy][metric]
                
            # -----------------------------------------------------------------
            # STEP 3: WORLD BANK SWEEP (Only override if field hit FALLBACK)
            # -----------------------------------------------------------------
            if status == "FALLBACK" and metric in WB_INDICATORS:
                wb_val = fetch_world_bank_value(iso3, metric)
                if wb_val is not None:
                    val = wb_val
                    status = "WORLD_BANK"
            
            ccy_vals[metric] = round(val, 2)
            ccy_audit[metric] = status
            
        ccy_vals["Real_2Y_Yield"] = round(ccy_vals["Nominal_2Y"] - ccy_vals["Inflation"], 2)
        
        output_vals[ccy] = ccy_vals
        audit_trail[ccy] = ccy_audit
        
    df_vals = pd.DataFrame.from_dict(output_vals, orient="index")
    df_audit = pd.DataFrame.from_dict(audit_trail, orient="index")
    
    # -----------------------------------------------------------------
    # CROSS-SECTIONAL SCORES ENGINE
    # -----------------------------------------------------------------
    output_scores = {}
    for ccy in series_map.keys():
        ccy_scores = {}
        ccy_scores["Nominal_2Y_A"] = df_vals["Nominal_2Y"].rank(pct=True)[ccy]
        ccy_scores["GDP_Growth_A"] = df_vals["GDP_Growth"].rank(pct=True)[ccy]
        ccy_scores["Current_Account_A"] = df_vals["Current_Account"].rank(pct=True)[ccy]
        ccy_scores["Inflation_A"] = (1.0 - df_vals["Inflation"].rank(pct=True))[ccy]
        
        output_scores[ccy] = ccy_scores
        
    df_scores = pd.DataFrame.from_dict(output_scores, orient="index")
    
    weights = {
        "Nominal_2Y_A": 0.30, 
        "Inflation_A": 0.20,
        "GDP_Growth_A": 0.30, 
        "Current_Account_A": 0.20
    }
    
    df_scores["Final_Historical_Score"] = 0.0
    for feature, weight in weights.items():
        df_scores["Final_Historical_Score"] += df_scores[feature] * weight
        
    # Round final scores to exactly 2 decimals before returning
    df_scores["Final_Historical_Score"] = df_scores["Final_Historical_Score"].round(2)
        
    return df_vals, df_scores.sort_values(by="Final_Historical_Score", ascending=False), df_audit

if __name__ == "__main__":
    vals, scores, audit = generate_daily_fred_scorecard()
    print("\n" + "="*50 + " AUDIT TRAIL DATA FLOW " + "="*50)
    print(audit)
    print("\n" + "="*50 + " MACRO SCORING MATRIX " + "="*50)
    print(scores[["Final_Historical_Score"]])
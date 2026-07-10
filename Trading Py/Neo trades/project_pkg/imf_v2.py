import requests
import pandas as pd
import datetime
import numpy as np

# =====================================================================
# GLOBAL CONFIGURATION & STATIC MAPS
# =====================================================================
# Map your 9 currencies to IMF ISO-3 country codes
# EUR is proxied by Germany (DEU) as the core structural/macro anchor
CCY_TO_ISO3 = {
    "USD": "USA", "EUR": "DEU", "GBP": "GBR", "JPY": "JPN",
    "AUD": "AUS", "CAD": "CAN", "CHF": "CHE", "CNY": "CHN", "INR": "IND"
}

# Centralized weight allocation - easy to tweak during market regime shifts
MODEL_WEIGHTS = {
    "Rank_Growth_Accel": 0.20,
    "Rank_Current_Account": 0.20,
    "Rank_Real_Rate": 0.20,
    "Rank_Inflation": 0.20,
    "Rank_Unemployment": 0.20
}

def generate_imf_scorecard():
    current_year = datetime.datetime.now().year
    next_year = current_year + 1
    
    current_year_str = str(current_year)
    next_year_str = str(next_year)
    
    # IMF DataMapper Systemic Indicators
    # Using 'FI_SI_3M' for short-term policy interest rate proxies from the live API
    indicators = {
        "GDP_Growth_Current": (f"NGDP_RPCH?periods={current_year}", current_year_str),
        "GDP_Growth_Forecast": (f"NGDP_RPCH?periods={next_year}", next_year_str),
        "Inflation": (f"PCPIPCH?periods={current_year}", current_year_str),
        "Unemployment": (f"LUR?periods={current_year}", current_year_str),
        "Current_Account_Pct_GDP": (f"BCA_NGDPD?periods={current_year}", current_year_str),
        "Short_Rates": (f"FI_SI_3M?periods={current_year}", current_year_str)
    }
    
    base_url = "https://www.imf.org/external/datamapper/api/v1"
    raw_extracted = {ccy: {} for ccy in CCY_TO_ISO3.keys()}
    
    print("[+] Gathering institutional macro data from IMF DataMapper REST API...")
    
    for metric_name, (endpoint, target_year) in indicators.items():
        try:
            indicator_code = endpoint.split("?")[0]
            url = f"{base_url}/{endpoint}"
            response = requests.get(url, timeout=15).json()
            values_data = response.get("values", {}).get(indicator_code, {})
            
            for ccy, iso3 in CCY_TO_ISO3.items():
                ccy_data = values_data.get(iso3, {})
                
                # Explicitly target the correct year key instead of positional indexing
                val = ccy_data.get(target_year, np.nan)
                raw_extracted[ccy][metric_name] = float(val) if val is not None else np.nan
                
        except Exception as e:
            print(f"[-] Request failed or timed out for indicator [{metric_name}]: {e}")
            for ccy in CCY_TO_ISO3.keys():
                raw_extracted[ccy][metric_name] = np.nan

    df = pd.DataFrame.from_dict(raw_extracted, orient="index")
    
    # Defensive Data Cleaning: Fill missing rows with the cross-sectional median
    if df.isna().sum().sum() > 0:
        print("[!] Warning: Gaps identified in raw feeds. Applying cross-sectional median fill...")
        df = df.fillna(df.median())
    
    # =====================================================================
    # FEATURE ENGINEERING & CROSS-SECTIONAL RANKING
    # =====================================================================
    # 1. Growth Momentum (Acceleration)
    df["Growth_Acceleration"] = df["GDP_Growth_Forecast"] - df["GDP_Growth_Current"]
    
    # 2. Yield Spread Metric (Real Policy Yield Estimate)
    df["Estimated_Real_Rate"] = df["Short_Rates"] - df["Inflation"]
    
    # 3. Cross-Sectional Percentile Rankings (Relative to peer group)
    df["Rank_Growth_Accel"] = df["Growth_Acceleration"].rank(pct=True)
    df["Rank_Current_Account"] = df["Current_Account_Pct_GDP"].rank(pct=True)
    df["Rank_Real_Rate"] = df["Estimated_Real_Rate"].rank(pct=True)
    df["Rank_Inflation"] = (1.0 - df["Inflation"].rank(pct=True))
    df["Rank_Unemployment"] = (1.0 - df["Unemployment"].rank(pct=True))
    
    # Dynamic weight aggregation using configuration dictionary
    df["Structural_Score"] = sum(df[rank_col] * weight for rank_col, weight in MODEL_WEIGHTS.items())
    
    return df.sort_values(by="Structural_Score", ascending=False)

if __name__ == "__main__":
    scorecard = generate_imf_scorecard()
    print("\n" + "=" * 110)
    print("                      COMPREHENSIVE IMF LIVE GLOBAL HEALTH LEADERBOARD                              ")
    print("=" * 110)
    columns_to_show = ["GDP_Growth_Current", "Inflation", "Short_Rates", "Estimated_Real_Rate", "Current_Account_Pct_GDP", "Structural_Score"]
    print(scorecard[columns_to_show].round(2).to_string())
    print("=" * 110)
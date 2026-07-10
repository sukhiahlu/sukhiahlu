import datetime
import numpy as np
import pandas as pd
import requests

# =====================================================================
# GLOBAL CONFIGURATION & METRIC WEIGHTS
# =====================================================================
# Map target currencies to IMF ISO-3 country codes.
CCY_TO_ISO3 = {
    "USD": ["USA"],
    "EUR": ["DEU", "FRA", "ITA", "ESP"],  # Multi-nation Eurozone composite
    "GBP": ["GBR"],
    "JPY": ["JPN"],
    "AUD": ["AUS"],
    "CAD": ["CAN"],
    "CHF": ["CHE"],
    "CNY": ["CHN"],
    "INR": ["IND"],
}

# Explicitly allocated weights summing to 1.00 using momentum profiles
MODEL_WEIGHTS = {
    "Rank_Growth_Accel": 0.3,  # Forecast growth minus current year growth
    "Rank_Inflation_Cooling": 0.2,  # Rate at which inflation is projected to drop
    "Rank_CA_Momentum": 0.25,  # Improvement or degradation of current account % GDP
    #"Rank_Forward_Real_Rate": 0.25,  # Short rates vs. Forward-looking expected inflation
    "Rank_Unemployment_Delta": 0.25,  # Structural labor market tightness (Directional)
}

# Hardcoded approximate multi-year average nominal GDP shares to serve as a 
# foolproof fallback mechanism for the Eurozone weight assembly.
EUR_GDP_FALLBACKS = {"DEU": 4.5, "FRA": 3.1, "ITA": 2.2, "ESP": 1.6}


def generate_forward_macro_scorecard():
    # Dynamically extract current and forward periods
    current_year = datetime.datetime.now().year
    next_year = current_year + 1

    current_year_str = str(current_year)
    next_year_str = str(next_year)

    print(
        f"[*] Initializing Forward-Looking Velocity Scorecard Engine ({current_year_str} vs {next_year_str})..."
    )

    # Flatten out distinct countries required for query payload
    all_countries = list(
        set([country for list_c in CCY_TO_ISO3.values() for country in list_c])
    )
    country_query_str = ",".join(all_countries)

    # Dictionary map using IMF Datamapper v1/v2 indicator metrics
    # Tapping into native multi-year forecast vectors for relative change analysis
    indicators = {
        "Nominal_GDP_USD": (f"NGDPD?periods={current_year}", current_year_str),
        "GDP_Growth_Current": (
            f"NGDP_RPCH?periods={current_year}",
            current_year_str,
        ),
        "GDP_Growth_Forecast": (
            f"NGDP_RPCH?periods={next_year}",
            next_year_str,
        ),
        "Inflation_Current": (f"PCPIPCH?periods={current_year}", current_year_str),
        "Inflation_Forecast": (f"PCPIPCH?periods={next_year}", next_year_str),
        "Current_Account_Current": (
            f"BCA_NGDPD?periods={current_year}",
            current_year_str,
        ),
        "Current_Account_Forecast": (
            f"BCA_NGDPD?periods={next_year}",
            next_year_str,
        ),
        "Unemployment_Current": (f"LUR?periods={current_year}", current_year_str),
        "Unemployment_Forecast": (f"LUR?periods={next_year}", next_year_str),
        "Short_Rates": (
            f"FI_SI_3M?periods={current_year}",
            current_year_str,
        ),  # Coincident anchor rate
    }

    # Initialize localized country database frame
    df_countries = pd.DataFrame(index=all_countries)

    # =====================================================================
    # DATA ACQUISITION LAYER
    # =====================================================================
    for col_name, (endpoint_path, year_key) in indicators.items():
        # Safeguard parsing query metrics cleanly
        indicator_code = endpoint_path.split("?")[0]
        url = f"https://www.imf.org/external/datamapper/api/v1/{indicator_code}/{country_query_str}?periods={year_key}"

        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()

            # Drill down into nested json structure from IMF payload
            values = data.get("values", {}).get(indicator_code, {})

            # Clean and map across index
            extracted_series = {}
            for country in all_countries:
                val = values.get(country, {}).get(year_key, np.nan)
                extracted_series[country] = (
                    float(val) if val is not None and str(val).strip() != "" else np.nan
                )

            df_countries[col_name] = pd.Series(extracted_series)

        except Exception as e:
            print(f"[!] Warning: Failed parsing metric {col_name} via API: {e}")
            df_countries[col_name] = np.nan

    # =====================================================================
    # DEFENSIVE IMPUTATION PROTOCOLS
    # =====================================================================
    # If dynamic nominal GDP data vector has gaps, employ safe static fallback configurations
    if df_countries["Nominal_GDP_USD"].isna().any():
        for country, fallback_gdp in EUR_GDP_FALLBACKS.items():
            if (
                country in df_countries.index
                and pd.isna(df_countries.loc[country, "Nominal_GDP_USD"])
            ):
                df_countries.loc[country, "Nominal_GDP_USD"] = fallback_gdp

    # Clean up non-critical missing parameters via cross-sectional basket medians
    df_countries = df_countries.fillna(df_countries.median())

    # =====================================================================
    # COMPOSITE BLENDING LAYER (DYNAMIC WEIGHTING ENGINE)
    # =====================================================================
    df_ccy = pd.DataFrame(index=CCY_TO_ISO3.keys())
    blend_metrics = [c for c in df_countries.columns if c != "Nominal_GDP_USD"]

    for ccy, countries in CCY_TO_ISO3.items():
        sub_df = df_countries.loc[countries]

        if len(countries) == 1:
            # Single nation currency (USD, GBP, JPY, etc.)
            for metric in blend_metrics:
                df_ccy.loc[ccy, metric] = sub_df[metric].values[0]
        else:
            # Multi-nation asset pool (EUR Zone Blending Optimization)
            total_gdp = sub_df["Nominal_GDP_USD"].sum()
            weights = (
                sub_df["Nominal_GDP_USD"] / total_gdp
                if total_gdp > 0
                else pd.Series(1 / len(countries), index=countries)
            )

            for metric in blend_metrics:
                # Dynamic dot-product calculation based on active relative GDP contributions
                df_ccy.loc[ccy, metric] = np.dot(sub_df[metric], weights)

    # =====================================================================
    # QUANT FEATURE ENGINEERING (VELOCITY & SPREAD DISTILLATION)
    # =====================================================================
    # 1. Growth Velocity Vector
    df_ccy["Growth_Acceleration"] = (
        df_ccy["GDP_Growth_Forecast"] - df_ccy["GDP_Growth_Current"]
    )

    # 2. Forward-Looking Real Yield Spread (Anchor short yield against future expected inflation run-rate)
    df_ccy["Forward_Real_Rate"] = (
        df_ccy["Short_Rates"] - df_ccy["Inflation_Forecast"]
    )

    # 3. Inflation Trajectory Vector (Positive values signify active compression toward target bounds)
    df_ccy["Inflation_Cooling"] = (
        df_ccy["Inflation_Current"] - df_ccy["Inflation_Forecast"]
    )

    # 4. Current Account Velocity Balance Dynamics
    df_ccy["CA_Momentum"] = (
        df_ccy["Current_Account_Forecast"] - df_ccy["Current_Account_Current"]
    )

    # 5. Labor Force Tightness Dynamics (Directional degradation or optimization metric)
    df_ccy["Unemployment_Delta"] = (
        df_ccy["Unemployment_Current"] - df_ccy["Unemployment_Forecast"]
    )

    # =====================================================================
    # RELATIVE RANKING SCORING MATRIX
    # =====================================================================
    # Standardize and map cross-sections inside relative spectrum bounds [0.0 - 1.0]
    df_ccy["Rank_Growth_Accel"] = df_ccy["Growth_Acceleration"].rank(pct=True)
    df_ccy["Rank_Forward_Real_Rate"] = df_ccy["Forward_Real_Rate"].rank(pct=True)
    df_ccy["Rank_Inflation_Cooling"] = df_ccy["Inflation_Cooling"].rank(pct=True)
    df_ccy["Rank_CA_Momentum"] = df_ccy["CA_Momentum"].rank(pct=True)
    df_ccy["Rank_Unemployment_Delta"] = df_ccy["Unemployment_Delta"].rank(pct=True)

    # Matrix multiplication assembly using validated explicit weight parameters
    df_ccy["Structural_Score"] = sum(
        df_ccy[rank_col] * weight for rank_col, weight in MODEL_WEIGHTS.items()
    )

    # Present structured payload outputs sorted by structural optimization rank
    return df_ccy.sort_values(by="Structural_Score", ascending=False)


if __name__ == "__main__":
    scorecard = generate_forward_macro_scorecard()

    print("\n" + "=" * 120)
    print(
        f"               Institutional FX Structural Momentum Scorecard Matrix ({datetime.datetime.now().year})         "
    )
    print("=" * 120)

    columns_to_show = [
        "Growth_Acceleration",
        "Forward_Real_Rate",
        "Inflation_Cooling",
        "CA_Momentum",
        "Structural_Score",
    ]

    print(scorecard[columns_to_show].to_string(formatters={
        "Growth_Acceleration": "{:,.2f}%".format,
        "Forward_Real_Rate": "{:,.2f}%".format,
        "Inflation_Cooling": "{:,.2f}%".format,
        "CA_Momentum": "{:,.2f}%".format,
        "Structural_Score": "{:,.3f}".format,
    }))
    print("=" * 120)
import pandas as pd
import numpy as np

def calculate_macro_scores(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the cross-sectional percentile scores across all columns 
    and returns a single Series representing the final macro score for each country.
    """
    df_scores = pd.DataFrame(index=df.index)
    
    # --- Actionable Feedback: Missing Value Backstop ---
    # If a column is missing values, fill them with the value from the "Euro Area" row
    if "Euro Area" in df.index:
        euro_area_values = df.loc["Euro Area"]
        df = df.fillna(euro_area_values)
    else:
        # Guardrail: If "Euro Area" isn't in your copied data, fallback to filling with 0
        df = df.fillna(0)
    
    # 1. Yield / Carry (Higher is better)
    df_scores["Interest_Rate_Score"] = df["Interest_Rate"].rank(pct=True)
    
    # 2. Economic Momentum (Higher is better)
    df_scores["GDP_Growth_Score"] = df["GDP_Growth"].rank(pct=True)
    
    # 3. Structural Friction & Fragility (Inverted: Lower raw value = Higher score)
    df_scores["Inflation_Score"] = 1.0 - df["Inflation_Rate"].rank(pct=True)
    df_scores["Unemployment_Score"] = 1.0 - df["Unemployment_rate"].rank(pct=True)
    df_scores["Gov_Budget_Score"] = df["Gov_Budget"].rank(pct=True) # Higher surplus/lower deficit is better
    df_scores["Gov_Debt_Score"] = 1.0 - df["Gov_Debt_to_GDP"].rank(pct=True)
    df_scores["Current_Account_Score"] = df["Current_Account_to_GDP"].rank(pct=True)

    # 4. Strategic Weights (Summing up to 1.0)
    weights = {
        "Interest_Rate_Score":   0.20,
        "GDP_Growth_Score":      0.20,
        "Inflation_Score":       0.15,
        "Unemployment_Score":    0.10,
        "Gov_Budget_Score":      0.10,
        "Gov_Debt_Score":        0.10,
        "Current_Account_Score": 0.15
    }
    
    # 5. Compute Weighted Combined Score
    final_score = pd.Series(0.0, index=df.index)
    for factor, weight in weights.items():
        final_score += df_scores[factor] * weight
        
    return final_score.round(2)
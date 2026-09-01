# src/predictor.py
import numpy as np
import pandas as pd

def calculate_addiction_risk(metrics):
    """
    Ingests the extracted smartphone metrics and applies a scoring logic
    inspired by tree-based ensemble boundary splits to predict behavioral risk.
    """
    # 1. Gracefully extract core continuous features with safe fallbacks
    total_time = float(metrics.get("total_screen_time_minutes", 0) or 0)
    social_time = float(metrics.get("social_media_minutes", 0) or 0)
    entertainment_time = float(metrics.get("entertainment_minutes", 0) or 0)
    productivity_time = float(metrics.get("productivity_minutes", 0) or 0)
    
    # Calculate lingering 'Other' bucket categories dynamically
    accounted_time = social_time + entertainment_time + productivity_time
    other_time = max(0.0, total_time - accounted_time)
    
    unlocks = metrics.get("total_unlocks")
    unlocks = float(unlocks) if unlocks is not None else 50.0 

    # 2. Advanced Feature Engineering
    total_screen_hours = total_time / 60.0
    dopamine_load = social_time + entertainment_time
    utility_offset = productivity_time + 1.0 
    load_ratio = dopamine_load / utility_offset
    unlock_intensity = unlocks / (total_screen_hours + 0.1)

    # 3. Simulated Ensemble Leaf Scoring
    base_score = 0.35  
    if total_time > 360:     
        base_score += 0.25
    elif total_time > 180:   
        base_score += 0.10
        
    if load_ratio > 3.0:
        base_score += 0.20
    elif load_ratio > 1.5:
        base_score += 0.08
        
    if unlock_intensity > 15: 
        base_score += 0.15
    elif unlock_intensity > 8:
        base_score += 0.05
        
    if productivity_time > 90 and total_time < 300:
        base_score -= 0.08

    addiction_probability = float(np.clip(base_score, 0.0, 1.0))

    if addiction_probability >= 0.75:
        risk_category = "Critical / High Additive Load"
        color = "error"
        advice = "Severe screen-dependence detected. Consider setting strict app-timer limits and blackouting phone usage 1 hour before bed."
    elif addiction_probability >= 0.45:
        risk_category = "Moderate / Elevated Habit Formation"
        color = "warning"
        advice = "Screen patterns are trending high. Try implementing 'Digital Detours' by moving addictive apps off your primary home screen grid."
    else:
        risk_category = "Low / Regulated Mindful Consumption"
        color = "success"
        advice = "Excellent baseline regulation! Your smartphone habits show stable contextual boundary splits."

    # 4. Generate structured dataset for Plotly charting engines
    chart_df = pd.DataFrame({
        "Category": ["Social Networking", "Entertainment / Media", "Productivity", "Other Utility"],
        "Minutes Spent": [social_time, entertainment_time, productivity_time, other_time]
    })

    # 5. Compile a clean text summary report artifact
    report_text = f"""==================================================
FOCUS_SENSE AI: BEHAVIORAL SCREEN-TIME AUDIT REPORT
==================================================
Diagnostic Status: {risk_category}
Addiction Probability Rating: {int(addiction_probability * 100)}%

METRICS SUMMARY SUMMARY PROFILE:
-------------------------------
- Total Logged Screen Duration: {total_time} mins ({round(total_screen_hours, 1)} hrs)
- Social Networking Time      : {social_time} mins
- Entertainment / Media Time  : {entertainment_time} mins
- Core Focused Productivity   : {productivity_time} mins
- Estimated Device Unlocks    : {unlocks} checks

ENGINEERED INSIGHT RATIOS:
-------------------------
- Dopamine-to-Utility Ratio   : {round(load_ratio, 2)}x
- Hourly Checking Intensity   : {round(unlock_intensity, 1)} unlocks/hr

RECOMMENDED ACTION PLAN:
-----------------------
{advice}

--------------------------------------------------
Generated via FocusSense AI Predictive Engine Cloud Cluster.
==================================================
"""

    return {
        "risk_probability": round(addiction_probability, 4),
        "risk_category": risk_category,
        "color_theme": color,
        "actionable_advice": advice,
        "engineered_features": {
            "dopamine_utility_ratio": round(load_ratio, 2),
            "unlocks_per_hour": round(unlock_intensity, 1),
            "total_active_hours": round(total_screen_hours, 1)
        },
        "chart_data": chart_df,
        "downloadable_report": report_text
    }

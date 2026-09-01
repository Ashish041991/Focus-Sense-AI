# src/predictor.py
import numpy as np

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
    
    # Handle unlocks safely (convert to float if present, else fallback to average baseline)
    unlocks = metrics.get("total_unlocks")
    unlocks = float(unlocks) if unlocks is not None else 50.0 

    # 2. Advanced Feature Engineering (Replicating Kaggle tabular interaction terms)
    # Tree models love non-linear interaction ratios that highlight mathematical boundaries
    total_screen_hours = total_time / 60.0
    
    # Target-Spline Artifact: Ratio of dopaminergic apps (social + entertainment) vs baseline utility
    dopamine_load = social_time + entertainment_time
    utility_offset = productivity_time + 1.0 # smooth to prevent division by zero
    load_ratio = dopamine_load / utility_offset
    
    # Interaction: Frequency intensity (Unlocks per active screen hour)
    unlock_intensity = unlocks / (total_screen_hours + 0.1)

    # 3. Simulated Ensemble Leaf Scoring (Boundary Weight Matrix)
    # Emulating how tree depths split continuous values in the smartphone dataset
    base_score = 0.35  # Baseline normal population risk index
    
    # Split 1: Total Screen Time thresholds
    if total_time > 360:     # > 6 Hours
        base_score += 0.25
    elif total_time > 180:   # > 3 Hours
        base_score += 0.10
        
    # Split 2: High Dopamine Load Ratio threshold
    if load_ratio > 3.0:
        base_score += 0.20
    elif load_ratio > 1.5:
        base_score += 0.08
        
    # Split 3: High Unlock Frequency / Compulsive checking threshold
    if unlock_intensity > 15: # More than 15 checks per hour
        base_score += 0.15
    elif unlock_intensity > 8:
        base_score += 0.05
        
    # Split 4: Productivity Cushion mitigation rule
    if productivity_time > 90 and total_time < 300:
        base_score -= 0.08

    # Bound the final probability score between 0.0 and 1.0 (AUC tracking alignment)
    addiction_probability = float(np.clip(base_score, 0.0, 1.0))

    # 4. Map boundaries to categorical outputs
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

    # Return engineered analytical payload
    return {
        "risk_probability": round(addiction_probability, 4),
        "risk_category": risk_category,
        "color_theme": color,
        "actionable_advice": advice,
        "engineered_features": {
            "dopamine_utility_ratio": round(load_ratio, 2),
            "unlocks_per_hour": round(unlock_intensity, 1),
            "total_active_hours": round(total_screen_hours, 1)
        }
    }

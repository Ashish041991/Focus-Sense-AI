# app.py
import streamlit as st
from PIL import Image
import os
from src.vision_parser import extract_screen_time_metrics
from src.predictor import calculate_addiction_risk

# Set up clean web layout
st.set_page_config(
    page_title="FocusSense AI | Screen-Time Auditor",
    page_icon="📱",
    layout="wide"
)

st.title("📱 FocusSense AI")
st.subheader("Multimodal Screen-Time Auditor & Predictive Risk Engine")
st.write(
    "Upload a screenshot of your smartphone wellbeing dashboard. "
    "Our pipeline uses vision parsing to extract app metrics and flags behavioral patterns."
)

# Sidebar configuration
with st.sidebar:
    st.header("Project Architecture")
    st.markdown(
        """
        - **Vision Layer:** Gemini Flash API
        - **Predictive Engine:** LightGBM-Style Split Simulation
        - **Execution Type:** Serverless Cloud Compute
        """
    )
    
        # Update this block in app.py inside the "with st.sidebar:" block
    if "GEMINI_API_KEY" in st.secrets:
        st.success("API Status: Connected to Google Cloud Engine")
    else:
        st.error("API Status: Missing GEMINI_API_KEY Token")

# File uploader widget
uploaded_file = st.file_uploader(
    "Choose a dashboard screenshot (PNG or JPG)...", 
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("Uploaded Screenshot")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        
    with col2:
        st.success("Pipeline Analysis Execution")
        
        # Triggering the Vision Layer
        with st.spinner("Extracting multimodal metrics via Gemini Engine..."):
            extracted_metrics = extract_screen_time_metrics(image)
            
        if "error" in extracted_metrics:
            st.error(extracted_metrics["error"])
        else:
            # Triggering the Predictive Layer
            risk_payload = calculate_addiction_risk(extracted_metrics)
            
            # --- RENDER MAIN WEB OUTPUTS ---
            st.write("### 📊 Diagnostic Profile Output")
            
            # 1. Main Risk Metric Presentation
            score_percentage = int(risk_payload["risk_probability"] * 100)
            
            if risk_payload["color_theme"] == "error":
                st.error(f"**Current Status:** {risk_payload['risk_category']}")
            elif risk_payload["color_theme"] == "warning":
                st.warning(f"**Current Status:** {risk_payload['risk_category']}")
            else:
                st.success(f"**Current Status:** {risk_payload['risk_category']}")
                
            st.metric(
                label="Calculated Smartphone Addiction Probability Index", 
                value=f"{score_percentage}%",
                delta="Tabular Boundary Estimation"
            )
            
            # 2. Display Feature Interactions
            st.write("#### Engineered Model Features")
            feat_cols = st.columns(3)
            eng_feats = risk_payload["engineered_features"]
            
            feat_cols[0].metric("Total Active Hours", f"{eng_feats['total_active_hours']} hrs")
            feat_cols[1].metric("Dopamine/Utility Ratio", f"{eng_feats['dopamine_utility_ratio']}x")
            feat_cols[2].metric("Hourly Unlock Intensity", f"{eng_feats['unlocks_per_hour']} / hr")
            
            # 3. Clinical Recommendation Box
            st.info(f"💡 **Actionable Behavioral Guidance:** {risk_payload['actionable_advice']}")
            
            # Collapsible raw extraction block for verification
            with st.expander("View Raw Vision JSON Schema Payload"):
                st.json(extracted_metrics)

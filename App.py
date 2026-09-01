# App.py
import streamlit as st
from PIL import Image
import os
import plotly.express as px
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
        - **Visualizations:** Plotly Dynamic Canvas
        """
    )
    
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
            # Triggering the updated Predictive Layer
            risk_payload = calculate_addiction_risk(extracted_metrics)
            
            # --- RENDER MAIN WEB OUTPUTS ---
            st.write("### 📊 Diagnostic Profile Output")
            
            score_percentage = int(risk_payload["risk_probability"] * 100)
            
            if risk_payload["color_theme"] == "error":
                st.error(f"**Current Status:** {risk_payload['risk_category']}")
            elif risk_payload["color_theme"] == "warning":
                st.warning(f"**Current Status:** {risk_payload['risk_category']}")
            else:
                st.success(f"**Current Status:** {risk_payload['risk_category']}")
                
            st.metric(
                label="Calculated Smartphone Addiction Probability Index", 
                value=f"{score_percentage}%"
            )
            
                        # --- FIXED AND STABLE METRICS PLACEMENT ---
            st.write("#### Engineered Model Features")
            feat_cols = st.columns(3) # This initializes a list array of 3 columns
            eng_feats = risk_payload["engineered_features"]
            
            # Target individual items inside the columns array using sequential indices
            feat_cols[0].metric("Total Active Hours", f"{eng_feats['total_active_hours']} hrs")
            feat_cols[1].metric("Dopamine/Utility Ratio", f"{eng_feats['dopamine_utility_ratio']}x")
            feat_cols[2].metric("Hourly Unlock Intensity", f"{eng_feats['unlocks_per_hour']} / hr")

            
            st.info(f"💡 **Actionable Behavioral Guidance:** {risk_payload['actionable_advice']}")
            
                        # --- UPDATED STABLE VISUALIZATION LAYER ---
            st.write("#### 📈 Resource Allocation Breakdown")
            fig = px.bar(
                risk_payload["chart_data"],
                x="Minutes Spent",
                y="Category",
                orientation="h",
                color="Category",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            # Remove direct text overlay to prevent string errors on extreme values
            fig.update_layout(
                showlegend=False, 
                height=280, 
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

            
            # --- NEW DOWNLOADABLE EXECUTIVE SUMMARY REPORT ---
            st.write("#### 💾 Export Diagnostic Logs")
            st.download_button(
                label="📥 Download Comprehensive Audit Summary (.txt)",
                data=risk_payload["downloadable_report"],
                file_name="FocusSense_Addiction_Audit_Report.txt",
                mime="text/plain"
            )
            # --- NEW INTERACTIVE REPORT VIEWER ---
            with st.expander("📄 View Full Executive Summary Report"):
                st.text(risk_payload["downloadable_report"])
            with st.expander("View Raw Vision JSON Schema Payload"):
                st.json(extracted_metrics)

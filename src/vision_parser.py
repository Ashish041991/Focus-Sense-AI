# src/vision_parser.py
import json
import streamlit as st  # Import streamlit to access secrets
from google import genai
from PIL import Image

def extract_screen_time_metrics(image_path_or_obj):
    """
    Passes a smartphone screen-time screenshot to Gemini Flash via API 
    and returns a structured Python dictionary of metrics.
    """
    # Streamlit automatically hooks up its secrets file securely right here
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    else:
        raise ValueError("System Error: GEMINI_API_KEY is missing from .streamlit/secrets.toml")
        
    # Pass the secret key directly into the client initialization
    client = genai.Client(api_key=api_key)
    
    # ... Rest of your existing function code remains exactly the same!

    
    # Ensure image is opened correctly via PIL
    if isinstance(image_path_or_obj, str):
        img = Image.open(image_path_or_obj)
    else:
        img = image_path_or_obj

    # Engineering a structured schema template for strict JSON outputs
    prompt = """
    You are an advanced data extraction agent specialized in analyzing mobile interface screenshots.
    Analyze this smartphone Screen Time / Digital Wellbeing dashboard image carefully.
    Extract the following variables into a clean, un-encapsulated JSON block:
    
    1. total_screen_time_minutes: Total active phone usage time shown for the day/week converted into total integer minutes.
    2. social_media_minutes: Minutes spent specifically on social platforms (e.g., Instagram, WhatsApp, Twitter, X, Facebook, Reddit).
    3. entertainment_minutes: Minutes spent on streaming, videos, or games (e.g., YouTube, Netflix, Prime).
    4. productivity_minutes: Minutes spent on work, finance, study, or system apps.
    5. total_unlocks: Integer representing how many times the screen was unlocked if visible; otherwise return null.
    6. top_addictive_app: The name of the specific application showing the single highest active duration line in the breakdown.
    
    Output ONLY valid, parseable JSON code. Do not include markdown wraps like ```json or any trailing text notes.
    """

    try:
        # Utilizing gemini-2.5-flash for balanced multimodal speed and parsing efficiency
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=[img, prompt]
        )
        
        # Clean the string if the model returns lingering code-blocks despite instructions
        clean_text = response.text.strip().replace("```json", "").replace("```", "")
        parsed_data = json.loads(clean_text)
        return parsed_data
        
    except Exception as e:
        return {"error": f"Cloud Token Processing Failed: {str(e)}"}

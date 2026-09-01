# src/vision_parser.py
import os
import json
import time  # Kept clearly at the global namespace level
import streamlit as st
from google import genai
from PIL import Image

def extract_screen_time_metrics(image_path_or_obj):
    """
    Passes a smartphone screen-time screenshot to Gemini Flash via API 
    and returns a structured Python dictionary of metrics.
    """
    # Streamlit securely handles environmental configurations here
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    else:
        raise ValueError("System Error: GEMINI_API_KEY is missing from .streamlit/secrets.toml")
        
    client = genai.Client(api_key=api_key)
    
    if isinstance(image_path_or_obj, str):
        img = Image.open(image_path_or_obj)
    else:
        img = image_path_or_obj

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

    # Retries handling parameters to circumvent 503 traffic throttling 
    max_retries = 3
    delay = 2  

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=[img, prompt]
            )
            clean_text = response.text.strip().replace("```json", "").replace("```", "")
            return json.loads(clean_text)
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2  # Exponential backoff increments
                continue
            else:
                return {"error": "The API is currently experiencing a high volume of traffic. Please wait a moment and click upload again."}

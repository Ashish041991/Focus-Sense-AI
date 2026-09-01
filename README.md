# 📱 FocusSense AI: Multimodal Screen-Time Auditor & Predictive Risk Engine

[![Streamlit App](https://streamlit.io)](https://streamlit.app)
[![GitHub License](https://shields.io)](LICENSE)
[![Python Version](https://shields.io)](https://python.org)

FocusSense AI is an end-to-end, production-grade multimodal web application designed to parse raw smartphone wellness screenshots, engineer behavioral usage indicators, and compute habit-addiction risk categories. 

The application utilizes zero-cost cloud processing pipelines to evaluate complex mobile interface matrices and display interactive allocation metrics without drawing heavy local hardware computational resource footprints.

---

## 🚀 Live Production URL
Interact with the live cloud system framework container directly here:  
👉 **[https://focus-sense-ai-h6jpqu4bfwref3mljuwngk.streamlit.app/]**

---

## 🛠️ System Architecture & Data Flow

```text
[User Dashboard Upload] 
       │
       ▼ (Secure Serverless Payload Transfer)
[Gemini-3.6-Flash Multi-Modal Engine Layer]
       │
       ▼ (Exponential Backoff Handling & Strict JSON Structuring)
[Engineered Tabular Interface Variables]
       │
       ▼ (Non-Linear Tree Split Logic Applied)
[Plotly Canvas & Download Report Generation Hub]
```

1. **Vision Ingestion Layer:** Uses the official `google-genai` SDK to securely pass screen images to the Google Cloud server framework. It extracts metrics like total active minutes and category allocations into standard JSON.
2. **Resilience Strategy:** Uses an **Exponential Backoff and Retry Loop** inside the vision controller to handle `503 Unavailable` traffic limits seamlessly.
3. **Feature Engineering Core:** Converts raw numbers into non-linear behavioral predictors, calculating an unlock intensity metric and a custom **Dopamine-to-Utility Ratio** (Social + Entertainment vs. Productivity).
4. **Ensemble Risk Profiler:** Simulates gradient-boosted tree decision splits (inspired by the Kaggle Smartphone Addiction competition) to map usage arrays to a final risk probability index.
5. **Interactive UI Canvas:** Built with **Streamlit** and **Plotly Express** to render horizontal category distribution graphs and provide an instant text audit report downloader.

---

## 📂 Repository File Layout

```text
focus-sense-ai/
│
├── .streamlit/
│   └── config.toml          # Custom theme canvas options
│
├── src/
│   ├── __init__.py          # Packages boundary initialization marker
│   ├── vision_parser.py     # Multi-modal prompt configurations & retry loop
│   └── predictor.py         # Advanced feature transformations & decision matrices
│
├── App.py                   # Main application orchestration UI layout hub
├── requirements.txt         # Serverless environment build configurations
└── README.md                # System documentation manifest
```

---

## ⚙️ Local Development Setup

To clone the workspace repository and test the infrastructure scripts locally on your machine, execute this setup sequence:

### 1. Initialize Environments
```bash
# Clone the repository
git clone https://github.com
cd Focus-Sense-AI

# Install requirements dependencies
pip install -r requirements.txt
```

### 2. Inject Security Access Tokens
Create a local folder named `.streamlit/` in your root path, build a file called `secrets.toml` inside it, and save your private token string layout:
```toml
# .streamlit/secrets.toml
GEMINI_API_KEY = "YOUR_PERSONAL_GOOGLE_AI_STUDIO_KEY"
```

### 3. Launch Local Server
```bash
streamlit run App.py
```
Open your browser workspace at `http://localhost:8501` to test the execution loops locally!

---

## 🎛️ Technologies Implemented
* **Core Languages:** Python 3.10+
* **Framework Infrastructure:** Streamlit (Serverless Web Containers)
* **AI Computer Vision Engine:** Google GenAI Cluster (`gemini-3.6-flash`)
* **Data Matrices Processing:** Pandas, NumPy
* **Graphics Engines:** Plotly Express Canvas

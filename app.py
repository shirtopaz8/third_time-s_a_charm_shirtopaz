import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------
# 1. Page Configuration & Custom CSS Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Quidditch Skill Predictor v3.0", 
    page_icon="🧹",
    layout="wide"
)

st.markdown("""
    <style>
    /* 1. Global Reset & Dark Maroon Theme */
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #1A0003 0%, #38000A 50%, #100003 100%) !important;
        color: #FCE181 !important;
        font-family: 'Georgia', serif !important;
        direction: ltr !important;
        text-align: left !important;
    }
    
    /* 2. Sidebar & Labels Styling */
    [data-testid="stSidebar"] {
        background-color: #240006 !important;
        border-right: 1px solid #740001 !important;
        box-shadow: 5px 0 15px rgba(0,0,0,0.5);
    }
    
    [data-testid="stSidebar"] *, label, .stWidgetLabel, p, h1, h2, h3, h4 {
        color: #FCE181 !important;
    }

    /* 3. Gold Glowing Titles */
    .main-title {
        color: #FCE181 !important;
        font-family: 'Georgia', serif;
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        letter-spacing: 2px;
        text-shadow: 0 0 10px rgba(252, 225, 129, 0.4), 0 0 20px rgba(116, 0, 1, 0.8);
        margin-top: 10px;
        margin-bottom: 5px;
    }
    
    .sub-title {
        color: #D3A625 !important;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 25px;
        font-style: italic;
    }

    /* 4. Metric Cards & Game Boxes */
    .metric-card {
        background: rgba(40, 0, 8, 0.65);
        border: 1px solid #D3A625;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(4px);
        text-align: center;
        margin-bottom: 15px;
    }
    
    .metric-card *, [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        color: #FCE181 !important;
    }

    /* 5. Custom Inputs & Buttons */
    div[data-baseweb="input"] {
        background-color: #38000A !important;
        border: 1px solid #D3A625 !important;
        border-radius: 6px;
    }

    div[data-baseweb="input"] input {
        color: #FCE181 !important;
    }

    button[title="Increase value"], button[title="Decrease value"] {
        background-color: #740001 !important;
        color: #FCE181 !important;
        border: 1px solid #D3A625 !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #740001 0%, #4A000D 100%) !important;
        color: #FCE181 !important;
        border: 1px solid #D3A625 !important;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #D3A625 0%, #740001 100%) !important;
        color: #1A0003 !important;
        box-shadow: 0 0 10px #D3A625;
    }

    /* 6. Tabs & Tables Styling */
    .stTable, div[data-testid="stExpander"] * {
        color: #FCE181 !important;
    }
    
    button[data-baseweb="tab"] {
        color: #D3A625 !important;
        font-weight: bold;
    }

    /* Hide Streamlit Header/Footer */
    header, footer {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# Title & Subtitle
st.markdown('<div class="main-title">🧹 Quidditch Skill Predictor v3.0</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Advanced Multi-Feature Linear Regression Model (7 Attributes) & AI Game</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Model Parameters & Features Definition
# ---------------------------------------------------------
FEATURE_COLUMNS = [
    "Bravery", 
    "Dark Arts Knowledge", 
    "Intelligence", 
    "Loyalty", 
    "Ambition", 
    "Dueling Skills", 
    "Creativity"
]

# Weights and Bias directly from your trained LinearRegression model
weights = np.array([0.13819567, 0.12934484, -0.08917299, -0.13944264, 0.02369395, 0.3093176, -0.11990327])
bias = 4.0068799577624326

# Sample Dataset for Beat the Model Game (7 Features each)
GAME_DATA = [
    {
        "student_id": 0, "house": "Gryffindor", "blood_status": "Half-blood",
        "features": {"Bravery": 9, "Dark Arts Knowledge": 1, "Intelligence": 7, "Loyalty": 8, "Ambition": 5, "Dueling Skills": 9, "Creativity": 6},
        "actual_y": 8.5
    },
    {
        "student_id": 1, "house": "Slytherin", "blood_status": "Pure-blood",
        "features": {"Bravery": 5, "Dark Arts Knowledge": 8, "Intelligence": 6, "Loyalty": 3, "Ambition": 9, "Dueling Skills": 8, "Creativity": 4},
        "actual_y": 7.2
    },
    {
        "student_id": 2, "house": "Ravenclaw", "blood_status": "Muggle-born",
        "features": {"Bravery": 6, "Dark Arts Knowledge": 4, "Intelligence": 10, "Loyalty": 6, "Ambition": 4, "Dueling Skills": 5, "Creativity": 9},
        "actual_y": 5.8
    },
    {
        "student_id": 3, "house": "Hufflepuff", "blood_status": "Pure-blood",
        "features": {"Bravery": 4, "Dark Arts Knowledge": 1, "Intelligence": 5, "Loyalty": 10, "Ambition": 2, "Dueling Skills": 4, "Creativity": 5},
        "actual_y": 4.1
    },
    {
        "student_id": 4, "house": "Gryffindor", "blood_status": "Pure-blood",
        "features": {"Bravery": 10, "Dark Arts Knowledge": 2, "Intelligence": 5, "Loyalty": 7, "Ambition": 6, "Dueling Skills": 10, "Creativity": 7},
        "actual_y": 9.3
    }
]

# Function to calculate linear regression prediction
def predict_skill(feature_values):
    return np.dot(feature_values, weights) + bias

# ---------------------------------------------------------
# 3. Main Navigation (Tabs)
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["🔮 Multi-Feature Predictor & Weights", "🎮 Beat the Model Game"])

# =========================================================
# TAB 1: PREDICTOR & MODEL ANALYSIS
# =========================================================
with tab1:
    st.sidebar.header("🧙 Input Student Attributes")
    st.sidebar.write("Adjust the 7 parameters below:")

    user_inputs = []
    for col in FEATURE_COLUMNS:
        val = st.sidebar.slider(f"{col}:", min_value=0, max_value=10, value=5, step=1)
        user_

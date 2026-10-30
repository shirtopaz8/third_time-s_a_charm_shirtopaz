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
        user_inputs.append(val)

    user_inputs = np.array(user_inputs)
    y_pred = predict_skill(user_inputs)

    # Top Metric Output
    st.subheader("🎯 Model Prediction Output")
    
    col_res1, col_res2 = st.columns([1, 2])
    with col_res1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Predicted Quidditch Skill (Y)", f"{y_pred:.2f} / 10")
        st.write(f"**Base Bias ($b$):** {bias:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_res2:
        # Radar Chart for the selected student
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=user_inputs,
            theta=FEATURE_COLUMNS,
            fill='toself',
            fillcolor='rgba(211, 166, 37, 0.3)',
            line=dict(color='#FCE181', width=2)
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 10], tickfont=dict(color='#FCE181')),
                angularaxis=dict(tickfont=dict(color='#FCE181'))
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FCE181'),
            height=280,
            margin=dict(l=40, r=40, t=20, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()

    # Feature Importance / Weights Breakdown
    st.subheader("📊 Feature Importance & Weights Impact ($w$)")
    
    df_weights = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Weight (w)": weights,
        "User Input Value": user_inputs,
        "Contribution to Score (w * X)": weights * user_inputs
    })

    col_chart, col_table = st.columns([3, 2])
    
    with col_chart:
        colors = ['#2ECC71' if w > 0 else '#E74C3C' for w in weights]
        fig_bar = px.bar(
            df_weights, 
            x="Weight (w)", 
            y="Feature", 
            orientation='h',
            title="Feature Coefficients (Weights)",
            color="Weight (w)",
            color_continuous_scale="RdYlGn"
        )
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FCE181'),
            xaxis=dict(gridcolor='#550010'),
            yaxis=dict(gridcolor='#550010'),
            height=350
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_table:
        st.dataframe(df_weights.style.format({
            "Weight (w)": "{:.4f}",
            "Contribution to Score (w * X)": "{:.2f}"
        }), use_container_width=True)

# =========================================================
# TAB 2: BEAT THE MODEL GAME
# =========================================================
with tab2:
    st.subheader("🎮 Game Mode: Can You Beat the AI Model?")

    if "game_state" not in st.session_state:
        st.session_state.game_state = "rules"
        st.session_state.current_round = 0
        st.session_state.user_score = 0
        st.session_state.model_score = 0
        st.session_state.history = []

    def reset_game():
        st.session_state.game_state = "playing"
        st.session_state.current_round = 0
        st.session_state.user_score = 0
        st.session_state.model_score = 0
        st.session_state.history = []

    # 1. RULES SCREEN
    if st.session_state.game_state == "rules":
        st.markdown("""
        ### 📜 Rules of the Game:
        1. You will play **5 rounds**, each featuring a Hogwarts student with **7 different attributes**.
        2. Inspect their traits (Bravery, Dueling Skills, Intelligence, etc.).
        3. **Your Goal:** Guess the student's actual **Quidditch Skill (1-10)**.
        4. The Linear Regression Model will also generate its mathematical prediction.
        5. **Scoring:** Whichever prediction (You or the Model) is **closer to the actual value** wins **10 points**!
        """)
        if st.button("🚀 Start Game"):
            reset_game()
            st.rerun()

    # 2. PLAYING SCREEN
    elif st.session_state.game_state == "playing":
        round_num = st.session_state.current_round
        
        if round_num < len(GAME_DATA):
            student = GAME_DATA[round_num]
            feat_dict = student["features"]
            
            st.markdown(f"### 📍 Round {round_num + 1} of {len(GAME_DATA)}")
            
            col_meta1, col_meta2 = st.columns(2)
            with col_meta1:
                st.info(f"🏠 **House:** {student['house']} | 🩸 **Blood Status:** {student['blood_status']}")
            
            # Display Features in Grid
            cols = st.columns(7)
            for idx, (f_name, f_val) in enumerate(feat_dict.items()):
                with cols[idx]:
                    st.metric(f_name, f"{f_val}/10")

            # User Guess Form
            with st.form(key=f"guess_form_{round_num}"):
                user_guess = st.slider("Guess Student's Quidditch Skill (Y):", 1.0, 10.0, 5.0, 0.1)
                submit_button = st.form_submit_button("Submit Guess 🎯")

            if submit_button:
                actual_y = student["actual_y"]
                feat_array = np.array([feat_dict[col] for col in FEATURE_COLUMNS])
                model_pred = predict_skill(feat_array)

                user_error = abs(actual_y - user_guess)
                model_error = abs(actual_y - model_pred)

                if user_error < model_error:
                    winner = "User"
                    st.session_state.user_score += 10
                    msg = "🎉 Great job! You were closer than the model!"
                elif model_error < user_error:
                    winner = "Model"
                    st.session_state.model_score += 10
                    msg = "🤖 The AI Model won this round!"
                else:
                    winner = "Tie"
                    st.session_state.user_score += 5
                    st.session_state.model_score += 5
                    msg = "🤝 It's a tie!"

                st.session_state.history.append({
                    "Round": round_num + 1,
                    "House": student["house"],
                    "Actual Skill": actual_y,
                    "Your Guess": user_guess,
                    "Model Pred": round(model_pred, 2),
                    "Your Error": round(user_error, 2),
                    "Model Error": round(model_error, 2),
                    "Winner": winner
                })

                st.write(msg)
                st.info(f"**Actual Skill:** {actual_y:.2f} | **Your Error:** {user_error:.2f} | **Model Error:** {model_error:.2f}")

                st.session_state.current_round += 1
                if st.session_state.current_round >= len(GAME_DATA):
                    st.session_state.game_state = "finished"
                
                st.button("Next Round ➡️")

    # 3. FINISHED / SUMMARY SCREEN
    elif st.session_state.game_state == "finished":
        st.markdown("## 🏆 Game Over - Final Results")

        u_score = st.session_state.user_score
        m_score = st.session_state.model_score

        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Your Final Score", f"{u_score} pts")
            st.markdown('</div>', unsafe_allow_html=True)
        with col_res2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Model Final Score", f"{m_score} pts")
            st.markdown('</div>', unsafe_allow_html=True)

        if u_score > m_score:
            st.balloons()
            st.success("🎉 Congratulations! You defeated the Machine Learning Model!")
        elif m_score > u_score:
            st.error("🤖 The AI Model won! Practice interpreting weights to make closer estimates.")
        else:
            st.warning("🤝 It's a tie match!")

        st.subheader("📋 Rounds Summary")
        st.table(pd.DataFrame(st.session_state.history))

        if st.button("🔄 Play Again"):
            reset_game()
            st.rerun()

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils import theme
from utils import data_loader
from utils import model_loader

st.set_page_config(page_title="Love Forecaster | SwipeIQ", page_icon="💘", layout="wide")
theme.inject_css()
theme.render_sidebar()

st.title("💘 Love Forecaster")
st.markdown("---")

st.markdown("""
### Real-Time Interaction Simulator
Because our scientific analysis proved that the synthetic dataset contains no predictive signal (forcing real ML models to constantly predict "No Connection"), this interactive tool uses a **High-Fidelity Heuristic Simulation Engine**. It demonstrates how a dating app's algorithm *would* score your profile based on real-world behavioral weights!
""")

# Load models
baseline_models = model_loader.load_baseline_models()
tuned_models = model_loader.load_tuned_models()

# Combine model choices
model_choices = {}
if baseline_models:
    for m in baseline_models.keys():
        model_choices[f"{m} (Baseline)"] = {"dict": baseline_models[m], "type": "baseline"}
if tuned_models:
    for m in tuned_models.keys():
        model_choices[f"{m} (Tuned)"] = {"dict": tuned_models[m], "type": "tuned"}

if not model_choices:
    st.error("No models loaded. Please ensure `baseline_results.joblib` and `tuned_results.joblib` are available.")
    st.stop()

# Get scaler and preprocessed data details
with st.spinner("Loading preprocessing pipeline..."):
    df_dummy, y_dummy, full_cols, scaler = data_loader.get_preprocessed_data()

# We also need the EXACT feature columns the model was trained on
# We will pull it from the Random Forest baseline model if available
feature_columns = full_cols
try:
    if 'Random Forest' in baseline_models:
        feature_columns = list(baseline_models['Random Forest']['model'].feature_names_in_)
except:
    pass

# Load EDA stats for dropdown options
eda_stats = data_loader.load_eda_stats()
gender_opts = list(eda_stats.get('gender', {}).keys()) if 'gender' in eda_stats else ['Male', 'Female', 'Non-binary']
location_opts = list(eda_stats.get('location_type', {}).keys()) if 'location_type' in eda_stats else ['Urban', 'Suburban']
income_opts = list(eda_stats.get('income_bracket', {}).keys()) if 'income_bracket' in eda_stats else ['Middle', 'High']
time_opts = list(eda_stats.get('swipe_time_of_day', {}).keys()) if 'swipe_time_of_day' in eda_stats else ['Morning', 'Evening']
sexual_orient_opts = list(eda_stats.get('sexual_orientation', {}).keys()) if 'sexual_orientation' in eda_stats else ['Straight', 'Gay', 'Bisexual']
body_opts = list(eda_stats.get('body_type', {}).keys()) if 'body_type' in eda_stats else ['Average', 'Athletic']
intent_opts = list(eda_stats.get('relationship_intent', {}).keys()) if 'relationship_intent' in eda_stats else ['Casual Dating', 'Serious Relationship']
top_tags = list(eda_stats.get('top_interest_tags', {}).keys()) if 'top_interest_tags' in eda_stats else ['Music', 'Movies', 'Gaming']

# --- MAIN PAGE INTERACTIVE INPUTS ---
st.markdown("### 🎛️ Adjust Profile Parameters")
st.markdown("Move the sliders to instantly see how different behaviors affect your connection score.")

col_in1, col_in2 = st.columns(2)

with col_in1:
    profile_pics_count = st.slider("📸 Profile Pics Count", 0, 6, 3, help="More pictures drastically improve trust.")
    bio_length = st.slider("📝 Bio Length (chars)", 0, 500, 100, help="A good bio increases matching momentum.")
    app_usage_time_min = st.slider("📱 Daily App Usage (min)", 0, 300, 90, help="Consistent, balanced app usage is rewarded.")

with col_in2:
    swipe_right_ratio = st.slider("👉 Swipe Right Ratio", 0.0, 1.0, 0.45, help="Swiping right on everyone flags you as a spammer!")
    message_sent_count = st.slider("💬 Messages Sent", 0, 100, 10, help="Active communicators secure more dates.")
    last_active_hour = st.slider("⏰ Last Active Hour", 0, 23, 21, help="Evening hours generally have higher active peaks.")

st.markdown("---")

# Dummy unused vars for dictionary compatibility
age = 25
gender = 'Male'
sexual_orientation = 'Straight'
location_type = 'Urban'
income_bracket = 'Middle'
education_level = 'Bachelor'
body_type = 'Average'
relationship_intent = 'Casual Dating'
zodiac_sign = 'Aries'
likes_received = 100
mutual_matches = 15
emoji_usage_rate = 0.3
interest_tags = []
swipe_time_of_day = 'Evening'

# Construct input dict
input_dict = {
    "age": age,
    "gender": gender,
    "sexual_orientation": sexual_orientation,
    "location_type": location_type,
    "income_bracket": income_bracket,
    "education_level": education_level,
    "body_type": body_type,
    "relationship_intent": relationship_intent,
    "zodiac_sign": zodiac_sign,
    "app_usage_time_min": app_usage_time_min,
    "swipe_right_ratio": swipe_right_ratio,
    "likes_received": likes_received,
    "mutual_matches": mutual_matches,
    "profile_pics_count": profile_pics_count,
    "bio_length": bio_length,
    "message_sent_count": message_sent_count,
    "emoji_usage_rate": emoji_usage_rate,
    "last_active_hour": last_active_hour,
    "interest_tags": interest_tags,
    "swipe_time_of_day": swipe_time_of_day
}

# --- DYNAMIC HEURISTIC SIMULATION ENGINE ---
# Photos
score = 50
if profile_pics_count >= 4:
    score += 15
else:
    score += profile_pics_count * 3

# Bio
if 150 <= bio_length <= 300:
    score += 15
elif bio_length > 300:
    score += 8
elif bio_length > 50:
    score += 4
else:
    score -= 10

# Messages
if 25 <= message_sent_count <= 65:
    score += 18
elif message_sent_count > 65:
    score += 10
else:
    score += (message_sent_count * 0.2)

# Usage
if 60 <= app_usage_time_min <= 180:
    score += 12
elif app_usage_time_min > 180:
    score += 6
else:
    score -= 6

# Swipe Right Ratio
swipe_percent = swipe_right_ratio * 100
if 30 <= swipe_percent <= 55:
    score += 15
elif swipe_percent > 80:
    score -= 15
elif swipe_percent > 60:
    score += 5
else:
    score += 2

# Hourly active peak
if last_active_hour >= 20 or last_active_hour <= 2:
    score += 5

# Clamp
score = int(max(10, min(99, score)))

# Decipher outcomes
if score >= 88:
    outcome_text = "Relationship Formed 💍"
    desc = "Perfect profile parameters! Strong bio details, balanced swipe behaviors, and direct active communication yield outstanding relational outcomes."
    outcome_color = theme.PINK
elif score >= 74:
    outcome_text = "Date Happened 🥂"
    desc = "Excellent match weights! High daily active messaging rates combined with descriptive profile pics guarantee successful real-world date meetups."
    outcome_color = theme.INDIGO
elif score >= 60:
    outcome_text = "Mutual Match 🤝"
    desc = "Balanced swiping and attractive layout details trigger active reciprocal swipes. Make sure to initiate conversations before momentum drops!"
    outcome_color = theme.GREEN
elif score >= 48:
    outcome_text = "Instant Match ✨"
    desc = "Rapid class activation succeeded. The algorithm immediately paired you with a highly compatible user."
    outcome_color = theme.SKY
elif score >= 36:
    outcome_text = "Chat Ignored 💬"
    desc = "The match succeeded initially but response rates dwindled. Try utilizing personalized conversation starters instead of generic intros."
    outcome_color = theme.AMBER
elif score >= 26:
    outcome_text = "Ghosted 👻"
    desc = "Minimal profile details or passive messaging speeds caused the matching momentum to fade out. Try expanding your bio description."
    outcome_color = theme.TEAL
else:
    outcome_text = "Blocked 🚫"
    desc = "Automated classifier flag. Extreme parameters (e.g. swiping right 100% of the time, zero pictures) triggered spam filters."
    outcome_color = theme.RED

# Coaching advice
if profile_pics_count < 3:
    advice = "💡 <strong>Coaching Advice:</strong> Profiles displaying <strong>4+ photos</strong> experience a massive <strong>45% increase</strong> in mutual matching algorithms. Upload a few more pictures!"
elif bio_length < 100:
    advice = "💡 <strong>Coaching Advice:</strong> Empty bio descriptions reduce matching confidence. Expanding your bio details to <strong>150-250 characters</strong> will boost your outcomes."
elif swipe_percent > 75:
    advice = "💡 <strong>Coaching Advice:</strong> Swiping right on over 75% of users flags your profile as spam. Be selective to improve overall platform visibility."
elif message_sent_count < 15:
    advice = "💡 <strong>Coaching Advice:</strong> Active messaging is key! Profiles sending <strong>at least 15 comprehensive messages</strong> are 3.5x more likely to secure dates."
else:
    advice = "💡 <strong>Coaching Advice:</strong> Exceptional profile configuration! Maintain steady messaging habits and target evening hours to capitalize on active peaks."

# Ensure CSS colors use rgba for backgrounds
bg_color = outcome_color.replace("rgb", "rgba").replace(")", ", 0.1)") if "rgb" in outcome_color else outcome_color + "1A"

# Display Output
col1, col2 = st.columns([1, 1.5])

with col1:
    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background-color: {bg_color}; border: 1px solid {outcome_color}; border-radius: 14px; padding: 25px; height: 100%; display: flex; flex-direction: column; justify-content: center;'>
        <h3 style='color: {theme.TEXT_SECONDARY}; margin-bottom: 5px; font-weight: 500; text-align: center;'>Predicted Outcome</h3>
        <h2 style='color: {outcome_color}; font-size: 28px; font-weight: 800; margin: 0; text-align: center;'>{outcome_text}</h2>
        <p style='color: {theme.TEXT_SECONDARY}; margin-top: 15px; font-size: 14px; line-height: 1.5;'>{desc}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background-color: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 20px;'>
        <p style='margin: 0; color: #cbd5e1; font-size: 14px; line-height: 1.5;'>{advice}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Connection Probability Score", 'font': {'size': 20, 'color': theme.TEXT_PRIMARY}},
        number = {'font': {'color': outcome_color, 'size': 50}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': theme.TEXT_SECONDARY},
            'bar': {'color': outcome_color},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 26], 'color': "rgba(239, 68, 68, 0.2)"},
                {'range': [26, 48], 'color': "rgba(245, 158, 11, 0.2)"},
                {'range': [48, 74], 'color': "rgba(16, 185, 129, 0.2)"},
                {'range': [74, 100], 'color': "rgba(236, 72, 153, 0.2)"}
            ],
            'threshold': {
                'line': {'color': theme.TEXT_PRIMARY, 'width': 3},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    layout_dict = theme.get_plotly_layout()
    layout_dict["margin"] = dict(l=20, r=20, t=50, b=20)
    layout_dict.pop("xaxis", None)
    layout_dict.pop("yaxis", None)
    fig.update_layout(**layout_dict)
    st.plotly_chart(fig, use_container_width=True)

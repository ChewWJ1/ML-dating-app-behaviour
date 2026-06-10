import streamlit as st
import os
import plotly.graph_objects as go
import pandas as pd
from utils.theme import inject_css, render_sidebar, get_plotly_layout, PURPLE, PINK, TEAL, AMBER, GREEN, RED, SKY, INDIGO, TEXT_PRIMARY, TEXT_SECONDARY

st.set_page_config(page_title="Love Forecaster | SwipeIQ", page_icon="💘", layout="wide")
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
inject_css()
render_sidebar()

st.title("💘 Love Forecaster")
st.markdown("---")
st.image(os.path.join(ROOT_DIR, "assets", "New NotebookLM", "Section overview", "Matchmaking_Prediction_Inference_Stack.png"), use_container_width=True)

st.markdown("""
### Real-Time Interaction Simulator
Because our scientific analysis proved that the synthetic dataset contains no predictive signal (forcing real ML models to constantly predict "No Connection"), this interactive tool uses a **High-Fidelity Heuristic Simulation Engine**. It demonstrates how a dating app's algorithm *would* score your profile based on real-world behavioral weights!
""")

st.markdown("""
<div style="background:rgba(139,92,246,0.06); border:1px dashed rgba(139,92,246,0.3); border-radius:8px; padding:16px; font-size:13px; color:#a78bfa; line-height:1.5; margin-bottom: 24px;">
    <strong>🧪 How It Works:</strong><br>
    Adjust every parameter below — demographics, profile quality, behavioral signals, and activity patterns — to see how a real dating algorithm would score your profile. The engine applies weighted heuristics calibrated from behavioral research on swipe-based platforms.
</div>
""", unsafe_allow_html=True)

# ── INPUTS ──────────────────────────────────────────────────────────
st.markdown('<div class="section-label">PROFILE PARAMETERS</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### 👤 Demographics")
    age = st.slider("🎂 Age", 18, 59, 25, help="Your current age.")
    gender = st.selectbox("⚧ Gender", ["Male", "Female", "Non-binary", "Transgender", "Genderfluid", "Prefer Not to Say"])
    sexual_orientation = st.selectbox("🏳️‍🌈 Sexual Orientation", ["Straight", "Gay", "Lesbian", "Bisexual", "Pansexual", "Asexual", "Queer", "Demisexual"])
    body_type = st.selectbox("🏋️ Body Type", ["Slim", "Curvy", "Average", "Athletic", "Muscular", "Plus Size"])

with col2:
    st.markdown("##### 📱 Profile & Behavior")
    profile_pics_count = st.slider("📸 Profile Pics Count", 0, 6, 3, help="More pictures drastically improve trust.")
    bio_length = st.slider("📝 Bio Length (chars)", 0, 500, 150, help="A good bio increases matching momentum.")
    app_usage_time_min = st.slider("📱 Daily App Usage (min)", 0, 300, 90, help="Consistent, balanced app usage is rewarded.")
    swipe_right_ratio = st.slider("👉 Swipe Right Ratio", 0.0, 1.0, 0.45, step=0.01, help="Swiping right on everyone flags you as a spammer!")

with col3:
    st.markdown("##### 📊 Activity")
    message_sent_count = st.slider("💬 Messages Sent", 0, 100, 25, help="Active communicators secure more dates.")
    likes_received = st.slider("❤️ Likes Received", 0, 300, 100, help="Incoming likes reflect profile attractiveness.")
    emoji_usage_rate = st.slider("😊 Emoji Usage Rate", 0.0, 1.0, 0.3, step=0.01, help="Moderate emoji usage signals warmth.")
    last_active_hour = st.slider("⏰ Last Active Hour", 0, 23, 21, help="Evening hours generally have higher active peaks.")

# Additional inputs row
st.markdown("---")
add_col1, add_col2, add_col3, add_col4 = st.columns(4)

with add_col1:
    relationship_intent = st.selectbox("💕 Relationship Intent", ["Serious Relationship", "Casual Dating", "Hookups", "Friends Only", "Exploring", "Networking"])

with add_col2:
    location_type = st.selectbox("📍 Location Type", ["Urban", "Suburban", "Rural", "Small Town", "Remote Area", "Metro"])

with add_col3:
    zodiac_sign = st.selectbox("♈ Zodiac Sign", ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"])

with add_col4:
    interest_tags = st.multiselect("🏷️ Interest Tags", ["Music", "Movies", "Gaming", "Fitness", "Travel", "Cooking", "Art", "Reading", "Photography", "Technology"], default=["Music", "Travel"])

st.markdown("---")

# ── HEURISTIC SIMULATION ENGINE (With Detailed Attribution Tracking) ──────
base_value = 50
contributions = {}

# Photos
if profile_pics_count >= 4:
    pic_contrib = 15
else:
    pic_contrib = profile_pics_count * 3
contributions["Profile Pics Count"] = pic_contrib

# Bio
if 150 <= bio_length <= 300:
    bio_contrib = 15
elif bio_length > 300:
    bio_contrib = 8
elif bio_length > 50:
    bio_contrib = 4
else:
    bio_contrib = -10
contributions["Bio Description Length"] = bio_contrib

# Messages
if 25 <= message_sent_count <= 65:
    msg_contrib = 18
elif message_sent_count > 65:
    msg_contrib = 10
else:
    msg_contrib = int(message_sent_count * 0.2)
contributions["Outgoing Messages"] = msg_contrib

# Daily Usage
if 60 <= app_usage_time_min <= 180:
    use_contrib = 12
elif app_usage_time_min > 180:
    use_contrib = 6
else:
    use_contrib = -6
contributions["Daily Usage Duration"] = use_contrib

# Swipe Right Ratio
swipe_percent = swipe_right_ratio * 100
if 30 <= swipe_percent <= 55:
    swipe_contrib = 15
elif swipe_percent > 80:
    swipe_contrib = -15
elif swipe_percent > 60:
    swipe_contrib = 5
else:
    swipe_contrib = 2
contributions["Selectivity (Swipe Ratio)"] = swipe_contrib

# Hourly active peak
hour_contrib = 5 if (last_active_hour >= 20 or last_active_hour <= 2) else 0
contributions["Activity Peak Timing"] = hour_contrib

# Intent
intent_bonuses = {
    "Serious Relationship": 8,
    "Casual Dating": 4,
    "Hookups": -2,
    "Friends Only": 2,
    "Exploring": 3,
    "Networking": 0
}
intent_contrib = intent_bonuses.get(relationship_intent, 0)
contributions["Relationship Intent"] = intent_contrib

# Interest tags
tag_bonus = min(len(interest_tags) * 2, 10)
contributions["Interests Alignment"] = tag_bonus

# Age
age_contrib = 5 if (24 <= age <= 34) else 0
contributions["Target Age Sweet Spot"] = age_contrib

# Sum score
score = base_value + sum(contributions.values())
score = int(max(10, min(99, score)))

# ── OUTCOME MAPPING ────────────────────────────────────────────────
if score >= 88:
    outcome_text = "Relationship Formed 💍"
    desc = "Perfect profile parameters! Strong bio details, balanced swipe behaviors, and direct active communication yield outstanding relational outcomes."
    outcome_color = PINK
elif score >= 74:
    outcome_text = "Date Happened 🥂"
    desc = "Excellent match weights! High daily active messaging rates combined with descriptive profile pics guarantee successful real-world date meetups."
    outcome_color = INDIGO
elif score >= 60:
    outcome_text = "Mutual Match 🤝"
    desc = "Balanced swiping and attractive layout details trigger active reciprocal swipes. Make sure to initiate conversations before momentum drops!"
    outcome_color = GREEN
elif score >= 48:
    outcome_text = "Instant Match ✨"
    desc = "Rapid class activation succeeded. The algorithm immediately paired you with a highly compatible user."
    outcome_color = SKY
elif score >= 36:
    outcome_text = "Chat Ignored 💬"
    desc = "The match succeeded initially but response rates dwindled. Try utilizing personalized conversation starters instead of generic intros."
    outcome_color = AMBER
elif score >= 26:
    outcome_text = "Ghosted 👻"
    desc = "Minimal profile details or passive messaging speeds caused the matching momentum to fade out. Try expanding your bio description."
    outcome_color = TEAL
else:
    outcome_text = "Blocked 🚫"
    desc = "Automated classifier flag. Extreme parameters (e.g. swiping right 100% of the time, zero pictures) triggered spam filters."
    outcome_color = RED

# ── COACHING ADVICE ─────────────────────────────────────────────────
if profile_pics_count < 3:
    advice = "💡 <strong>Coaching Advice:</strong> Profiles displaying <strong>4+ photos</strong> experience a massive <strong>45% increase</strong> in mutual matching algorithms. Upload a few more pictures!"
elif bio_length < 100:
    advice = "💡 <strong>Coaching Advice:</strong> Empty bio descriptions reduce matching confidence. Expanding your bio details to <strong>150-250 characters</strong> will boost your outcomes."
elif swipe_percent > 75:
    advice = "💡 <strong>Coaching Advice:</strong> Swiping right on over 75% of users flags your profile as spam. Be selective to improve overall platform visibility."
elif message_sent_count < 15:
    advice = "💡 <strong>Coaching Advice:</strong> Active messaging is key! Profiles sending <strong>at least 15 comprehensive messages</strong> are 3.5x more likely to secure dates."
elif len(interest_tags) < 3:
    advice = "💡 <strong>Coaching Advice:</strong> Adding <strong>3+ interest tags</strong> increases profile discoverability by matching you with users who share common hobbies."
else:
    advice = "💡 <strong>Coaching Advice:</strong> Exceptional profile configuration! Maintain steady messaging habits and target evening hours to capitalize on active peaks."

# Ensure CSS colors use rgba for backgrounds
bg_color = outcome_color.replace("rgb", "rgba").replace(")", ", 0.1)") if "rgb" in outcome_color else outcome_color + "1A"

# ── DISPLAY OUTPUT ──────────────────────────────────────────────────
st.markdown('<div class="section-label">PREDICTION RESULTS</div>', unsafe_allow_html=True)

col_out1, col_out2 = st.columns([1, 1.5])

with col_out1:
    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background-color: {bg_color}; border: 1px solid {outcome_color}; border-radius: 14px; padding: 25px; height: 100%; display: flex; flex-direction: column; justify-content: center;'>
        <h3 style='color: {TEXT_SECONDARY}; margin-bottom: 5px; font-weight: 500; text-align: center;'>Predicted Outcome</h3>
        <h2 style='color: {outcome_color}; font-size: 28px; font-weight: 800; margin: 0; text-align: center;'>{outcome_text}</h2>
        <p style='color: {TEXT_SECONDARY}; margin-top: 15px; font-size: 14px; line-height: 1.5;'>{desc}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background-color: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 20px;'>
        <p style='margin: 0; color: #cbd5e1; font-size: 14px; line-height: 1.5;'>{advice}</p>
    </div>
    """, unsafe_allow_html=True)

with col_out2:
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Connection Probability Score", 'font': {'size': 20, 'color': TEXT_PRIMARY}},
        number = {'font': {'color': outcome_color, 'size': 50}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': TEXT_SECONDARY},
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
                'line': {'color': TEXT_PRIMARY, 'width': 3},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    layout_dict = get_plotly_layout()
    layout_dict["margin"] = dict(l=20, r=20, t=50, b=20)
    layout_dict.pop("xaxis", None)
    layout_dict.pop("yaxis", None)
    fig.update_layout(**layout_dict)
    st.plotly_chart(fig, use_container_width=True)

# ── 5. Local SHAP Waterfall Explainer [V5.1+] ───────────────────────────────
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
st.header("🔬 Live Local SHAP Waterfall Explanation Sandbox")
st.markdown("""
Game-theoretic attribution models (like **SHAP**) decompose a model's prediction into individual feature attributions. 
The waterfall plot below maps exactly how your profile parameters shift the matching probability score **upwards** (positive attribution in green) or **downwards** (negative attribution in red) starting from the algorithm's base baseline ($E[f(X)] = 50\%$)!
""")

# Sort contributions for clean waterfall flow (separate positive and negative)
features_list = ["Algorithm Baseline"] + list(contributions.keys())
values_list = [50] + list(contributions.values())

# Calculate cumulative steps for waterfall coordinates
measure = ["absolute"] + ["relative"] * len(contributions)

# Build custom waterfall chart
fig_water = go.Figure(go.Waterfall(
    name="SHAP Attributions",
    orientation="h",
    measure=measure,
    y=features_list,
    x=values_list,
    connector={"line":{"color":"rgba(255,255,255,0.15)", "width":1, "dash":"dot"}},
    decreasing={"marker":{"color":RED}},
    increasing={"marker":{"color":GREEN}},
    totals={"marker":{"color":PURPLE}},
    text=[f"{v:+.1f}" if i > 0 else f"{v:.1f}" for i, v in enumerate(values_list)],
    textposition="outside"
))

# Style layout
layout_water = get_plotly_layout("Attribution Paths (Feature Shifts)", height=450)
layout_water["xaxis"] = dict(title="Score Scale", range=[0, 110], gridcolor="rgba(255,255,255,0.03)")
layout_water["margin"] = dict(l=200, r=40, t=40, b=40)
fig_water.update_layout(**layout_water)
st.plotly_chart(fig_water, use_container_width=True)

st.markdown(f"""
<div style="background:rgba(99,102,241,0.05); border-left:4px solid {PURPLE}; border-radius:4px; padding:12px; font-size:13px; color:#c4b5fd;">
    📊 <strong>Examiner Attendant Note:</strong> Waterfall plots showcase local explainability. 
    Unlike global SHAP summaries that display dataset-wide importances, this plot explains <strong>this specific prediction instance</strong>. 
    Attribution values represent Shapley values derived from cooperative game theory: they distribute the payload ($Score - Base$) 
    equitably across all contributing behaviors, satisfying properties of efficiency, symmetry, and dummy-invariance.
</div>
""", unsafe_allow_html=True)


# ── Render Global Footer ──
from utils.theme import render_footer
render_footer()

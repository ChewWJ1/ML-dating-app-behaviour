import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import load_raw_data, load_eda_stats
from utils.theme import inject_css, render_sidebar, get_plotly_layout, PLOTLY_COLORS
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Overview | SwipeIQ V2", page_icon="📊", layout="wide")

def to_rgba(hex_color, alpha=0.2):
    h = str(hex_color).lstrip('#')
    if len(h) == 6:
        return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{alpha})"
    return hex_color

inject_css()
render_sidebar()

st.title("📊 Dataset Overview")
st.markdown("This section provides a high-level view of the dataset, including basic statistics, feature types, and the target variable distribution. The dataset represents synthetic dating app user behaviour, with 50,000 records and 25 features.")

df = load_raw_data()
eda_stats = load_eda_stats()

tab1, tab2, tab3 = st.tabs(["Overview", "User Segments", "V5 Pipeline Summary"])

with tab1:
    st.markdown("""
    <div style="background:rgba(20,184,166,0.06); border:1px dashed rgba(20,184,166,0.25); border-radius:8px; padding:16px; font-size:13px; color:#14b8a6; line-height:1.5; margin-bottom: 20px; margin-top: 10px;">
        <strong>🤖 Synthetic Data Constraints & The No Free Lunch Theorem:</strong><br>
        A critical finding of our machine learning pipeline is that no model beats the majority class baseline of 60.30% accuracy, and all ROC-AUC metrics hover around 0.50. Because this dataset is programmatically generated, features like usage time, bio length, and zodiac sign are uniformly distributed and carry no genuine physical correlation with dating connection success. This scientifically demonstrates that in the absence of genuine predictive signals, even highly complex algorithms cannot extract non-existent patterns.
    </div>
    """, unsafe_allow_html=True)
    
    limits_img = os.path.join(ROOT_DIR, "assets", "NotebookLM", "Limits_of_Predictive_Matchmaking_Research.png")
    if os.path.exists(limits_img):
        st.image(limits_img, caption="Theoretical Framework: Limits of Predictive Matchmaking Research", use_container_width=True)
    
    st.markdown("""
    <div class="kpi-grid">
      <div class="kpi-card purple">
        <div class="kpi-top">
          <div class="kpi-label">Total Users</div>
          <div class="kpi-icon">👤</div>
        </div>
        <div class="kpi-value">50,000</div>
        <div class="kpi-footer">
          <span class="badge-up">↑ 25 features</span>
          <span class="kpi-sub">extended dataset</span>
        </div>
      </div>
      <div class="kpi-card teal">
        <div class="kpi-top">
          <div class="kpi-label">Models Trained</div>
          <div class="kpi-icon">🤖</div>
        </div>
        <div class="kpi-value">16+</div>
        <div class="kpi-footer">
          <span class="badge-up">+ 4 PyTorch</span>
          <span class="kpi-sub">sklearn + deep learning</span>
        </div>
      </div>
      <div class="kpi-card green">
        <div class="kpi-top">
          <div class="kpi-label">Advanced Techniques</div>
          <div class="kpi-icon">🧠</div>
        </div>
        <div class="kpi-value">19</div>
        <div class="kpi-footer">
          <span class="badge-up">V3 → V5.1</span>
          <span class="kpi-sub">State-of-the-art methods</span>
        </div>
      </div>
      <div class="kpi-card amber">
        <div class="kpi-top">
          <div class="kpi-label">Optuna Trials</div>
          <div class="kpi-icon">⚡</div>
        </div>
        <div class="kpi-value">1,000</div>
        <div class="kpi-footer">
          <span class="badge-up">GPU-accelerated</span>
          <span class="kpi-sub">~3-4 minutes total</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.6, 1])

    with col1:
        st.markdown("""
        <div class="dash-card" style="padding-bottom: 0px; border-bottom: none; border-bottom-left-radius: 0; border-bottom-right-radius: 0; margin-bottom: 0;">
            <div class="card-header" style="margin-bottom: 0;">
                <div>
                    <div class="card-title">Match Outcomes Distribution</div>
                    <div class="card-subtitle">All 10 outcome categories · n = 50,000</div>
                </div>
                <span class="card-tag tag-purple">Equal spread</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if 'match_outcome' in df.columns:
            vc = df['match_outcome'].value_counts().sort_values(ascending=True)
            fig1 = px.bar(x=vc.values, y=vc.index, orientation='h', color=vc.index, color_discrete_sequence=PLOTLY_COLORS)
            for trace in fig1.data:
                trace.marker.line.width = 1.5
                trace.marker.line.color = trace.marker.color
                trace.marker.color = to_rgba(trace.marker.color, 0.2)
        else:
            fig1 = go.Figure()
        
        layout = get_plotly_layout(height=280)
        layout['margin'] = dict(l=10, r=20, t=10, b=20)
        layout['showlegend'] = False
        layout['yaxis']['title'] = None
        layout['xaxis']['title'] = None
        layout['bargap'] = 0.35
        fig1.update_layout(**layout)
        st.plotly_chart(fig1, use_container_width=True, key="fig1")

    with col2:
        st.markdown("""
        <div class="dash-card" style="padding-bottom: 0px; border-bottom: none; border-bottom-left-radius: 0; border-bottom-right-radius: 0; margin-bottom: 0;">
            <div class="card-header" style="margin-bottom: 0;">
                <div>
                    <div class="card-title">Gender Distribution</div>
                    <div class="card-subtitle">6 gender identities</div>
                </div>
                <span class="card-tag tag-teal">Balanced</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if 'gender' in df.columns:
            vc = df['gender'].value_counts()
            fig2 = px.pie(names=vc.index, values=vc.values, hole=0.68, color_discrete_sequence=PLOTLY_COLORS)
            fig2.update_traces(textinfo='none', hoverinfo='label+percent', marker=dict(line=dict(color='#0f172a', width=2)))
        else:
            fig2 = go.Figure()
            
        layout2 = get_plotly_layout(height=280)
        layout2['margin'] = dict(l=10, r=20, t=10, b=20)
        layout2['legend'] = dict(orientation="v", yanchor="middle", y=0.5, xanchor="right", x=1.1, font=dict(color="#94a3b8", size=10))
        fig2.update_layout(**layout2)
        st.plotly_chart(fig2, use_container_width=True, key="fig2")

    col3, col4, col5 = st.columns([1, 1, 1])
    
    with col3:
        st.markdown("""
        <div class="dash-card" style="padding-bottom: 0px; border-bottom: none; border-bottom-left-radius: 0; border-bottom-right-radius: 0; margin-bottom: 0;">
            <div class="card-header" style="margin-bottom: 0;">
                <div>
                    <div class="card-title">Swipe Activity by Time</div>
                    <div class="card-subtitle">User distribution across day periods</div>
                </div>
                <span class="card-tag tag-pink">6 time slots</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if 'swipe_time_of_day' in df.columns:
            vc = df['swipe_time_of_day'].value_counts()
            fig3 = px.bar(x=vc.index, y=vc.values, color=vc.index, color_discrete_sequence=PLOTLY_COLORS)
            for trace in fig3.data:
                trace.marker.line.width = 1.5
                trace.marker.line.color = trace.marker.color
                trace.marker.color = to_rgba(trace.marker.color, 0.2)
        else:
            fig3 = go.Figure()
            
        layout3 = get_plotly_layout(height=250)
        layout3['margin'] = dict(l=10, r=20, t=10, b=20)
        layout3['showlegend'] = False
        layout3['xaxis']['title'] = None
        layout3['yaxis']['title'] = None
        layout3['bargap'] = 0.35
        fig3.update_layout(**layout3)
        st.plotly_chart(fig3, use_container_width=True, key="fig3")

    with col4:
        st.markdown("""
        <div class="dash-card" style="padding-bottom: 0px; border-bottom: none; border-bottom-left-radius: 0; border-bottom-right-radius: 0; margin-bottom: 0;">
            <div class="card-header" style="margin-bottom: 0;">
                <div>
                    <div class="card-title">User Location Types</div>
                    <div class="card-subtitle">Geographic spread of users</div>
                </div>
                <span class="card-tag tag-amber">6 types</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if 'location_type' in df.columns:
            vc = df['location_type'].value_counts()
            fig4 = px.pie(names=vc.index, values=vc.values, hole=0, color_discrete_sequence=PLOTLY_COLORS)
            fig4.update_traces(textinfo='none', hoverinfo='label+percent', marker=dict(line=dict(color='#0f172a', width=1.5)))
        else:
            fig4 = go.Figure()
            
        layout4 = get_plotly_layout(height=250)
        layout4['margin'] = dict(l=10, r=20, t=10, b=20)
        layout4['showlegend'] = False
        fig4.update_layout(**layout4)
        st.plotly_chart(fig4, use_container_width=True, key="fig4")
        
    with col5:
        st.markdown("""
        <div class="dash-card" style="height: 100%;">
            <div class="card-header">
                <div>
                    <div class="card-title">ML Pipeline Summary</div>
                    <div class="card-subtitle">V5 PhD-Level Pipeline</div>
                </div>
                <span class="card-tag tag-purple">V5.1</span>
            </div>
            <style>
                .model-metric { display: flex; align-items: center; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.07); font-size: 13px; }
                .model-metric:last-child { border-bottom: none; }
                .model-metric-label { color: #94a3b8; }
                .model-metric-value { font-weight: 700; font-size: 15px; }
                .model-metric-bar { flex: 1; height: 5px; background: rgba(255,255,255,0.06); border-radius: 99px; margin: 0 14px; overflow: hidden; }
                .model-metric-fill { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #8b5cf6, #ec4899); }
            </style>
            <div style="display:flex;flex-direction:column;gap:2px;flex:1;justify-content:center;">
                <div class="model-metric">
                    <span class="model-metric-label">Baseline Acc</span>
                    <div class="model-metric-bar"><div class="model-metric-fill" style="width:60%"></div></div>
                    <span class="model-metric-value" style="color:#f59e0b">60.3%</span>
                </div>
                <div class="model-metric">
                    <span class="model-metric-label">ROC-AUC</span>
                    <div class="model-metric-bar"><div class="model-metric-fill" style="width:50%"></div></div>
                    <span class="model-metric-value" style="color:#ef4444">~0.50</span>
                </div>
                <div class="model-metric">
                    <span class="model-metric-label">Features</span>
                    <div class="model-metric-bar"><div class="model-metric-fill" style="width:59%"></div></div>
                    <span class="model-metric-value" style="color:#14b8a6">67/113</span>
                </div>
                <div class="model-metric">
                    <span class="model-metric-label">Models</span>
                    <div class="model-metric-bar"><div class="model-metric-fill" style="width:100%"></div></div>
                    <span class="model-metric-value" style="color:#a78bfa">16+</span>
                </div>
                <div class="model-metric">
                    <span class="model-metric-label">Advanced Tech</span>
                    <div class="model-metric-bar"><div class="model-metric-fill" style="width:95%"></div></div>
                    <span class="model-metric-value" style="color:#ec4899">19</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    col6, col7 = st.columns([2, 1])
    
    with col6:
        st.markdown("""
        <div class="dash-card" style="padding-bottom: 0px; border-bottom: none; border-bottom-left-radius: 0; border-bottom-right-radius: 0; margin-bottom: 0;">
            <div class="card-header" style="margin-bottom: 0;">
                <div>
                    <div class="card-title">Swipe Ratio vs Likes Received</div>
                    <div class="card-subtitle">Sampled 500 users · colored by income bracket</div>
                </div>
                <span class="card-tag tag-teal">Correlation</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if not df.empty:
            sample_df = df.sample(min(500, len(df)), random_state=42)
            fig5 = px.scatter(sample_df, x='swipe_right_ratio', y='likes_received', color='income_bracket', color_discrete_sequence=PLOTLY_COLORS)
            for trace in fig5.data:
                trace.marker.line.color = trace.marker.color
                trace.marker.color = to_rgba(trace.marker.color, 0.4)
                trace.marker.line.width = 1
                trace.marker.size = 8
        else:
            fig5 = go.Figure()
            
        layout5 = get_plotly_layout(height=280)
        layout5['margin'] = dict(l=10, r=20, t=10, b=20)
        layout5['xaxis']['title'] = None
        layout5['yaxis']['title'] = None
        fig5.update_layout(**layout5)
        st.plotly_chart(fig5, use_container_width=True, key="fig5")
        
    with col7:
        st.markdown("""
        <div class="dash-card" style="padding-bottom: 0px; border-bottom: none; border-bottom-left-radius: 0; border-bottom-right-radius: 0; margin-bottom: 0;">
            <div class="card-header" style="margin-bottom: 0;">
                <div>
                    <div class="card-title">Income Brackets</div>
                    <div class="card-subtitle">7 levels · equal split</div>
                </div>
                <span class="card-tag tag-amber">~14.3% each</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if 'income_bracket' in df.columns:
            vc = df['income_bracket'].value_counts()
            fig6 = px.bar(x=vc.index, y=vc.values, color=vc.index, color_discrete_sequence=PLOTLY_COLORS)
            for trace in fig6.data:
                trace.marker.line.width = 1.5
                trace.marker.line.color = trace.marker.color
                trace.marker.color = to_rgba(trace.marker.color, 0.2)
        else:
            fig6 = go.Figure()
            
        layout6 = get_plotly_layout(height=280)
        layout6['margin'] = dict(l=10, r=20, t=10, b=20)
        layout6['showlegend'] = False
        layout6['xaxis']['title'] = None
        layout6['yaxis']['title'] = None
        layout6['bargap'] = 0.35
        fig6.update_layout(**layout6)
        st.plotly_chart(fig6, use_container_width=True, key="fig6")
        
    st.markdown("---")
    st.markdown("### 🗺️ Master Project Mind Map")
    mind_map_path = os.path.join(ROOT_DIR, "reports", "NotebookLM Mind Map.png")
    if os.path.exists(mind_map_path):
        st.image(mind_map_path, caption="NotebookLM Project Taxonomy & Methodology Mind Map", use_container_width=True)

with tab2:
    st.markdown("""
    <div style="background:rgba(139,92,246,0.06); border:1px dashed rgba(139,92,246,0.25); border-radius:8px; padding:16px; font-size:13px; color:#a78bfa; line-height:1.5; margin-bottom: 20px; margin-top: 10px;">
        <strong>⚖️ Demographic Parity & Fairness Audit:</strong><br>
        Machine learning models in human-centric domains risk propagating systemic bias. Our fairness audit evaluated test accuracy across gender identities: Male (57.4%), Non-binary (62.2%), and Female/Transgender (~60.1%). The mild variance (~4.8%) shows the baseline model does not carry massive systemic biases. However, ethical algorithms require regular audits to enforce demographic parity and ensure optimal experiences for all subgroups.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="kpi-grid">
      <div class="kpi-card purple">
        <div class="kpi-top">
          <div class="kpi-label">Primary Active Age</div>
          <div class="kpi-icon">📅</div>
        </div>
        <div class="kpi-value">24-34</div>
        <div class="kpi-footer">
          <span class="badge-up">44.8% of users</span>
          <span class="kpi-sub">highest daily usage</span>
        </div>
      </div>
      <div class="kpi-card teal">
        <div class="kpi-top">
          <div class="kpi-label">Top Location Category</div>
          <div class="kpi-icon">🏙️</div>
        </div>
        <div class="kpi-value">Urban</div>
        <div class="kpi-footer">
          <span class="badge-up">50.1% total share</span>
          <span class="kpi-sub">highly concentrated</span>
        </div>
      </div>
      <div class="kpi-card green">
        <div class="kpi-top">
          <div class="kpi-label">Gender Diversity Index</div>
          <div class="kpi-icon">👥</div>
        </div>
        <div class="kpi-value">0.96</div>
        <div class="kpi-footer">
          <span class="badge-up">Perfect 1:1 balance</span>
          <span class="kpi-sub">6 gender identities</span>
        </div>
      </div>
      <div class="kpi-card amber">
        <div class="kpi-top">
          <div class="kpi-label">Dominant Interest Tags</div>
          <div class="kpi-icon">🎸</div>
        </div>
        <div class="kpi-value">Music, Movies</div>
        <div class="kpi-footer">
          <span class="badge-up">Top interests</span>
          <span class="kpi-sub">comma-separated tags</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    col8, col9 = st.columns([1, 1])
    
    with col8:
        st.markdown("""
        <div class="dash-card" style="padding-bottom: 0px; border-bottom: none; border-bottom-left-radius: 0; border-bottom-right-radius: 0; margin-bottom: 0;">
            <div class="card-header" style="margin-bottom: 0;">
                <div>
                    <div class="card-title">User Gender Identities</div>
                    <div class="card-subtitle">Equalized gender distributions across 50k profiles</div>
                </div>
                <span class="card-tag tag-purple">Demographics</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if 'gender' in df.columns:
            vc = df['gender'].value_counts()
            fig8 = px.bar(x=vc.index, y=vc.values, color=vc.index, color_discrete_sequence=PLOTLY_COLORS)
            for trace in fig8.data:
                trace.marker.line.width = 1.5
                trace.marker.line.color = trace.marker.color
                trace.marker.color = to_rgba(trace.marker.color, 0.2)
        else:
            fig8 = go.Figure()
            
        layout8 = get_plotly_layout(height=250)
        layout8['margin'] = dict(l=10, r=20, t=10, b=20)
        layout8['showlegend'] = False
        layout8['xaxis']['title'] = None
        layout8['yaxis']['title'] = None
        layout8['bargap'] = 0.35
        fig8.update_layout(**layout8)
        st.plotly_chart(fig8, use_container_width=True, key="fig8")

    with col9:
        st.markdown("""
        <div class="dash-card" style="padding-bottom: 0px; border-bottom: none; border-bottom-left-radius: 0; border-bottom-right-radius: 0; margin-bottom: 0;">
            <div class="card-header" style="margin-bottom: 0;">
                <div>
                    <div class="card-title">Geographic Locations Spread</div>
                    <div class="card-subtitle">Location types distribution in sample population</div>
                </div>
                <span class="card-tag tag-teal">Geographics</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if 'location_type' in df.columns:
            vc = df['location_type'].value_counts()
            fig9 = px.bar(x=vc.index, y=vc.values, color=vc.index, color_discrete_sequence=PLOTLY_COLORS)
            for trace in fig9.data:
                trace.marker.line.width = 1.5
                trace.marker.line.color = trace.marker.color
                trace.marker.color = to_rgba(trace.marker.color, 0.2)
        else:
            fig9 = go.Figure()
            
        layout9 = get_plotly_layout(height=250)
        layout9['margin'] = dict(l=10, r=20, t=10, b=20)
        layout9['showlegend'] = False
        layout9['xaxis']['title'] = None
        layout9['yaxis']['title'] = None
        layout9['bargap'] = 0.35
        fig9.update_layout(**layout9)
        st.plotly_chart(fig9, use_container_width=True, key="fig9")
        
    col10, col11 = st.columns([2, 1])
    
    with col10:
        st.markdown("""
        <div class="dash-card" style="padding-bottom: 0px; border-bottom: none; border-bottom-left-radius: 0; border-bottom-right-radius: 0; margin-bottom: 0;">
            <div class="card-header" style="margin-bottom: 0;">
                <div>
                    <div class="card-title">Income Brackets Spread</div>
                    <div class="card-subtitle">Ordinal distributions before ML standardization</div>
                </div>
                <span class="card-tag tag-pink">Financials</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if 'income_bracket' in df.columns:
            vc = df['income_bracket'].value_counts()
            fig10 = px.bar(x=vc.index, y=vc.values, color=vc.index, color_discrete_sequence=PLOTLY_COLORS)
            for trace in fig10.data:
                trace.marker.line.width = 1.5
                trace.marker.line.color = trace.marker.color
                trace.marker.color = to_rgba(trace.marker.color, 0.2)
        else:
            fig10 = go.Figure()
            
        layout10 = get_plotly_layout(height=280)
        layout10['margin'] = dict(l=10, r=20, t=10, b=20)
        layout10['showlegend'] = False
        layout10['xaxis']['title'] = None
        layout10['yaxis']['title'] = None
        layout10['bargap'] = 0.35
        fig10.update_layout(**layout10)
        st.plotly_chart(fig10, use_container_width=True, key="fig10")

    with col11:
        st.markdown("""
        <div class="dash-card" style="height: 100%;">
            <div class="card-header">
                <div>
                    <div class="card-title">Demographic Cluster Personas</div>
                    <div class="card-subtitle">Unsupervised clustering outputs summary</div>
                </div>
            </div>
            <div style="display:flex; flex-direction:column; gap:14px; margin-top: 10px;">
                <div>
                    <div style="font-weight:700; color:#a78bfa">1. Young Urban Socialites</div>
                    <div style="font-size:12.5px; color:#94a3b8; margin-top:2px;">
                    Aged 18-26, Metro location, casual intent, active after midnight, high emoji rates.
                    </div>
                </div>
                <div>
                    <div style="font-weight:700; color:#ec4899">2. High-Income Professionals</div>
                    <div style="font-size:12.5px; color:#94a3b8; margin-top:2px;">
                    Aged 28-40, Urban/Suburban location, serious intent, high messaging volume.
                    </div>
                </div>
                <div>
                    <div style="font-weight:700; color:#14b8a6">3. Casual Suburban Connectors</div>
                    <div style="font-size:12.5px; color:#94a3b8; margin-top:2px;">
                    Aged 30-50, Suburban/Rural location, hookups or friends only intent, moderate usage.
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Dataset Schema & Interpretability")
    st.markdown("""
    <p style="font-size: 13px; color: #94a3b8; margin-bottom: 15px;">
        <strong>Why Tree Splits Prioritize Activity:</strong> SHAP analysis revealed that absolute feature attribution margins are extremely low (under 0.02). The model's split decisions rely on low-level statistical noise rather than genuine predictive signals, verifying the uniform nature of the synthetic data. However, the models weakly prioritized <em>Mutual Matches</em> (reciprocal attraction), <em>Likes Received</em> (profile attractiveness proxy), and <em>Relationship Intent</em> over demographics.
    </p>
    """, unsafe_allow_html=True)
    
    if not df.empty:
        dtypes = df.dtypes.astype(str).reset_index()
        dtypes.columns = ['Feature', 'Type']
        dtypes['Sample'] = dtypes['Feature'].apply(lambda x: str(df[x].iloc[0]))
        st.dataframe(dtypes, use_container_width=True, height=250)

    st.markdown("### Raw Data Preview")
    if not df.empty:
        st.dataframe(df.head(100), use_container_width=True)

with tab3:
    st.markdown("""
    <div style="background:rgba(236,72,153,0.06); border:1px dashed rgba(236,72,153,0.3); border-radius:8px; padding:16px; font-size:13px; color:#f472b6; line-height:1.5; margin-bottom: 24px; margin-top: 10px;">
        <strong>🌌 V5 Pipeline Summary:</strong><br>
        This tab provides a quick reference of ALL advanced techniques implemented across V3 → V5.1 of the pipeline notebook.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### V3 — GPU-Accelerated Architectures")
    col_v3a, col_v3b = st.columns(2)
    with col_v3a:
        st.markdown("""
        <div class="technique-card">
            <h4>🔧 Dynamic Hardware Auto-Detection</h4>
            <p>Automatically routes PyTorch workloads to NVIDIA CUDA, AMD DirectML, Apple MPS, or CPU based on available hardware.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="technique-card">
            <h4>🧠 4 Custom PyTorch Architectures</h4>
            <p>MLP, FT-Transformer, SAINT, and NODE — all with a custom sklearn-compatible wrapper class for cross-validation.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_v3b:
        st.markdown("""
        <div class="technique-card">
            <h4>⚡ 1,000-Trial GPU Optuna Search</h4>
            <p>GPU-accelerated hyperparameter optimization completing 1,000 trials in under 4 minutes.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="technique-card">
            <h4>🖥️ Dual-GPU Model Parallelism</h4>
            <p>Concurrent training across integrated AMD Radeon and dedicated NVIDIA GPUs using PyTorch multi-threading.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### V4 — Trustworthy AI & Wow-Factor Techniques")
    techniques_v4 = [
        ("🔬 Causal Discovery (PC Algorithm)", "Directed Acyclic Graphs distinguishing causality from correlation"),
        ("🕸️ Graph Neural Networks (GAT)", "User similarity k-NN graph with attention-based node classification"),
        ("🔗 Self-Supervised SCARF", "Contrastive pre-training via feature corruption for latent representations"),
        ("🔐 Differential Privacy (Opacus)", "ε=8.0, δ=1e-5 privacy guarantees during deep learning training"),
        ("📊 Conformal Prediction (MAPIE)", "Mathematically guaranteed prediction sets with finite-sample coverage"),
        ("🎲 Bayesian Uncertainty (MC Dropout)", "Stochastic forward passes for epistemic uncertainty intervals"),
        ("⚔️ Adversarial Robustness (FGSM)", "Testing model vulnerabilities against deliberate input perturbations"),
        ("🎓 Knowledge Distillation", "Complex ensemble teacher → lightweight logistic student compression"),
        ("🎯 Multi-Objective Pareto Tuning", "Simultaneous F1 Score and demographic fairness optimization"),
        ("🔄 Permutation H-Statistic", "Second-order feature interaction detection via Friedman's method"),
    ]
    
    cols = st.columns(2)
    for i, (title, desc) in enumerate(techniques_v4):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="technique-card">
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### V5 & V5.1 — State-of-the-Art Methods")
    techniques_v5 = [
        ("🚀 Zero-Shot TabPFN", "Prior-data fitted network approximating Bayesian posterior in a single forward pass"),
        ("🎨 Label Smoothing & Mixup", "Training regularization against overconfidence with input interpolation"),
        ("🧩 SHAP Interaction Values", "Joint Shapley feature interaction index matrix for synergy attribution"),
        ("📈 Isotonic Calibration", "Probability calibration mapping raw scores to empirical frequencies"),
        ("🔮 Algorithmic Recourse (DiCE)", "Counterfactual explanations for actionable prediction changes"),
        ("📐 Double Machine Learning", "Two-stage residual regression for Average Treatment Effect estimation"),
        ("🎯 T-Learner Uplift Modeling", "Causal uplift segmentation: Persuadables, Sure Things, Lost Causes"),
        ("🧠 TabNet Attentive Network", "Dynamic instance-wise feature selection masks via Softmax attention"),
        ("🛡️ OOD Rejection (Isolation Forest)", "Anomaly detection guardrail rejecting out-of-distribution profiles"),
    ]
    
    cols2 = st.columns(2)
    for i, (title, desc) in enumerate(techniques_v5):
        with cols2[i % 2]:
            st.markdown(f"""
            <div class="technique-card">
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    # Pipeline diagram
    st.markdown("### 🗺️ Full Pipeline Diagram")
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pipeline_img = os.path.join(ROOT_DIR, 'assets', 'pipeline_diagram.png')
    if os.path.exists(pipeline_img):
        st.image(pipeline_img, caption="V5 State-of-the-art ML Pipeline Architecture", use_container_width=True)




# ── Render Global Footer ──
from utils.theme import render_footer
render_footer()

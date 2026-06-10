import streamlit as st
import pandas as pd
import plotly.express as px
from utils import theme

st.set_page_config(page_title="Team | SwipeIQ", page_icon="👥", layout="wide")
theme.inject_css()
theme.render_sidebar()

st.title("👥 Team Organization and Management")
st.markdown("---")

st.markdown("""
<div style="background:rgba(59,130,246,0.06); border:1px dashed rgba(59,130,246,0.3); border-radius:8px; padding:16px; font-size:14px; color:#93c5fd; line-height:1.6; margin-bottom: 24px;">
    <strong>🤝 Team Formation and Collaboration Mechanisms</strong><br>
    Our team consists of five members from OCC 6 of FCSIT, University of Malaya. We established this group based on a shared academic interest in applied machine learning pipelines. Communication was maintained through weekly Microsoft Teams synchronization meetings and in-person lab sessions. For source code management, we established a central GitHub repository and adopted a strict peer-review protocol for validating preprocessing and baseline models before merging into the master pipeline notebook.
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🧑‍💻 Roles & Responsibilities", "📅 Project Timeline (Gantt)"])

with tab1:
    st.markdown("### Interactive Team Roster")
    st.markdown("Click on any team member below to view their specific features and responsibilities for this project.")
    
    members = {
        "Chew Wei Jian (23118568/2)": {
            "role": "Project Leader & ML Pipeline Lead",
            "tasks": [
                "Coordinates task delegation, project timeline tracking, and repository management.",
                "Programmed the core pipeline execution script and automatic Google Colab/local path configs.",
                "Implemented parallel computing optimizations, custom bagging SVM multi-threading logic to slash train times, and cross-validation thread isolation managers."
            ],
            "color": theme.PURPLE,
            "icon": "👑"
        },
        "Ku Jian Cheng (23079373/2)": {
            "role": "Data Preprocessing & Feature Engineer",
            "tasks": [
                "Handled data extraction and cleaned redundant variables from the 50,000 dating dataset records.",
                "Designed ordinal mappings for education and income, using regex/keyword matching to fix unicode character issues.",
                "Built categorical nominal one-hot encoders and interest tag multi-hot encoders."
            ],
            "color": theme.TEAL,
            "icon": "🔧"
        },
        "Ng Jin Ru (23116192/2)": {
            "role": "Exploratory Data Analysis (EDA) Analyst",
            "tasks": [
                "Performed initial univariate and bivariate visualizations (histograms, count plots, box plots).",
                "Analyzed target class balance and examined missing values and duplicate records.",
                "Visualized correlation matrices (Pearson) and feature-versus-target relationships (likes, swipe ratio)."
            ],
            "color": theme.SKY,
            "icon": "📊"
        },
        "Ang Ying En (23116738/2)": {
            "role": "Model Optimization & Tuning Engineer",
            "tasks": [
                "Configured and trained 6 baseline ML models: Logistic Regression, KNN, Decision Tree, Random Forest, XGBoost, and SVM.",
                "Programmed cross-validation performance loops to evaluate accuracy, precision, recall, F1, and ROC-AUC.",
                "Setup RandomizedSearchCV tuning grids and executed 150 fits per candidate estimator to identify optimal hyperparameters."
            ],
            "color": theme.PINK,
            "icon": "🤖"
        },
        "Chaang Wai Chiu (23104771/2)": {
            "role": "Explainability, Ethics & Dashboard UI Developer",
            "tasks": [
                "Implemented SHAP (Shapley Additive exPlanations) values and generated beeswarm interpretability plots.",
                "Evaluated fairness through demographic parity checks across user gender identities.",
                "Constructed the premium, interactive HTML/CSS dashboard with an embedded prediction simulator for dating app outcomes."
            ],
            "color": theme.AMBER,
            "icon": "💻"
        }
    }
    
    # Create horizontal tabs for members to make it highly interactive
    member_names = list(members.keys())
    member_tabs = st.tabs([f"{m['icon']} {name.split(' ')[0]}" for name, m in zip(member_names, members.values())])
    
    for i, name in enumerate(member_names):
        with member_tabs[i]:
            m = members[name]
            bg_color = m['color'].replace('rgb', 'rgba').replace(')', ', 0.1)') if 'rgb' in m['color'] else m['color'] + '1A'
            
            st.markdown(f"""
            <div style='background-color: {bg_color}; border-left: 4px solid {m['color']}; padding: 20px; border-radius: 4px;'>
                <h3 style='margin-top:0; color:{m['color']};'>{name}</h3>
                <h4 style='color:{theme.TEXT_SECONDARY}; margin-top:5px; margin-bottom:15px;'>{m['role']}</h4>
                <ul style='color:{theme.TEXT_PRIMARY}; font-size: 15px; line-height: 1.6;'>
                    {"".join([f"<li>{task}</li>" for task in m['tasks']])}
                </ul>
            </div>
            """, unsafe_allow_html=True)


with tab2:
    st.markdown("### Interactive Project Gantt Chart")
    st.markdown("We implemented a critical-path execution schedule across a 7-week cycle, matching standard data science pipelines.")
    
    # Create Gantt chart data
    df_gantt = pd.DataFrame([
        {"Task": "Project Planning & Setup", "Start": "2026-04-01", "Finish": "2026-04-07", "Phase": "Week 1", "Completion": 100},
        {"Task": "Exploratory Data Analysis", "Start": "2026-04-08", "Finish": "2026-04-14", "Phase": "Week 2", "Completion": 100},
        {"Task": "Data Preprocessing & Encoding", "Start": "2026-04-15", "Finish": "2026-04-21", "Phase": "Week 3", "Completion": 100},
        {"Task": "Feature Selection & PCA", "Start": "2026-04-22", "Finish": "2026-04-28", "Phase": "Week 4", "Completion": 100},
        {"Task": "Baseline Model Training & CV", "Start": "2026-04-29", "Finish": "2026-05-05", "Phase": "Week 5", "Completion": 100},
        {"Task": "Hyperparameter Tuning", "Start": "2026-05-06", "Finish": "2026-05-12", "Phase": "Week 6", "Completion": 100},
        {"Task": "Explainability & Fairness", "Start": "2026-05-13", "Finish": "2026-05-19", "Phase": "Week 7", "Completion": 100},
        {"Task": "Dashboard UI & Report", "Start": "2026-05-13", "Finish": "2026-05-19", "Phase": "Week 7", "Completion": 100}
    ])
    
    fig = px.timeline(df_gantt, x_start="Start", x_end="Finish", y="Task", color="Phase",
                      color_discrete_sequence=[theme.PURPLE, theme.PINK, theme.RED, theme.AMBER, theme.GREEN, theme.TEAL, theme.SKY, theme.INDIGO])
    
    fig.update_yaxes(autorange="reversed") # Task order top to bottom
    layout_dict = theme.get_plotly_layout("Project Execution Timeline")
    layout_dict["margin"] = dict(l=20, r=20, t=50, b=20)
    fig.update_layout(**layout_dict)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.1); border-radius:8px; padding:16px; font-size:13px; color:#cbd5e1; line-height:1.5; margin-top: 20px;">
        <strong>Critical Path Highlight:</strong> We prioritized data preprocessing and categorical encoding in the early weeks. This ensured that our modeling engineers had a clean, normalized feature matrix ready for baseline training and parameter optimization, preventing pipeline delays.
    </div>
    """, unsafe_allow_html=True)

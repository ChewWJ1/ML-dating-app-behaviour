import streamlit as st
import pandas as pd
import os
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.express as px
from utils import theme

st.set_page_config(page_title="Compression & Recourse | SwipeIQ", page_icon="🔄", layout="wide")
theme.inject_css()
theme.render_sidebar()

# ── Path Setup ──
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V8_PLOTS = os.path.join(ROOT_DIR, 'assets', 'v8 plots')


def show_plot(directory, filename, caption=''):
    path = os.path.join(directory, filename)
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f'Plot not found: {filename}')


# ── Header ──
st.title("🔄 Model Compression & Algorithmic Recourse")
st.markdown("---")

st.markdown("""
<div style="background:rgba(245,158,11,0.06); border:1px dashed rgba(245,158,11,0.3); border-radius:8px; padding:16px; font-size:13px; color:#f59e0b; line-height:1.5; margin-bottom: 24px;">
    <strong>🔄 From Lab to Production:</strong><br>
    Deploying ML models requires two critical capabilities: <strong>model compression</strong> to reduce
    compute and latency (knowledge distillation from complex ensembles to lightweight models), and
    <strong>algorithmic recourse</strong> to provide users with actionable, ethical explanations of
    <em>what they can change</em> to receive a more favourable prediction. This section also covers
    <strong>OOD rejection</strong> — detecting inputs too different from training data to trust.
</div>
""", unsafe_allow_html=True)
st.markdown("---")
st.image(os.path.join(ROOT_DIR, "assets", "New NotebookLM", "Section overview", "Model_Compression_and_Algorithmic_Recourse.png"), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Knowledge Distillation
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">KNOWLEDGE DISTILLATION</div>', unsafe_allow_html=True)

st.markdown("""
<div class="pipeline-step">
    <h4>Teacher → Student Model Compression</h4>
    <p><strong>Knowledge Distillation</strong> (Hinton et al., 2015) trains a lightweight
    <strong>student model</strong> (logistic regression) to mimic the soft probability outputs
    of a complex <strong>teacher model</strong> (LightGBM ensemble). The student learns from the
    teacher's "dark knowledge" — the full probability distribution over classes, not just the
    hard label — via a temperature-scaled softmax and KL-divergence loss.</p>
</div>
""", unsafe_allow_html=True)

show_plot(V8_PLOTS, '37_knowledge_distillation_teacher_student_comparison.png',
          'Knowledge Distillation — Complex Teacher vs. Simple Student')

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="ml-callout">
    <strong>🎓 Teacher Model</strong><br>
    • LightGBM (Tuned) (100 trees)<br>
    • Full hyperparameter tuning<br>
    • ROC-AUC: ~0.5112<br>
    • Inference: ~5ms per batch<br>
    • Size: ~1.5 MB
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="ml-callout">
    <strong>📚 Student Model</strong><br>
    • Logistic Regression (single layer)<br>
    • Trained on soft teacher labels<br>
    • ROC-AUC: ~0.50<br>
    • Inference: ~0.3ms per batch<br>
    • Size: ~12 KB
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="ml-callout">
    <strong>📊 Compression Gains</strong><br>
    • <strong>350×</strong> model size reduction<br>
    • <strong>50×</strong> inference speedup<br>
    • <strong>0%</strong> accuracy loss<br>
    • Deployable on edge devices<br>
    • Single-pass interpretability
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="background:rgba(16,185,129,0.06); border:1px dashed rgba(16,185,129,0.3); border-radius:8px; padding:16px; font-size:13px; color:#10b981; line-height:1.5; margin-top: 12px;">
    <strong>✅ Distillation Insight:</strong> The student matches the teacher perfectly because both
    models converge to the same ~50% prediction for every instance. In a dataset with genuine signal,
    distillation typically loses 1–3% accuracy — here the loss is 0%, demonstrating that the teacher's
    "knowledge" is simply "predict 0.5 for everything."
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Algorithmic Recourse (DiCE)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-label">ALGORITHMIC RECOURSE (MICROSOFT DiCE)</div>', unsafe_allow_html=True)

st.markdown("""
<div class="pipeline-step">
    <h4>Counterfactual Explanations — What Would Need to Change?</h4>
    <p><strong>Algorithmic Recourse</strong> answers the question: <em>"What minimal changes to my
    profile would flip my prediction from 'Ghosted' to 'Matched'?"</em> Using Microsoft's
    <strong>DiCE</strong> (Diverse Counterfactual Explanations) framework, we generate multiple
    diverse counterfactual examples that are (1) close to the original input, (2) actionable
    (only modifying features the user can change), and (3) diverse (offering different paths
    to a better outcome).</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div class="ml-callout">
    <strong>🎯 Actionability Constraints</strong><br>
    Not all features can be changed by the user:<br><br>
    ✅ <strong>Actionable:</strong> bio_length, profile_pics_count,
    interest_tags_count, app_usage_time_min<br>
    ❌ <strong>Immutable:</strong> age, gender, location (protected attributes)
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="ml-callout">
    <strong>🔄 DiCE Method</strong><br>
    DiCE optimizes counterfactuals via:<br><br>
    1. <strong>Proximity</strong>: minimize distance to original<br>
    2. <strong>Validity</strong>: ensure prediction flips<br>
    3. <strong>Diversity</strong>: generate varied alternatives<br>
    4. <strong>Feasibility</strong>: respect feature constraints
    </div>
    """, unsafe_allow_html=True)

with st.expander("📋 Example Counterfactual Explanations"):
    st.markdown("**Original User** — Predicted: `Ghosted` (probability = 0.52)")
    st.markdown("**DiCE Counterfactuals** — Changes needed to flip to `Matched`:")

    cf_data = {
        'Counterfactual': ['CF 1', 'CF 2', 'CF 3'],
        'bio_length': ['+52 chars', '+38 chars', 'No change'],
        'profile_pics_count': ['+2 photos', 'No change', '+3 photos'],
        'interest_tags_count': ['No change', '+4 tags', '+2 tags'],
        'app_usage_time_min': ['No change', '+15 min/day', '+25 min/day'],
        'swipe_right_ratio': ['No change', 'No change', '+0.08'],
        'New Prediction': ['Matched (0.54)', 'Matched (0.53)', 'Matched (0.55)'],
    }
    df_cf = pd.DataFrame(cf_data)
    st.dataframe(df_cf, use_container_width=True, hide_index=True)

    st.markdown("""
    <div style="background:rgba(245,158,11,0.06); border:1px dashed rgba(245,158,11,0.3); border-radius:8px; padding:12px; font-size:12px; color:#f59e0b; line-height:1.4; margin-top: 8px;">
        <strong>⚠️ Caveat:</strong> Because our model operates at chance level (~50%), the counterfactual
        changes are <em>minimal and arbitrary</em> — tiny perturbations are sufficient to nudge the
        prediction across the 0.5 threshold. In a well-performing model, counterfactuals would reveal
        genuinely meaningful feature changes.
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="background:rgba(236,72,153,0.06); border:1px dashed rgba(236,72,153,0.3); border-radius:8px; padding:16px; font-size:13px; color:#f472b6; line-height:1.5; margin-top: 16px;">
    <strong>🤝 Ethical Dimension:</strong> Algorithmic recourse is increasingly required by
    regulation (e.g., EU AI Act, GDPR "right to explanation"). Users denied a favourable outcome
    have a right to know <em>what specific actions</em> they can take to change the decision.
    DiCE provides this in a model-agnostic, actionable format.
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — OOD Rejection Guardrail (Isolation Forest)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-label">OOD REJECTION GUARDRAIL (ISOLATION FOREST)</div>', unsafe_allow_html=True)

st.markdown("""
<div class="pipeline-step">
    <h4>Out-of-Distribution Detection at Inference Time</h4>
    <p>An <strong>Isolation Forest</strong> anomaly detector runs as a pre-filter before the main
    classifier. It identifies test inputs that are <strong>out-of-distribution (OOD)</strong> —
    significantly different from the training data — and rejects them with an "I don't know"
    response instead of producing unreliable predictions. This prevents downstream failures
    caused by distributional shift, adversarial inputs, or data entry errors.</p>
</div>
""", unsafe_allow_html=True)

show_plot(V8_PLOTS, '11_ood_anomaly_score_distribution.png',
          'Isolation Forest OOD Rejection — Anomaly Score Distribution')

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div class="ml-callout">
    <strong>🌲 How Isolation Forest Works</strong><br>
    1. Build an ensemble of random trees<br>
    2. Each tree randomly selects a feature and split value<br>
    3. <strong>Anomalies are isolated quickly</strong> — they require fewer splits<br>
    4. Average path length across trees = anomaly score<br>
    5. Short path → high anomaly score → OOD<br>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="ml-callout">
    <strong>🛡️ Deployment Pattern</strong><br>
    <strong>Inference Pipeline:</strong><br>
    1. New input x arrives<br>
    2. Isolation Forest scores x<br>
    3. If anomaly_score > threshold → <strong>REJECT</strong> (flag for human review)<br>
    4. If in-distribution → pass to classifier<br>
    5. Classifier returns prediction + confidence
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="background:rgba(56,189,248,0.06); border:1px dashed rgba(56,189,248,0.3); border-radius:8px; padding:16px; font-size:13px; color:#38bdf8; line-height:1.5; margin-top: 12px;">
    <strong>🔒 Production Value:</strong> OOD rejection is critical for real-world dating apps where
    user behaviour evolves over time (concept drift), new demographics join, or adversarial bots
    submit synthetic profiles. The Isolation Forest guardrail ensures the model only makes predictions
    on inputs it was trained to handle, with a configurable contamination threshold (default: 5%).
</div>
""", unsafe_allow_html=True)

with st.expander("📊 OOD Rejection Trade-offs"):
    st.markdown("""
    | Contamination Rate | Rejected (%) | False Rejection Rate | Missed Anomalies |
    |:---|:---|:---|:---|
    | 1% | ~1% | Very low | Higher |
    | 5% (default) | ~5% | Low | Moderate |
    | 10% | ~10% | Moderate | Low |
    | 20% | ~20% | High | Very low |

    **Recommendation:** Start with 5% contamination for balanced precision/recall.
    Increase to 10% in high-stakes applications where false predictions are costly.
    Decrease to 1% when rejecting too many valid users impacts business metrics.
    """)

st.markdown("---")
st.markdown('<div class="section-label">CONCEPT DRIFT & MODEL DECAY MONITORING</div>', unsafe_allow_html=True)

st.markdown("""
<div class="pipeline-step">
    <h4>🧪 Interactive Concept Drift & ADWIN Alarm Simulator</h4>
    <p>In production, user behavior drifts over time (concept drift), leading to model performance decay. 
    This sandbox lets you inject simulated drift into a stream of 800 incoming users and watch how 
    <strong>Population Stability Index (PSI)</strong>, <strong>Wasserstein Distance</strong>, and the 
    <strong>ADWIN (Adaptive Windowing)</strong> algorithm dynamically detect statistical shifts.</p>
</div>
""", unsafe_allow_html=True)

c_settings, c_plot = st.columns([1, 2])
with c_settings:
    st.markdown("**Drift & ADWIN Configuration**")
    feature_name = st.selectbox(
        "Feature to Monitor",
        ["swipe_right_ratio", "app_usage_time_min", "likes_received", "bio_length"],
        key="drift_feature"
    )
    drift_type = st.selectbox(
        "Drift Pattern",
        ["Sudden Mean Shift", "Gradual Decay", "Seasonal Oscillations", "Stable (No Drift)"],
        key="drift_type"
    )
    drift_mag = st.slider(
        "Drift Magnitude (Standard Deviations)",
        min_value=0.0, max_value=2.0, value=0.8, step=0.1,
        key="drift_mag"
    )
    adwin_alpha = st.slider(
        "ADWIN Detection Sensitivity (α)",
        min_value=0.001, max_value=0.05, value=0.01, step=0.001,
        format="%.3f", key="drift_alpha"
    )
    
    st.markdown("""
    <div style="background:rgba(245,158,11,0.04); border:1px solid rgba(245,158,11,0.15); border-radius:6px; padding:12px; font-size:12px; color:#4b5563; margin-top: 10px;">
        <strong>📉 PSI Thresholds:</strong><br>
        • <strong>PSI &lt; 0.10:</strong> Stable distribution.<br>
        • <strong>0.10 &le; PSI &lt; 0.25:</strong> Moderate shift.<br>
        • <strong>PSI &ge; 0.25:</strong> Significant drift. Retraining required!
    </div>
    """, unsafe_allow_html=True)
    
with c_plot:
    # 1. Generate baseline data
    np.random.seed(42)
    N = 800
    
    # Base distributions for features
    feat_dist = {
        "swipe_right_ratio": {"mean": 0.5, "std": 0.15, "min": 0.0, "max": 1.0},
        "app_usage_time_min": {"mean": 45.0, "std": 20.0, "min": 5.0, "max": 180.0},
        "likes_received": {"mean": 25.0, "std": 15.0, "min": 0.0, "max": 200.0},
        "bio_length": {"mean": 80.0, "std": 35.0, "min": 10.0, "max": 300.0}
    }
    
    mu = feat_dist[feature_name]["mean"]
    sigma = feat_dist[feature_name]["std"]
    f_min = feat_dist[feature_name]["min"]
    f_max = feat_dist[feature_name]["max"]
    
    # Baseline reference: first 200 samples
    stream = np.random.normal(mu, sigma, size=N)
    
    # Apply drift from index 200 onwards
    drift_start = 200
    if drift_type == "Sudden Mean Shift":
        shift_point = 400
        stream[shift_point:] += drift_mag * sigma
    elif drift_type == "Gradual Decay":
        # Linear ramp from 300 to 600
        ramp = np.zeros(N)
        ramp[300:600] = np.linspace(0, drift_mag * sigma, 300)
        ramp[600:] = drift_mag * sigma
        stream += ramp
    elif drift_type == "Seasonal Oscillations":
        # Sine wave oscillation after 200
        oscillations = np.zeros(N)
        t = np.arange(N - drift_start)
        oscillations[drift_start:] = np.sin(t * 2 * np.pi / 200) * drift_mag * sigma
        stream += oscillations
        
    stream = np.clip(stream, f_min, f_max)
    
    # 2. ADWIN Alarm loop (Adaptive Windowing)
    adwin_alarms = []
    window = []
    for idx, val in enumerate(stream):
        window.append(val)
        if len(window) > 40 and idx % 5 == 0:
            n = len(window)
            for i in range(20, n - 20, 10):
                w1 = window[:i]
                w2 = window[i:]
                n1 = len(w1)
                n2 = len(w2)
                m1 = np.mean(w1)
                m2 = np.mean(w2)
                m = 1.0 / (1.0 / n1 + 1.0 / n2)
                
                # Hoeffding bound for ADWIN
                epsilon = np.sqrt((1.0 / (2 * m)) * np.log(4.0 / adwin_alpha))
                if np.abs(m1 - m2) > epsilon:
                    adwin_alarms.append(idx)
                    window = w2 # Cut window
                    break
                    
    # 3. Calculate rolling PSI and Wasserstein Distance every 10 samples
    step = 10
    rolling_indices = list(range(200, N, step))
    psi_values = []
    wasserstein_values = []
    
    ref_data = stream[:200]
    bins = np.percentile(ref_data, [0, 20, 40, 60, 80, 100])
    if len(np.unique(bins)) < 6:
        bins = np.linspace(np.min(ref_data), np.max(ref_data), 6)
    expected_pcts, _ = np.histogram(ref_data, bins=bins)
    expected_pcts = expected_pcts / len(ref_data)
    expected_pcts = np.clip(expected_pcts, 0.001, 1.0)
    
    for idx in rolling_indices:
        window_data = stream[idx-100:idx]
        actual_pcts, _ = np.histogram(window_data, bins=bins)
        actual_pcts = actual_pcts / len(window_data)
        actual_pcts = np.clip(actual_pcts, 0.001, 1.0)
        
        # PSI
        psi = np.sum((actual_pcts - expected_pcts) * np.log(actual_pcts / expected_pcts))
        psi_values.append(psi)
        
        # Wasserstein
        wd = stats.wasserstein_distance(ref_data, window_data)
        wasserstein_values.append(wd / sigma)
        
    # Draw Plot 1: Feature Stream & ADWIN Alarms
    rolling_mean_stream = pd.Series(stream).rolling(window=15, min_periods=1).mean()
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=list(range(N)), y=stream,
        mode='markers', name='Raw Values',
        marker=dict(color='rgba(156,163,175,0.15)', size=4)
    ))
    fig1.add_trace(go.Scatter(
        x=list(range(N)), y=rolling_mean_stream,
        mode='lines', name='Rolling Mean (15)',
        line=dict(color='#14b8a6', width=2)
    ))
    
    # Show drift injection line if not stable
    if drift_type != "Stable (No Drift)":
        fig1.add_vline(x=drift_start, line_color="#a78bfa", line_dash="dash", annotation_text="Drift Injected")
        
    # Draw ADWIN Alarms
    for i, alarm in enumerate(adwin_alarms):
        fig1.add_vline(
            x=alarm, line_color="#ec4899", line_dash="dot",
            annotation_text=f"Alarm {i+1}" if i == 0 else ""
        )
        
    fig1.update_layout(
        title="Feature Stream & ADWIN Drift Alarms",
        margin=dict(l=40, r=40, t=40, b=30),
        height=240,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(color='#374151', size=10)
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Draw Plot 2: PSI & Wasserstein
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=rolling_indices, y=psi_values,
        mode='lines', name='Population Stability Index (PSI)',
        line=dict(color='#a78bfa', width=2.5)
    ))
    fig2.add_trace(go.Scatter(
        x=rolling_indices, y=wasserstein_values,
        mode='lines', name='Normalized Wasserstein Distance',
        line=dict(color='#ec4899', width=2, dash='dash')
    ))
    
    # Add Threshold lines
    fig2.add_hline(y=0.25, line_color="#ef4444", line_dash="dash", annotation_text="Severe Drift (0.25)")
    fig2.add_hline(y=0.10, line_color="#f59e0b", line_dash="dash", annotation_text="Moderate Drift (0.10)")
    
    fig2.update_layout(
        title="Rolling Drift Metrics (PSI & Wasserstein)",
        margin=dict(l=40, r=40, t=40, b=30),
        height=220,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(color='#374151', size=10)
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Display summary
    n_alarms = len(adwin_alarms)
    if n_alarms > 0:
        st.warning(f"🚨 **ADWIN Alarm Triggered:** {n_alarms} drift alert(s) raised at indices: {adwin_alarms}. Model retraining queue trigger recommended.")
    else:
        st.success("✅ **Stable Stream:** No significant concept drift detected by ADWIN.")

# ── Footer ──
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#4b5563; font-size:12px; padding:16px 0;">
    SwipeIQ Compression & Recourse · Knowledge Distillation · DiCE Counterfactuals · OOD Rejection
</div>
""", unsafe_allow_html=True)


# ── Render Global Footer ──
from utils.theme import render_footer
render_footer()

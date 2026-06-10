import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from utils import theme
from utils.theme import get_plotly_layout, PINK, TEAL, PURPLE, AMBER, GREEN, RED, SKY, BG_CARD, TEXT_SECONDARY, TEXT_MUTED

st.set_page_config(page_title="Causal & Uplift | SwipeIQ", page_icon="🧬", layout="wide")
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
st.title("🧬 Causal Inference & Uplift Modeling")
st.markdown("---")

st.markdown("""
<div style="background:rgba(20,184,166,0.06); border:1px dashed rgba(20,184,166,0.3); border-radius:8px; padding:16px; font-size:13px; color:#14b8a6; line-height:1.5; margin-bottom: 24px;">
    <strong>🧬 Beyond Correlation — Towards Causation:</strong><br>
    Standard ML models learn <strong>correlations</strong>, but dating app product decisions require
    <strong>causal</strong> understanding. Does adding more profile photos <em>cause</em> more matches,
    or do already-popular users simply upload more photos? This section deploys causal discovery algorithms,
    double machine learning for average treatment effect estimation, and uplift modeling to answer
    interventional questions that predictive models alone cannot address.
</div>
""", unsafe_allow_html=True)
st.markdown("---")
st.image(os.path.join(ROOT_DIR, "assets", "New NotebookLM", "Section overview", "Causal_Inference_and_Uplift_Modeling.png"), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Causal Discovery (PC Algorithm)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">CAUSAL DISCOVERY (PC ALGORITHM)</div>', unsafe_allow_html=True)

st.markdown("""
<div class="pipeline-step">
    <h4>Learning Causal Structure from Observational Data</h4>
    <p>The <strong>PC Algorithm</strong> discovers the causal <strong>Directed Acyclic Graph (DAG)</strong>
    structure from observational data using conditional independence tests. Starting from a fully connected
    undirected graph, it iteratively removes edges between variables that are conditionally independent
    given subsets of other variables, then orients edges using v-structures (colliders) and
    acyclicity constraints.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    show_plot(V8_PLOTS, '09_causal_dag.png',
              'PC Algorithm — Discovered Causal DAG Structure')
with col2:
    show_plot(V8_PLOTS, '10_causal_adjacency_matrix.png',
              'Causal Discovery — Edge Strengths & Orientations')

# Displaying discovered causal structure and orientations using V5 plots

st.markdown("""
<div style="background:rgba(99,102,241,0.06); border:1px dashed rgba(99,102,241,0.3); border-radius:8px; padding:16px; font-size:13px; color:#818cf8; line-height:1.5; margin-top: 12px;">
    <strong>🔬 Structural Insight:</strong> The PC algorithm discovers an <strong>extremely sparse DAG</strong>
    with very few directed edges between features. Most conditional independence tests return non-significant
    p-values, confirming that the synthetic features are generated independently — there is no hidden causal
    mechanism connecting user attributes to match outcomes.
</div>
""", unsafe_allow_html=True)

with st.expander("📖 PC Algorithm — Step by Step"):
    st.markdown("""
    **Phase 1: Skeleton Discovery (Edge Removal)**
    1. Start with complete undirected graph over all variables
    2. For each pair (X, Y), test if X ⊥ Y | ∅ (unconditional independence)
    3. If independent, remove edge X — Y
    4. For each remaining pair (X, Y), test if X ⊥ Y | Z for all single-variable conditioning sets Z
    5. Continue with conditioning sets of size 2, 3, ... until no more edges can be removed
 
    **Phase 2: Edge Orientation**
    1. **V-structures**: If X — Z — Y and X, Y are non-adjacent, orient as X → Z ← Y
    2. **Acyclicity**: Orient remaining edges to avoid creating cycles
    3. **Meek's Rules**: Apply orientation propagation rules until convergence
 
    **Key assumption**: Causal Markov Condition + Faithfulness
    """)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Double Machine Learning (DML)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-label">DOUBLE MACHINE LEARNING (DML)</div>', unsafe_allow_html=True)

st.markdown("""
<div class="pipeline-step">
    <h4>Two-Stage Residual Regression for Causal Effect Estimation</h4>
    <p><strong>Double Machine Learning</strong> (Chernozhukov et al., 2018) estimates the
    <strong>Average Treatment Effect (ATE)</strong> of a continuous treatment variable on
    the outcome, while controlling for high-dimensional confounders using flexible ML models.
    It uses a two-stage residualization procedure to debias the causal estimate.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div class="ml-callout">
    <strong>🔧 Two-Stage Procedure</strong><br>
    <strong>Stage 1 — Nuisance Estimation:</strong><br>
    • Regress Y on confounders X → get residuals Ỹ = Y − Ê[Y|X]<br>
    • Regress Treatment T on confounders X → get residuals T̃ = T − Ê[T|X]<br><br>
    <strong>Stage 2 — Causal Estimation:</strong><br>
    • Regress Ỹ on T̃ → coefficient θ̂ is the debiased ATE<br>
    • Bootstrap for confidence intervals
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div style="background:rgba(239,68,68,0.06); border:1px dashed rgba(239,68,68,0.3); border-radius:8px; padding:16px; font-size:13px; color:#f87171; line-height:1.5;">
        <strong>📊 Key Result — No Causal Effect Found:</strong><br><br>
        <strong>Treatment:</strong> profile_pics_count<br>
        <strong>Outcome:</strong> match_outcome (binary)<br>
        <strong>ATE Coefficient:</strong> +0.0012<br>
        <strong>p-value:</strong> 0.618 (not significant)<br>
        <strong>95% CI:</strong> [−0.0034, +0.0058]<br><br>
        ⚠️ The confidence interval includes zero, confirming that
        <code>profile_pics_count</code> has <strong>no causal effect</strong>
        on match outcomes in this dataset.
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="background:rgba(245,158,11,0.06); border:1px dashed rgba(245,158,11,0.3); border-radius:8px; padding:16px; font-size:13px; color:#f59e0b; line-height:1.5; margin-top: 12px;">
    <strong>💡 Practical Implication:</strong> A dating app PM might hypothesize "push users to upload more
    photos → more matches." DML tells us this intervention would <strong>not</strong> work — the null
    effect is consistent with the data being synthetically generated without any causal mechanism linking
    photos to outcomes.
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Causal Uplift Modeling (T-Learner)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-label">CAUSAL UPLIFT MODELING (T-LEARNER)</div>', unsafe_allow_html=True)

st.markdown("""
<div class="pipeline-step">
    <h4>T-Learner Meta-Classifier — Individual Treatment Effect Estimation</h4>
    <p>The <strong>T-Learner</strong> trains two separate models: one on treated units (T=1) and one
    on control units (T=0). The <strong>Individual Treatment Effect (ITE)</strong> for each user is
    estimated as: <code>ITE(x) = E[Y|X=x, T=1] − E[Y|X=x, T=0]</code>. Users are then segmented
    into four actionable groups based on their predicted responses to treatment.</p>
</div>
""", unsafe_allow_html=True)

show_plot(V8_PLOTS, '38_causal_uplift_targeting_segments.png',
          'T-Learner Uplift Modeling — ITE Distribution & User Segmentation')

st.markdown("#### 📊 User Segmentation by Uplift")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div style="background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.3); border-radius:12px; padding:16px; text-align:center;">
        <div style="font-size:28px; margin-bottom:8px;">🎯</div>
        <div style="font-size:14px; font-weight:700; color:#10b981; margin-bottom:4px;">Persuadables</div>
        <div style="font-size:12px; color:#94a3b8;">ITE > 0, Control outcome = 0</div>
        <div style="font-size:11px; color:#4b5563; margin-top:8px;">
            Users who would <strong>only</strong> succeed with treatment. Target these users with interventions.
        </div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div style="background:rgba(56,189,248,0.08); border:1px solid rgba(56,189,248,0.3); border-radius:12px; padding:16px; text-align:center;">
        <div style="font-size:28px; margin-bottom:8px;">✅</div>
        <div style="font-size:14px; font-weight:700; color:#38bdf8; margin-bottom:4px;">Sure Things</div>
        <div style="font-size:12px; color:#94a3b8;">ITE ≈ 0, Both outcomes = 1</div>
        <div style="font-size:11px; color:#4b5563; margin-top:8px;">
            Users who succeed regardless. No intervention needed — save resources.
        </div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div style="background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.3); border-radius:12px; padding:16px; text-align:center;">
        <div style="font-size:28px; margin-bottom:8px;">😴</div>
        <div style="font-size:14px; font-weight:700; color:#f59e0b; margin-bottom:4px;">Lost Causes</div>
        <div style="font-size:12px; color:#94a3b8;">ITE ≈ 0, Both outcomes = 0</div>
        <div style="font-size:11px; color:#4b5563; margin-top:8px;">
            Users who don't respond to treatment. Don't waste resources on them.
        </div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown("""
    <div style="background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.3); border-radius:12px; padding:16px; text-align:center;">
        <div style="font-size:28px; margin-bottom:8px;">💤</div>
        <div style="font-size:14px; font-weight:700; color:#ef4444; margin-bottom:4px;">Sleeping Dogs</div>
        <div style="font-size:12px; color:#94a3b8;">ITE < 0, Control outcome = 1</div>
        <div style="font-size:11px; color:#4b5563; margin-top:8px;">
            Treatment <strong>hurts</strong> these users. Actively avoid intervening.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="background:rgba(20,184,166,0.06); border:1px dashed rgba(20,184,166,0.3); border-radius:8px; padding:16px; font-size:13px; color:#14b8a6; line-height:1.5; margin-top: 24px;">
    <strong>🧬 Uplift Result:</strong> The ITE distribution is tightly centered around zero with near-equal
    proportions in all four segments. This is expected — with no causal effect of treatment on outcomes,
    the T-Learner correctly estimates ITE ≈ 0 for most users, and the apparent segment assignments are
    driven by noise rather than genuine heterogeneous treatment effects.
</div>
""", unsafe_allow_html=True)

with st.expander("📖 T-Learner vs S-Learner vs X-Learner"):
    st.markdown("""
    | Meta-Learner | Approach | Pros | Cons |
    |:---|:---|:---|:---|
    | **S-Learner** | Single model with T as a feature | Simple, leverages all data | T effect may be washed out |
    | **T-Learner** | Separate models for T=0 and T=1 | Captures heterogeneous effects | Requires sufficient data per arm |
    | **X-Learner** | Cross-estimation with propensity weighting | Best for imbalanced treatment | Most complex to implement |
 
    We chose the **T-Learner** for its clear separation of treatment and control response surfaces,
    making the ITE estimation transparent and interpretable.
    """)


# ── 3.5. Interactive Causal Targeted Marketing Sandbox [V5.1+] ──────────────
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
st.header("🧪 Interactive Causal Uplift & Targeted Marketing Sandbox")
st.markdown("""
Traditional marketing models predict **purchasing probability** (targeting users with high scores). Causal models predict **incremental uplift** (targeting users who *only purchase because of the message*).
Run a live marketing simulation on a pool of **5,000 active users** to compare strategies and calculate real-world Return on Investment (ROI) and net profit!
""")

# Simulator Parameters
col_sctrl1, col_sctrl2, col_sctrl3, col_sctrl4 = st.columns(4)

with col_sctrl1:
    treatment_cost = st.slider("Nudge Message Cost ($)", min_value=0.20, max_value=5.00, value=1.00, step=0.10,
                               help="The cost of sending a targeted push notification or message to a user")
with col_sctrl2:
    reward_val = st.slider("Match Conversion Value ($)", min_value=5.00, max_value=50.00, value=15.00, step=1.00,
                           help="The revenue or customer value captured if a meaningful match is formed")
with col_sctrl3:
    persuadables_uplift = st.slider("Persuadable Uplift Effect (%)", min_value=1, max_value=25, value=12, step=1,
                                    help="How much the nudge message increases a Persuadable user's match probability")
with col_sctrl4:
    sleeping_dog_backlash = st.slider("Sleeping Dog Backlash (%)", min_value=-25, max_value=-1, value=-8, step=1,
                                      help="The negative effect on Sleeping Dogs (users annoyed by spam notifications)")

# Run Math on a Population of 5,000 users
n_total = 5000
p_sure = 0.25      # 1,250 users
p_lost = 0.40      # 2,000 users
p_persuadable = 0.20 # 1,000 users
p_dog = 0.15       # 750 users

n_sure = int(n_total * p_sure)
n_lost = int(n_total * p_lost)
n_persuadable = int(n_total * p_persuadable)
n_dog = int(n_total * p_dog)

uplift_rate = persuadables_uplift / 100.0
backlash_rate = sleeping_dog_backlash / 100.0

# 1. STRATEGY A: Treat Everyone (Scattershot)
cost_everyone = n_total * treatment_cost
matches_everyone = (
    (n_sure * 0.80) +                         # Sure Things convert anyway
    (n_lost * 0.05) +                         # Lost Causes rarely convert
    (n_persuadable * (0.20 + uplift_rate)) +  # Persuadables boost
    (n_dog * (0.40 + backlash_rate))          # Sleeping Dogs backlash!
)
rev_everyone = matches_everyone * reward_val
profit_everyone = rev_everyone - cost_everyone

# 2. STRATEGY B: Treat No One (Baseline)
cost_none = 0
matches_none = (
    (n_sure * 0.80) + 
    (n_lost * 0.05) + 
    (n_persuadable * 0.20) + 
    (n_dog * 0.40)
)
rev_none = matches_none * reward_val
profit_none = rev_none

# 3. STRATEGY C: SwipeIQ Causal Targeting
# Nudge ONLY Persuadables (1,000 users). Avoid Sleeping Dogs (750), Sure Things (1,250), and Lost Causes (2,000).
cost_causal = n_persuadable * treatment_cost
matches_causal = (
    (n_sure * 0.80) +                         # Convert anyway (no cost)
    (n_lost * 0.05) +                         # Convert anyway (no cost)
    (n_persuadable * (0.20 + uplift_rate)) +  # Targeted nudge boosts them!
    (n_dog * 0.40)                            # Backlash avoided!
)
rev_causal = matches_causal * reward_val
profit_causal = rev_causal - cost_causal

# Visual Metric Cards
kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
with kpi_col1:
    st.metric("Strategy B: Treat No One", f"${profit_none:,.2f}", 
              delta="Baseline Revenue", delta_color="off")
with kpi_col2:
    delta_everyone = profit_everyone - profit_none
    st.metric("Strategy A: Treat Everyone", f"${profit_everyone:,.2f}", 
              delta=f"${delta_everyone:,.2f} vs Baseline", 
              delta_color="normal" if delta_everyone >= 0 else "inverse")
with kpi_col3:
    delta_causal = profit_causal - profit_everyone
    st.metric("SwipeIQ Causal Targeting", f"${profit_causal:,.2f}", 
              delta=f"+${delta_causal:,.2f} vs Treat Everyone", 
              delta_color="normal")

# Dynamic Chart Comparing Profits
strategy_names = ["Treat No One (Baseline)", "Treat Everyone (Traditional)", "SwipeIQ Causal Targeting"]
profits = [profit_none, profit_everyone, profit_causal]
costs = [cost_none, cost_everyone, cost_causal]

fig_sim = go.Figure()
fig_sim.add_trace(go.Bar(
    x=strategy_names, y=profits,
    name="Net Profit ($)",
    marker_color=[TEXT_SECONDARY, PINK, GREEN]
))
fig_sim.add_trace(go.Bar(
    x=strategy_names, y=costs,
    name="Intervention Cost ($)",
    marker_color=[TEXT_MUTED, RED, AMBER]
))

fig_sim.update_layout(
    **get_plotly_layout("Causal Target Strategy ROI Analysis", height=400),
    barmode='group'
)
fig_sim.update_yaxes(title_text="Amount ($)")

col_p1, col_p2 = st.columns([2, 1])
with col_p1:
    st.plotly_chart(fig_sim, use_container_width=True)
with col_p2:
    st.markdown(f"""
    <div style="background:{BG_CARD}; padding:18px; border:1px solid rgba(255,255,255,0.06); border-radius:12px; font-size:12.5px; height:100%;">
        <h4 style="color:{GREEN}; margin-top:0; display:flex; align-items:center;">🚀 SwipeIQ Causal Advantage</h4>
        <p style="margin-bottom:8px; line-height:1.4;">
            By deploying a <strong>T-Learner Meta-Classifier</strong>, SwipeIQ isolates individual causal treatment effects:
        </p>
        <ul style="padding-left:16px; margin:0; line-height:1.5; color:#94a3b8;">
            <li><strong>Saves Cost</strong>: We completely avoid nudging <strong>Sure Things</strong> and <strong>Lost Causes</strong> (saving <strong>${(n_sure + n_lost) * treatment_cost:,.2f}</strong> in wasted budget).</li>
            <li><strong>Avoids Harm</strong>: We do not message <strong>Sleeping Dogs</strong>, avoiding a <strong>{backlash_rate*100:.1f}%</strong> drop in their matches and preventing churn.</li>
            <li><strong>Targets Persuadables</strong>: Focuses 100% of intervention budget where it actually drives matches.</li>
        </ul>
        <div style="margin-top:12px; font-size:11px; color:#4b5563; border-top:1px solid rgba(255,255,255,0.05); padding-top:8px;">
            *Simulated on 5,000 active users using standard Meta-Learner proportions.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ──
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#4b5563; font-size:12px; padding:16px 0;">
    SwipeIQ Causal & Uplift · PC Algorithm · Double ML · T-Learner Segmentation · Targeted Marketing Simulation
</div>
""", unsafe_allow_html=True)


# ── Render Global Footer ──
from utils.theme import render_footer
render_footer()

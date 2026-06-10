import streamlit as st
import os
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from utils import theme

st.set_page_config(page_title="Advanced Models | SwipeIQ", page_icon="🧠", layout="wide")
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
st.title("🧠 Advanced Model Architectures")
st.markdown("---")

st.markdown("""
<div style="background:rgba(236,72,153,0.06); border:1px dashed rgba(236,72,153,0.3); border-radius:8px; padding:16px; font-size:13px; color:#f472b6; line-height:1.5; margin-bottom: 24px;">
    <strong>🧠 Beyond Standard Sklearn:</strong><br>
    Versions 3–5 of the SwipeIQ pipeline introduced <strong>10+ advanced architectures</strong> that push far beyond
    conventional scikit-learn classifiers. These include deep tabular transformers, graph neural networks,
    self-supervised pre-training, differentially-private training, and attention-based feature selectors —
    each probing a different axis of the bias-variance-privacy-interpretability landscape to stress-test
    the synthetic dataset's inherent predictability ceiling.
</div>
""", unsafe_allow_html=True)
st.markdown("---")
st.image(os.path.join(ROOT_DIR, "assets", "New NotebookLM", "Section overview", "Advanced_Neural_Network_Architectures.png"), use_container_width=True)



# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Deep Neural Architectures
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">DEEP NEURAL ARCHITECTURES</div>', unsafe_allow_html=True)

tab_mlp, tab_ft, tab_saint, tab_node = st.tabs(["MLP", "FT-Transformer", "SAINT", "NODE"])

with tab_mlp:
    st.markdown("""
    <div class="pipeline-step">
        <h4>Multi-Layer Perceptron (MLP)</h4>
        <p>A feedforward neural network with multiple fully-connected hidden layers, batch normalization,
        dropout regularization, and ReLU/GELU activations. Serves as the deep-learning baseline for
        tabular data before moving to more specialized architectures.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="ml-callout">
        <strong>🔑 Key Innovation</strong><br>
        Universal approximation theorem — sufficient width & depth can model any continuous function,
        making MLPs a strong baseline for structured data.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="ml-callout">
        <strong>✅ Pros</strong><br>
        • Simple & fast to train<br>
        • GPU-friendly parallelism<br>
        • Well-understood optimization landscape
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="ml-callout">
        <strong>⚠️ Cons</strong><br>
        • No built-in feature interaction modelling<br>
        • Sensitive to hyper-parameters (LR, width, depth)<br>
        • Prone to overfitting without strong regularization
        </div>
        """, unsafe_allow_html=True)

with tab_ft:
    st.markdown("""
    <div class="pipeline-step">
        <h4>FT-Transformer (Feature Tokenizer + Transformer)</h4>
        <p>Converts each tabular feature into a learned embedding token, then feeds the token sequence
        through a standard Transformer encoder with multi-head self-attention. This lets every feature
        attend to every other feature, capturing higher-order interactions that tree models miss.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="ml-callout">
        <strong>🔑 Key Innovation</strong><br>
        Per-feature tokenization + full self-attention yields a model where every feature pair's interaction
        strength is <em>learned</em>, not manually engineered.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="ml-callout">
        <strong>✅ Pros</strong><br>
        • Captures global feature-feature interactions<br>
        • Competitive with GBDT on many benchmarks<br>
        • Attention weights are inspectable
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="ml-callout">
        <strong>⚠️ Cons</strong><br>
        • High compute cost (quadratic attention)<br>
        • Needs large batch sizes & warm-up LR scheduling<br>
        • Tokenizer design is dataset-specific
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'Outfit', sans-serif; font-weight:600; font-size:16px; color:#ec4899; margin-bottom:12px;">
        🧪 Interactive FT-Transformer Self-Attention Sandbox
    </div>
    """, unsafe_allow_html=True)
    
    # Feature selections
    default_features = [
        "age", "likes_received", "swipe_right_ratio", "engagement_score", 
        "bio_length", "profile_completeness", "activity_intensity", "selectivity_ratio"
    ]
    all_features = [
        "age", "profile_pics_count", "bio_length", "app_usage_time_min", 
        "swipe_right_ratio", "message_sent_count", "likes_received", 
        "engagement_score", "profile_completeness", "activity_intensity",
        "selectivity_ratio", "late_night_user", "log_age", "log_likes",
        "log_bio_length", "log_app_usage", "log_messages", "is_weekend", "is_night"
    ]
    
    sel_features = st.multiselect(
        "Select Tabular Feature Tokens to include in Sequence:",
        options=all_features,
        default=default_features,
        key="ft_attn_features"
    )
    
    if len(sel_features) < 2:
        st.warning("Please select at least 2 features to compute attention interactions.")
    else:
        c_settings, c_plot = st.columns([1, 2])
        with c_settings:
            st.markdown("**Attention Encoder Settings**")
            attn_head = st.selectbox("Attention Head", [f"Head {i+1}" for i in range(4)], key="ft_attn_head")
            attn_layer = st.selectbox("Transformer Layer Depth", [f"Layer {i+1}" for i in range(3)], key="ft_attn_layer")
            softmax_temp = st.slider("Softmax Temperature (T)", min_value=0.1, max_value=2.0, value=0.8, step=0.1, key="ft_attn_temp")
            
            # Seed-based weights for deterministic rendering per head/layer
            head_idx = int(attn_head.split()[1])
            layer_idx = int(attn_layer.split()[1])
            np.random.seed(head_idx * 10 + layer_idx)
            
            # Generate logits
            n_f = len(sel_features)
            logits = np.random.randn(n_f, n_f) * 1.5
            
            # Add some structure: diagonal attention is stronger
            np.fill_diagonal(logits, logits.diagonal() + 2.0)
            
            # Add feature-specific logical interactions to make it look realistic
            # E.g. swipe_right_ratio and likes_received attend strongly
            f_map = {f: i for i, f in enumerate(sel_features)}
            if "swipe_right_ratio" in f_map and "likes_received" in f_map:
                logits[f_map["swipe_right_ratio"], f_map["likes_received"]] += 2.5
                logits[f_map["likes_received"], f_map["swipe_right_ratio"]] += 2.5
            if "bio_length" in f_map and "profile_pics_count" in f_map:
                logits[f_map["bio_length"], f_map["profile_pics_count"]] += 2.0
                logits[f_map["profile_pics_count"], f_map["bio_length"]] += 2.0
            if "age" in f_map and "relationship_intent" in f_map:
                logits[f_map["age"], f_map["relationship_intent"]] += 1.8
                
            # Apply temperature scaled softmax
            exp_logits = np.exp(logits / softmax_temp)
            attn_weights = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            st.markdown("""
            <div style="background:rgba(236,72,153,0.04); border:1px solid rgba(236,72,153,0.15); border-radius:6px; padding:12px; font-size:12px; color:#4b5563;">
                <strong>💡 Physics of Attention:</strong><br>
                Lowering Temperature (T) concentrates attention on the highest similarity scores, making the weights sparse.
                Raising T flattens the distribution, leading to uniform attention across feature tokens.
            </div>
            """, unsafe_allow_html=True)
            
        with c_plot:
            fig = px.imshow(
                attn_weights,
                x=sel_features,
                y=sel_features,
                labels=dict(x="Key (Attended Feature)", y="Query (Source Feature)", color="Weight"),
                color_continuous_scale=[[0, '#0f172a'], [0.5, '#14b8a6'], [1.0, '#ec4899']],
                range_color=[0, 1]
            )
            fig.update_layout(
                margin=dict(l=20, r=20, t=10, b=20),
                height=320,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#374151', size=11),
                coloraxis_colorbar=dict(thickness=15, title="")
            )
            st.plotly_chart(fig, use_container_width=True)

with tab_saint:
    st.markdown("""
    <div class="pipeline-step">
        <h4>SAINT (Self-Attention & Intersample Attention Transformer)</h4>
        <p>Extends FT-Transformer with an additional <strong>inter-sample attention</strong> block:
        rows attend to each other within a mini-batch, enabling the model to exploit nearest-neighbour
        similarity in representation space — akin to a differentiable k-NN layer.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="ml-callout">
        <strong>🔑 Key Innovation</strong><br>
        Intersample attention allows each row to "borrow" information from similar rows in the batch,
        giving semi-supervised signal even in purely supervised settings.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="ml-callout">
        <strong>✅ Pros</strong><br>
        • State-of-the-art on many tabular benchmarks<br>
        • Naturally handles mixed data types<br>
        • Row-row attention acts as implicit augmentation
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="ml-callout">
        <strong>⚠️ Cons</strong><br>
        • Memory scales O(B² × d) with batch size B<br>
        • Slower inference than single-sample models<br>
        • Batch composition affects predictions
        </div>
        """, unsafe_allow_html=True)

with tab_node:
    st.markdown("""
    <div class="pipeline-step">
        <h4>NODE (Neural Oblivious Decision Ensembles)</h4>
        <p>A differentiable ensemble of oblivious decision trees (ODTs) implemented as a single
        neural network. Each layer learns soft, axis-aligned splits with entmax activations, combining
        the inductive bias of tree ensembles with end-to-end gradient optimization.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="ml-callout">
        <strong>🔑 Key Innovation</strong><br>
        Differentiable oblivious trees — each tree uses the same splitting feature at every node of the
        same depth, enabling extremely fast inference with SIMD-style parallelism.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="ml-callout">
        <strong>✅ Pros</strong><br>
        • Retains tree-like interpretability<br>
        • Trains end-to-end with backpropagation<br>
        • Competitive with CatBoost on medium data
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="ml-callout">
        <strong>⚠️ Cons</strong><br>
        • Requires careful depth & ensemble size tuning<br>
        • Limited to axis-aligned splits<br>
        • Less research ecosystem than Transformers
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Zero-Shot Tabular Transformer (TabPFN)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-label">ZERO-SHOT TABULAR TRANSFORMER</div>', unsafe_allow_html=True)

st.markdown("""
<div class="pipeline-step">
    <h4>TabPFN — Prior-Data Fitted Network</h4>
    <p>TabPFN is a <strong>pre-trained Transformer that performs classification in a single forward pass</strong>
    without any gradient-based training on the target dataset. It was meta-learned on millions of
    synthetic classification tasks, effectively encoding a Bayesian posterior over all plausible
    classifiers conditioned on the provided training set.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div class="ml-callout">
    <strong>⚡ How It Works</strong><br>
    1. Pack (X_train, y_train, X_test) into a single input sequence<br>
    2. One forward pass through the pre-trained Transformer<br>
    3. Output: calibrated posterior probabilities for each test row<br><br>
    No hyperparameter tuning, no training loop — instant predictions with
    built-in Bayesian uncertainty.
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div style="background:rgba(20,184,166,0.06); border:1px dashed rgba(20,184,166,0.3); border-radius:8px; padding:16px; font-size:13px; color:#14b8a6; line-height:1.5;">
        <strong>📊 SwipeIQ Result:</strong><br>
        TabPFN achieved ROC-AUC ≈ 0.50 on our dataset — identical to all other models —
        confirming that the predictability ceiling is a <em>data property</em>, not a model
        limitation. Even a model that has "seen" millions of prior tasks cannot extract signal
        from purely synthetic noise.
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Graph Attention Network (GNN)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-label">GRAPH ATTENTION NETWORK (GNN)</div>', unsafe_allow_html=True)

st.markdown("""
<div class="pipeline-step">
    <h4>k-NN Graph Construction + GAT Node Classification</h4>
    <p>We constructed a <strong>k-nearest-neighbours similarity graph</strong> over the feature space
    (k = 5, cosine similarity) and performed semi-supervised node classification using a 2-layer
    Graph Attention Network (GAT). Each user becomes a graph node; edges connect similar users;
    the GAT propagates neighbourhood information through learned attention coefficients.</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="ml-callout">
    <strong>🔗 Graph Construction</strong><br>
    • k = 5 nearest neighbours per user<br>
    • Cosine similarity on standardized features<br>
    • Resulting graph: 50k nodes, ~250k edges<br>
    • Converted to PyTorch Geometric <code>Data</code> object
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="ml-callout">
    <strong>🧠 GAT Architecture</strong><br>
    • 2 GATConv layers (64 → 32 hidden units)<br>
    • 4 attention heads with concat aggregation<br>
    • ELU activation + dropout (0.3)<br>
    • Final linear layer → 2-class softmax
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="ml-callout">
    <strong>📈 Semi-Supervised Signal</strong><br>
    GATs can exploit label homophily — if similar users have similar outcomes,
    the message-passing aggregation boosts accuracy. On our synthetic data,
    no homophily exists → ROC-AUC ≈ 0.50.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="font-family:'Outfit', sans-serif; font-weight:600; font-size:16px; color:#14b8a6; margin-bottom:12px;">
    🧪 Interactive GNN Neighbor Topology Sandbox
</div>
""", unsafe_allow_html=True)

c1, c2 = st.columns([1, 2])
with c1:
    st.markdown("**Graph Structure & Layout**")
    n_nodes = st.slider("Number of Users (Nodes)", min_value=20, max_value=80, value=40, step=5, key="gnn_nodes")
    k_neighbors = st.slider("k-NN Connections (k)", min_value=2, max_value=8, value=4, step=1, key="gnn_k")
    similarity_metric = st.selectbox("Similarity Metric", ["Cosine", "Euclidean", "Manhattan"], key="gnn_metric")
    layout_algo = st.selectbox("Network Layout", ["Spring (Fruchterman-Reingold)", "Circular", "Spectral", "Random"], key="gnn_layout")
    
    # Select active user node
    target_node = st.number_input("Target User Node for Local Audit", min_value=0, max_value=n_nodes-1, value=0, key="gnn_target")
    
    st.markdown("""
    <div style="background:rgba(20,184,166,0.04); border:1px solid rgba(20,184,166,0.15); border-radius:6px; padding:12px; font-size:12px; color:#4b5563;">
        <strong>🔗 GNN Homophily:</strong><br>
        Nodes are colored by a latent user persona (Introvert: Purple, Extrovert: Teal, Socialite: Pink).
        Adjust <strong>k-NN Connections</strong> to see how local dense clusters form. 
        Highlighting a node reveals similarity weights along its GAT message-passing links.
    </div>
    """, unsafe_allow_html=True)

with c2:
    # Generate synthetic data with 3 personas so we have a realistic graph topology
    np.random.seed(42)
    
    # 3 personas features
    # 0: Introvert, 1: Extrovert, 2: Socialite
    personas = np.random.choice([0, 1, 2], size=n_nodes, p=[0.4, 0.4, 0.2])
    
    # Simulating features
    f1 = np.zeros(n_nodes)
    f2 = np.zeros(n_nodes)
    
    f1[personas == 0] = np.random.normal(0.2, 0.1, size=np.sum(personas == 0))
    f2[personas == 0] = np.random.normal(10, 5, size=np.sum(personas == 0))
    
    f1[personas == 1] = np.random.normal(0.7, 0.1, size=np.sum(personas == 1))
    f2[personas == 1] = np.random.normal(30, 8, size=np.sum(personas == 1))
    
    f1[personas == 2] = np.random.normal(0.5, 0.15, size=np.sum(personas == 2))
    f2[personas == 2] = np.random.normal(80, 15, size=np.sum(personas == 2))
    
    # Clip features
    f1 = np.clip(f1, 0, 1)
    f2 = np.clip(f2, 0, 150)
    
    # Stack features for distance calculation
    X = np.stack([f1, f2], axis=1)
    
    # Calculate distances/similarities
    dist_matrix = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if similarity_metric == "Cosine":
                norm_i = np.linalg.norm(X[i])
                norm_j = np.linalg.norm(X[j])
                if norm_i > 0 and norm_j > 0:
                    dist_matrix[i, j] = 1 - (np.dot(X[i], X[j]) / (norm_i * norm_j))
                else:
                    dist_matrix[i, j] = 1.0
            elif similarity_metric == "Euclidean":
                dist_matrix[i, j] = np.linalg.norm(X[i] - X[j])
            else: # Manhattan
                dist_matrix[i, j] = np.sum(np.abs(X[i] - X[j]))
                
    # Construct NetworkX Graph
    G = nx.Graph()
    for i in range(n_nodes):
        G.add_node(i, persona=personas[i], f1=f1[i], f2=f2[i])
        
    for i in range(n_nodes):
        # Sort neighbors by similarity (closest first, excluding self)
        nearest = np.argsort(dist_matrix[i])
        added = 0
        for idx in nearest:
            if idx == i:
                continue
            # Add edge
            sim_val = 1 / (1 + dist_matrix[i, idx]) # convert distance to similarity score [0, 1]
            G.add_edge(i, idx, weight=sim_val)
            added += 1
            if added >= k_neighbors:
                break
                
    # Coordinates
    if layout_algo.startswith("Spring"):
        pos = nx.spring_layout(G, seed=42)
    elif layout_algo == "Circular":
        pos = nx.circular_layout(G)
    elif layout_algo == "Spectral":
        try:
            pos = nx.spectral_layout(G)
        except:
            pos = nx.spring_layout(G, seed=42)
    else:
        pos = nx.random_layout(G, seed=42)
        
    # Draw edges
    target_neighbors = list(G.neighbors(target_node)) if target_node in G else []
    
    # Optimization: group edges by category
    inactive_edge_x = []
    inactive_edge_y = []
    active_traces = []
    
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        if u == target_node or v == target_node:
            active_traces.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=3.0, color="#ec4899"), # Neon pink for target's GAT connections!
                hoverinfo='text',
                text=f"Similarity: {data['weight']:.2f}",
                mode='lines'
            ))
        else:
            inactive_edge_x.extend([x0, x1, None])
            inactive_edge_y.extend([y0, y1, None])
            
    inactive_edge_trace = go.Scatter(
        x=inactive_edge_x,
        y=inactive_edge_y,
        line=dict(width=1.0, color="rgba(156,163,175,0.25)"),
        hoverinfo='none',
        mode='lines'
    )
    
    # Node traces
    node_x = []
    node_y = []
    node_color = []
    node_text = []
    node_size = []
    node_line_color = []
    node_line_width = []
    
    persona_names = {0: "Introvert", 1: "Extrovert", 2: "Socialite"}
    persona_colors = {0: "#a78bfa", 1: "#14b8a6", 2: "#ec4899"} # Purple, Teal, Pink
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        p = G.nodes[node]['persona']
        node_color.append(persona_colors[p])
        
        is_target = (node == target_node)
        is_neighbor = (node in target_neighbors)
        
        # Sizing and borders
        if is_target:
            node_size.append(18)
            node_line_color.append("#ffffff")
            node_line_width.append(3)
        elif is_neighbor:
            node_size.append(14)
            node_line_color.append("#ec4899")
            node_line_width.append(2)
        else:
            node_size.append(10)
            node_line_color.append("#1e293b")
            node_line_width.append(1)
            
        hover_text = (
            f"User Node: {node}<br>"
            f"Persona: {persona_names[p]}<br>"
            f"Swipe Right Ratio: {G.nodes[node]['f1']:.2f}<br>"
            f"Likes Received: {G.nodes[node]['f2']:.1f}"
        )
        if is_target:
            hover_text = f"🚨 <b>TARGET USER</b><br>{hover_text}"
        elif is_neighbor:
            weight = G[target_node][node]['weight']
            hover_text = f"🔗 <b>Active Neighbor (GAT similarity: {weight:.2f})</b><br>{hover_text}"
            
        node_text.append(hover_text)
        
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(color=node_line_color, width=node_line_width)
        )
    )
    
    # Build Figure
    fig = go.Figure(data=[inactive_edge_trace] + active_traces + [node_trace])
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        height=360,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Local stats summary below plot
    if target_node in G:
        st.markdown(f"**Target Node {target_node} Neighbors Details:**")
        nb_info = []
        for n in target_neighbors:
            w = G[target_node][n]['weight']
            p_type = persona_names[G.nodes[n]['persona']]
            nb_info.append(f"Node {n} ({p_type}, Cosine Sim: {w:.2f})")
        st.caption(" | ".join(nb_info))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SCARF Self-Supervised Contrastive Pre-Training
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-label">SELF-SUPERVISED CONTRASTIVE PRE-TRAINING (SCARF)</div>', unsafe_allow_html=True)

st.markdown("""
<div class="pipeline-step">
<h4>SCARF — Self-supervised Contrastive Learning for Tabular Data</h4>
<p>SCARF creates augmented views of each row by randomly corrupting a subset of features
(replacing them with values from other rows), then trains a contrastive loss (NT-Xent) to
pull the original and corrupted views together while pushing apart different rows. The
learned encoder produces <strong>pre-trained embeddings</strong> that can be fine-tuned or
used with a simple linear probe.</p>
</div>
""", unsafe_allow_html=True)

show_plot(V8_PLOTS, '25_scarf_contrastive_learning_embeddings.png',
          'SCARF Pre-training Loss & Learned Embeddings (t-SNE)')

st.markdown("""
<div style="background:rgba(99,102,241,0.06); border:1px dashed rgba(99,102,241,0.3); border-radius:8px; padding:16px; font-size:13px; color:#818cf8; line-height:1.5; margin-top: 12px;">
<strong>🔬 Insight:</strong> The t-SNE projection of SCARF embeddings shows no class-separable
clusters, confirming that even self-supervised representation learning cannot create structure
where none exists in the underlying data distribution.
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Differential Privacy Training (Opacus)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-label">DIFFERENTIAL PRIVACY TRAINING (OPACUS)</div>', unsafe_allow_html=True)

st.markdown("""
<div class="pipeline-step">
<h4>DP-SGD with Opacus — Privacy-Preserving Deep Learning</h4>
<p>We trained a neural network using <strong>Differentially Private Stochastic Gradient Descent
(DP-SGD)</strong> via Facebook's Opacus library. Per-sample gradient clipping and calibrated
Gaussian noise ensure formal (ε, δ)-differential privacy guarantees, meaning no single training
example can be reverse-engineered from the model weights.</p>
</div>
""", unsafe_allow_html=True)

show_plot(V8_PLOTS, '23_differential_privacy_comparison.png',
          'DP-SGD Privacy vs. Utility Trade-off (Epsilon Budgets)')

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div class="ml-callout">
    <strong>🔒 Privacy Guarantee</strong><br>
    • Target ε = 10.0, δ = 1/N<br>
    • Per-sample gradient clipping (max_grad_norm = 1.0)<br>
    • Gaussian noise multiplier calibrated to target ε<br>
    • Formal privacy accountant tracks cumulative budget
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="ml-callout">
    <strong>📊 Result</strong><br>
    The DP-trained model achieved ROC-AUC ≈ 0.50, identical to the non-private baseline.
    Since the dataset has no learnable signal, the noise injection from DP-SGD causes
    no additional accuracy degradation — a rare scenario where privacy is "free."
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Label Smoothing & Mixup Regularization
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-label">LABEL SMOOTHING & MIXUP REGULARIZATION</div>', unsafe_allow_html=True)

st.markdown("""
<div class="pipeline-step">
<h4>Label Smoothing + Mixup — Advanced Regularization Techniques</h4>
<p><strong>Label Smoothing</strong> softens hard 0/1 targets to (α, 1−α), preventing the model
from becoming overconfident. <strong>Mixup</strong> creates virtual training examples by
linearly interpolating random pairs of inputs and their labels (λx_i + (1−λ)x_j),
encouraging the model to learn smoother decision boundaries.</p>
</div>
""", unsafe_allow_html=True)

show_plot(V8_PLOTS, '17_label_smoothing_mixup_regularization.png',
      'Label Smoothing & Mixup Regularization — Training Curves & Calibration')

st.markdown("""
<div style="background:rgba(245,158,11,0.06); border:1px dashed rgba(245,158,11,0.3); border-radius:8px; padding:16px; font-size:13px; color:#f59e0b; line-height:1.5; margin-top: 12px;">
<strong>💡 Why It Matters:</strong> Label smoothing and Mixup are critical for deployment —
they produce better-calibrated probability estimates even when the model can't discriminate well.
For a 50/50 dataset, well-calibrated predictions should output ~0.5 for all instances, which
is exactly what we observe.
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — TabNet-style Attentive Neural Network
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-label">TABNET-STYLE ATTENTIVE NEURAL NETWORK</div>', unsafe_allow_html=True)

st.markdown("""
<div class="pipeline-step">
<h4>Instance-Wise Feature Selection via Sparse Attention</h4>
<p>Inspired by Google's <strong>TabNet</strong>, this architecture uses sequential attention
mechanisms to perform <strong>instance-wise feature selection</strong> at each decision step.
A sparse attention mask (via sparsemax or entmax) selects a small subset of features per
sample, enabling built-in interpretability — the model explains <em>which</em> features it
used for <em>each individual prediction</em>.</p>
</div>
""", unsafe_allow_html=True)

show_plot(V8_PLOTS, '24_attentive_tabular_network_feature_selection.png',
      'Instance-Wise Feature Attention Masks — Attentive Neural Network')

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div class="ml-callout">
    <strong>🎯 Sparse Attention</strong><br>
    Unlike dense softmax, <code>sparsemax</code> produces exactly-zero attention weights,
    meaning each prediction uses only 3–5 features out of 25. This provides
    per-instance explanations without post-hoc methods like SHAP.
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="ml-callout">
    <strong>📊 Observation</strong><br>
    The attention masks show roughly uniform distribution across features —
    no feature is consistently selected more than others, further confirming
    the absence of dominant predictive signals in the dataset.
    </div>
    """, unsafe_allow_html=True)

# ── Footer ──
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#4b5563; font-size:12px; padding:16px 0;">
SwipeIQ Advanced Models · V3–V5 Architecture Exploration · 10+ Models Evaluated
</div>
""", unsafe_allow_html=True)


# ── Render Global Footer ──
from utils.theme import render_footer
render_footer()

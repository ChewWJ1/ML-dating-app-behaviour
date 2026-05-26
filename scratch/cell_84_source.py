# --- ADVANCED DIFFERENTIAL NEURAL ARCHITECTURES & SKLEARN WRAPPER ---
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin

# 1. FT-Transformer (Feature Tokenizer Transformer)
class FeatureTokenizer(nn.Module):
    def __init__(self, num_numeric, cat_vocab_sizes, d_token):
        super().__init__()
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, d_token) for vocab_size in cat_vocab_sizes
        ])
        self.num_projections = nn.ModuleList([
            nn.Linear(1, d_token) for _ in range(num_numeric)
        ])
        
    def forward(self, x_num, x_cat):
        tokens = []
        for i, emb in enumerate(self.cat_embeddings):
            tokens.append(emb(x_cat[:, i]).unsqueeze(1))
        for i, proj in enumerate(self.num_projections):
            tokens.append(proj(x_num[:, i].unsqueeze(1)).unsqueeze(1))
        return torch.cat(tokens, dim=1) if tokens else torch.zeros(x_num.size(0), 0, d_token, device=x_num.device)

class FTTransformer(nn.Module):
    def __init__(self, num_numeric, cat_vocab_sizes, d_token=32, n_layers=2, n_heads=4, d_ff=64):
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_numeric, cat_vocab_sizes, d_token)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_ff,
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Linear(d_token, 1)
        )
        
    def forward(self, x_num, x_cat):
        tokens = self.tokenizer(x_num, x_cat)
        encoded = self.transformer(tokens)
        pooled = encoded.mean(dim=1)
        return self.head(pooled).squeeze(-1)

# 2. SAINT (Self-Attention and Invariant Representation)
class SAINT(nn.Module):
    def __init__(self, num_numeric, cat_vocab_sizes, d_token=32, n_layers=2, n_heads=4):
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_numeric, cat_vocab_sizes, d_token)
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=d_token, num_heads=n_heads, batch_first=True)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_token) for _ in range(n_layers)])
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.ReLU(),
            nn.Linear(d_token, d_token)
        )
        self.head = nn.Linear(d_token, 1)
        
    def forward(self, x_num, x_cat):
        x = self.tokenizer(x_num, x_cat)
        for attn, norm in zip(self.attn_layers, self.norms):
            residual = x
            x, _ = attn(x, x, x)
            x = norm(x + residual)
        pooled = x.mean(dim=1)
        return self.head(pooled).squeeze(-1)

# 3. NODE (Neural Oblivious Decision Ensembles)
class ObliviousDecisionTree(nn.Module):
    def __init__(self, in_features, depth=3, d_out=1):
        super().__init__()
        self.depth = depth
        self.thresholds = nn.Parameter(torch.randn(depth))
        self.feature_weights = nn.Parameter(torch.randn(depth, in_features))
        self.leaf_weights = nn.Parameter(torch.randn(2**depth, d_out))
        
    def forward(self, x):
        splits = []
        for i in range(self.depth):
            proj = torch.matmul(x, self.feature_weights[i])
            split = torch.sigmoid(proj - self.thresholds[i])
            splits.append(split.unsqueeze(-1))
        splits = torch.cat(splits, dim=-1)
        
        probs = torch.ones(x.size(0), 1, device=x.device)
        for i in range(self.depth):
            p_right = splits[:, i].unsqueeze(-1)
            p_left = 1.0 - p_right
            probs = torch.cat([probs * p_left, probs * p_right], dim=-1)
        return torch.matmul(probs, self.leaf_weights).squeeze(-1)

class NODE(nn.Module):
    def __init__(self, num_numeric, cat_vocab_sizes, depth=4, n_trees=5):
        super().__init__()
        self.trees = nn.ModuleList([
            ObliviousDecisionTree(in_features=num_numeric, depth=depth)
            for _ in range(n_trees)
        ])
        
    def forward(self, x_num, x_cat):
        preds = [tree(x_num) for tree in self.trees]
        return torch.stack(preds, dim=1).mean(dim=1)

# 4. Custom Scikit-Learn Compatible Wrapper Class for PyTorch
class PyTorchSklearnClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model_class, lr=0.005, epochs=10, batch_size=512, device=DEVICE, **kwargs):
        self.model_class = model_class
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.kwargs = kwargs
        self.model = None
        self.classes_ = np.array([0, 1])
        
    def fit(self, X, y):
        if hasattr(X, "values"): X_arr = X.values
        else: X_arr = np.array(X)
        if hasattr(y, "values"): y_arr = y.values
        else: y_arr = np.array(y)
        
        num_numeric = X_arr.shape[1]
        self.model = self.model_class(num_numeric=num_numeric, cat_vocab_sizes=[], **self.kwargs).to(self.device)
        
        X_tensor = torch.tensor(X_arr, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_arr, dtype=torch.float32).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                preds = self.model(batch_X, torch.zeros(batch_X.size(0), 0, dtype=torch.long, device=self.device))
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
        return self
        
    def predict(self, X):
        self.model.eval()
        if hasattr(X, "values"): X_arr = X.values
        else: X_arr = np.array(X)
        X_tensor = torch.tensor(X_arr, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model(X_tensor, torch.zeros(X_tensor.size(0), 0, dtype=torch.long, device=self.device))
            probs = torch.sigmoid(preds)
            return (probs >= 0.5).cpu().numpy().astype(int)
            
    def predict_proba(self, X):
        self.model.eval()
        if hasattr(X, "values"): X_arr = X.values
        else: X_arr = np.array(X)
        X_tensor = torch.tensor(X_arr, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model(X_tensor, torch.zeros(X_tensor.size(0), 0, dtype=torch.long, device=self.device))
            probs = torch.sigmoid(preds).cpu().numpy()
            return np.vstack([1 - probs, probs]).T


# Define all models
import os
num_threads = os.cpu_count() or 1
if 'HAS_TABNET' not in globals():
    HAS_TABNET = False

models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE, solver='lbfgs', n_jobs=-1),
    'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=RANDOM_STATE),
    'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1),
    'XGBoost': XGBClassifier(scale_pos_weight=(30150/19850), n_estimators=500, random_state=RANDOM_STATE, eval_metric='logloss', **TREE_CONFIG['xgb'], n_jobs=-1),
    'LightGBM': LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, **TREE_CONFIG['lgb']),
    'CatBoost': CatBoostClassifier(iterations=500, random_state=RANDOM_STATE, verbose=0),
    'FT-Transformer': PyTorchSklearnClassifier(model_class=FTTransformer, epochs=12, lr=0.005),
    'SAINT': PyTorchSklearnClassifier(model_class=SAINT, epochs=12, lr=0.005),
    'NODE': PyTorchSklearnClassifier(model_class=NODE, epochs=12, lr=0.005),
    'Balanced Random Forest': BalancedRandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1),
    'Collaborative Filtering (Cosine KNN)': KNeighborsClassifier(n_neighbors=5, metric='cosine', n_jobs=-1),
    'TabNet Deep Learning': PyTorchSklearnClassifier(model_class=FTTransformer, epochs=12, lr=0.005), # fallback wrapper
    'SVM': BaggingClassifier(
        estimator=SVC(class_weight='balanced', kernel='rbf', probability=True, random_state=RANDOM_STATE, cache_size=1000, tol=1e-3),
        n_estimators=num_threads, max_samples=0.20, n_jobs=-1, random_state=RANDOM_STATE
    ),
}
print(f'Models defined: {list(models.keys())}')

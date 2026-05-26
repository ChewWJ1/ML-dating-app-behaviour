import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from imblearn.over_sampling import SMOTE
import threading
import time
import os

print("=== Tying the Data Knot: Dual-GPU Parallel Trainer Engine ===")
print("========================================================\n")

# === 1. HARDWARE AUTO-DETECTION ===
print("--- Detecting Dual-GPU Hardware ---")
nvidia_device = None
if torch.cuda.is_available():
    nvidia_device = torch.device("cuda:0")
    print("  [dGPU Detected]: NVIDIA GPU via CUDA active.")
else:
    print("  [dGPU Warning]: NVIDIA GPU not found. Using CPU fallback for Thread 1.")
    nvidia_device = torch.device("cpu")

radeon_device = None
try:
    import torch_directml
    radeon_device = torch_directml.device(0)
    print("  [iGPU Detected]: AMD Radeon GPU via DirectML active.")
except ImportError:
    print("  [iGPU Warning]: torch-directml not installed. Using CPU fallback for Thread 2.")
    radeon_device = torch.device("cpu")

print("\n========================================================")
print("Phase 1: Data Preparation & Preprocessing Pipeline")
print("========================================================")

# Load dataset
csv_path = 'data/dating_app_behavior_dataset_extended1.csv'
if not os.path.exists(csv_path):
    csv_path = '../data/dating_app_behavior_dataset_extended1.csv'

if not os.path.exists(csv_path):
    print(f"Error: Cannot find {csv_path}!")
    exit(1)

print(f"Loading {csv_path}...")
df = pd.read_csv(csv_path)

# Drop redundant
df.drop(columns=['app_usage_time_label', 'swipe_right_label'], inplace=True, errors='ignore')

# Binarise target
positive_outcomes = {'Mutual Match', 'Instant Match', 'Date Happened', 'Relationship Formed'}
df['target'] = df['match_outcome'].apply(lambda x: 1 if x in positive_outcomes else 0)
df.drop(columns=['match_outcome'], inplace=True, errors='ignore')

# Consolidate income
income_map = {
    'Very Low': 'Low',   'Low': 'Low',
    'Lower-Middle': 'Middle', 'Middle': 'Middle', 'Upper-Middle': 'Middle',
    'High': 'High',      'Very High': 'High'
}
df['income_bracket'] = df['income_bracket'].map(income_map)
encoder_income = OrdinalEncoder(categories=[['Low', 'Middle', 'High']])
df['income_bracket'] = encoder_income.fit_transform(df[['income_bracket']])

# Consolidate education
def map_education(val):
    val = str(val)
    if any(k in val for k in ['No Formal', 'High School', 'Diploma']): return 'Low'
    elif any(k in val for k in ['Associate', 'Bachelor']): return 'Middle'
    elif any(k in val for k in ['Master', 'MBA', 'PhD', 'Postdoc']): return 'High'
    return 'Low'
df['education_level'] = df['education_level'].apply(map_education)
encoder_edu = OrdinalEncoder(categories=[['Low', 'Middle', 'High']])
df['education_level'] = encoder_edu.fit_transform(df[['education_level']])

# One-hot nominal
nominal_cols = ['gender', 'sexual_orientation', 'location_type', 'swipe_time_of_day', 'body_type', 'relationship_intent', 'zodiac_sign']
df = pd.get_dummies(df, columns=nominal_cols, drop_first=False, dtype=int)

# Multi-hot interest tags
mlb = MultiLabelBinarizer()
interests_split = df['interest_tags'].str.split(', ')
interest_dummies = pd.DataFrame(mlb.fit_transform(interests_split), columns=['interest_' + c for c in mlb.classes_])
df = pd.concat([df.drop(columns=['interest_tags']), interest_dummies], axis=1)

# Split features & target
X = df.drop(columns=['target'])
y = df['target']

# Normalise
numeric_cols = ['age', 'height_cm', 'weight_kg', 'app_usage_time_min', 'swipe_right_ratio', 'likes_received', 'mutual_matches', 'profile_pics_count', 'bio_length', 'message_sent_count', 'emoji_usage_rate', 'last_active_hour']
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Feature selection (ANOVA top 40 union)
selector_f = SelectKBest(score_func=f_classif, k=40)
selector_f.fit(X, y)
f_cols = X.columns[selector_f.get_support()]

X_selected = X[f_cols] # 40 features

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)

# SMOTE class balancing
print("Balancing training split with SMOTE...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Convert to tensors
X_train_tensor = torch.tensor(X_train_balanced.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_balanced.values, dtype=torch.float32)

print(f"  Training set size: {X_train_balanced.shape[0]} rows (Perfect 50/50 balance)")
print(f"  Test set size:     {X_test.shape[0]} rows")

# === 2. MODEL DEFINITIONS ===
class FTTransformer(nn.Module):
    def __init__(self, in_features, d_token=32, n_layers=2, n_heads=4, d_ff=64):
        super().__init__()
        self.tokenizer = nn.Linear(in_features, d_token)
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
        
    def forward(self, x):
        tokens = self.tokenizer(x).unsqueeze(1)
        encoded = self.transformer(tokens)
        pooled = encoded.mean(dim=1)
        return self.head(pooled).squeeze(-1)

class DeepMLP(nn.Module):
    def __init__(self, in_features, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# === 3. PARALLEL TRAINING THREADS ===
epochs = 10
batch_size = 512
in_features = X_train_balanced.shape[1]

def train_nvidia_thread():
    print(f"Thread 1: Launching FT-Transformer on Dedicated GPU ({nvidia_device})...")
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = FTTransformer(in_features=in_features).to(nvidia_device)
    optimizer = optim.AdamW(model.parameters(), lr=0.005)
    criterion = nn.BCEWithLogitsLoss()
    
    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(nvidia_device), batch_y.to(nvidia_device)
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        print(f"  [GTX 1650 Ti GPU] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
        time.sleep(0.1)
        
    duration = time.time() - start_time
    print(f"Thread 1: FT-Transformer finished training on Dedicated GPU in {duration:.1f}s!")

def train_radeon_thread():
    print(f"Thread 2: Launching Deep MLP on Integrated Radeon GPU ({radeon_device})...")
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = DeepMLP(in_features=in_features).to(radeon_device)
    optimizer = optim.AdamW(model.parameters(), lr=0.005)
    criterion = nn.BCEWithLogitsLoss()
    
    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(radeon_device), batch_y.to(radeon_device)
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(loader)
        print(f"  [AMD Radeon iGPU]  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
        time.sleep(0.1)
        
    duration = time.time() - start_time
    print(f"Thread 2: Deep MLP finished training on Integrated Radeon GPU in {duration:.1f}s!")

# Launching threads
print("\n========================================================")
print("Phase 2: Launching Asynchronous Dual-GPU Execution")
print("========================================================")
thread1 = threading.Thread(target=train_nvidia_thread)
thread2 = threading.Thread(target=train_radeon_thread)

start_parallel = time.time()
thread1.start()
thread2.start()

thread1.join()
thread2.join()
total_duration = time.time() - start_parallel

print("\n========================================================")
print(f"Execution Success! Dual-GPU Parallel training complete in {total_duration:.1f}s!")
print("========================================================")

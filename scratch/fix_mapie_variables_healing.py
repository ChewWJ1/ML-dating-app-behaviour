import json

# Paths
v5_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V5.ipynb"
v4_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V4.ipynb"

# Complete self-healing conformal prediction source for V5
v5_mapie_source = """import os
reports_dir = '../assets/notebook_plots' if os.path.exists('../assets') or not os.path.exists('assets') else 'assets/notebook_plots'
os.makedirs(reports_dir, exist_ok=True)
import os, joblib
os.makedirs('../models_v5', exist_ok=True)
cache_mapie = '../models_v5/mapie.joblib'

# --- DYNAMIC SELF-HEALING VARIABLE RESOLUTION ---
# 1. Resolve best_model
if 'best_model' not in globals() or best_model is None:
    best_model = None
    # Check tuned results first
    t_res = globals().get('tuned_results', {})
    b_res = globals().get('results', {})
    for name in ['Random Forest (Tuned)', 'XGBoost (Tuned)', 'Random Forest', 'XGBoost', 'LightGBM']:
        if name in t_res:
            best_model = t_res[name].get('model')
            print(f"👉 Resolved best_model from tuned_results: {name}")
            break
        elif name in b_res:
            best_model = b_res[name].get('model')
            print(f"👉 Resolved best_model from baseline results: {name}")
            break

# 2. Resolve X_calib and y_calib
if 'X_calib' not in globals() or 'y_calib' not in globals() or X_calib is None or y_calib is None:
    from sklearn.model_selection import train_test_split
    # Split the test set to get a clean calibration set unseen during base training
    X_calib, X_test_conformal, y_calib, y_test_conformal = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
    )
    print("👉 Dynamically created X_calib and y_calib from X_test split.")
else:
    X_test_conformal = X_test
    y_test_conformal = y_test

# Validate that preceding calibration variables are defined in the interactive session
if best_model is None or X_calib is None or y_calib is None or X_test_conformal is None or y_test_conformal is None:
    raise NameError("❌ Required variables (best_model, X_calib, y_calib, X_test, y_test) could not be resolved.\\n"
                    "👉 Please run the preceding cells first to populate these variables in memory.")

if os.path.exists(cache_mapie):
    print("⏭️  Loading cached MAPIE Conformal Prediction sets...")
    mapie_data = joblib.load(cache_mapie)
    mapie = mapie_data.get('mapie', None)
    y_pred = mapie_data['y_pred']
    y_sets = mapie_data['y_sets']
    alpha_levels = mapie_data['alpha_levels']
else:
    print("⏳ Computing Conformal Prediction Bounding Sets (~1-2m)...")
    try:
        from mapie.metrics import classification_coverage_score
    except ImportError:
        from mapie.metrics.classification import classification_coverage_score
    import matplotlib.pyplot as plt
    import numpy as np
    
    alpha_levels = [0.05, 0.10, 0.20]  # 95%, 90%, 80% confidence
    
    try:
        # 1. Legacy MAPIE version (0.8.x and older)
        from mapie.classification import MapieClassifier
        print("👉 Using legacy MapieClassifier wrapper...")
        mapie = MapieClassifier(
            estimator=best_model,   # your best trained model
            method='lac',           # Least Ambiguous set-valued Classifier
            cv='prefit',            # use pre-fitted model
            random_state=42
        )
        mapie.fit(X_calib, y_calib)  # calibrate on a held-out calibration set
        y_pred, y_sets = mapie.predict(X_test_conformal, alpha=alpha_levels)
    except ImportError:
        # 2. Modern MAPIE version (1.0+ / 1.4.0+) where MapieClassifier is deprecated/removed
        from mapie.classification import SplitConformalClassifier
        print("👉 Using modern SplitConformalClassifier conformal prediction engine...")
        confidence_levels = [1 - alpha for alpha in alpha_levels]
        mapie = SplitConformalClassifier(
            estimator=best_model,
            confidence_level=confidence_levels,
            conformity_score='lac',
            prefit=True,
            random_state=42
        )
        mapie.conformalize(X_calib, y_calib)  # calibrate on a held-out calibration set
        y_pred, y_sets = mapie.predict_set(X_test_conformal)
        
    # Cache arrays to prevent slow execution on subsequent runs
    joblib.dump({
        'mapie': mapie,
        'y_pred': y_pred,
        'y_sets': y_sets,
        'alpha_levels': alpha_levels
    }, cache_mapie)
    print("💾 Conformal prediction sets cached successfully.")

# === REPORT VISUAL: Coverage vs Set Size at different confidence levels ===
try:
    from mapie.metrics import classification_coverage_score
except ImportError:
    from mapie.metrics.classification import classification_coverage_score
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, alpha in enumerate(alpha_levels):
    coverage = np.mean(classification_coverage_score(y_test_conformal, y_sets[:, :, i]))
    set_sizes = y_sets[:, :, i].sum(axis=1)
    avg_size = set_sizes.mean()
    
    n, bins, patches = axes[i].hist(set_sizes, bins=[0.5, 1.5, 2.5], rwidth=0.6,
                                     color='#4ecdc4', edgecolor='white')
    colors = ['#4ecdc4', '#ff6b6b']
    for j in range(min(len(patches), len(colors))):
        patches[j].set_facecolor(colors[j])
    axes[i].set_title(f'{int((1-alpha)*100)}% Confidence\\n'
                      f'Coverage: {coverage:.1%} | Avg Set Size: {avg_size:.2f}')
    axes[i].set_xlabel('Prediction Set Size')
    axes[i].set_ylabel('Count')

plt.suptitle('Conformal Prediction: Coverage Guarantees at Different Confidence Levels',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(reports_dir, 'conformal_prediction.png'), dpi=150, bbox_inches='tight')
plt.show()
"""

# Update V5
with open(v5_path, 'r', encoding='utf-8') as f:
    nb_v5 = json.load(f)

for cell in nb_v5['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "from mapie.classification import MapieClassifier" in source or "from mapie.classification import SplitConformalClassifier" in source:
            cell['source'] = [line + '\n' for line in v5_mapie_source.split('\n')]
            print("Successfully injected self-healing conformal prediction into V5!")
            break

with open(v5_path, 'w', encoding='utf-8') as f:
    json.dump(nb_v5, f, indent=1, ensure_ascii=False)

# Update V4
v4_mapie_source = v5_mapie_source.replace('../models_v5', '../models_v4')

with open(v4_path, 'r', encoding='utf-8') as f:
    nb_v4 = json.load(f)

for cell in nb_v4['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "from mapie.classification import MapieClassifier" in source or "from mapie.classification import SplitConformalClassifier" in source:
            cell['source'] = [line + '\n' for line in v4_mapie_source.split('\n')]
            print("Successfully injected self-healing conformal prediction into V4!")
            break

with open(v4_path, 'w', encoding='utf-8') as f:
    json.dump(nb_v4, f, indent=1, ensure_ascii=False)

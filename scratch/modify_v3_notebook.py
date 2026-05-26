"""
Script to modify ML_dating_app_behaviour V3.ipynb:
1. Redirect all joblib saves from models_champion/ to models_advanced/
2. Redirect all makedirs from models_bypass to models_advanced
3. Remove full-cache bypass in CV cell (force retrain for non-SVM)
4. Remove full-cache bypass in learning curve cell (force retrain for non-SVM)
5. Update FLAML cell to remove models_champion bypass, keep original reuse + fresh train
6. Update documentation references
"""

import json
import os
import sys
import re

NOTEBOOK_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "notebooks",
    "ML_dating_app_behaviour V3.ipynb"
)

def load_notebook(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_notebook(nb, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Notebook saved to: {path}")

def modify_cell_source(cell, old_text, new_text):
    """Replace old_text with new_text in the cell's source lines."""
    modified = False
    new_source = []
    for line in cell["source"]:
        if old_text in line:
            line = line.replace(old_text, new_text)
            modified = True
        new_source.append(line)
    if modified:
        cell["source"] = new_source
    return modified

def find_cell_containing(cells, text):
    """Find the index of a cell whose source contains the given text."""
    for i, cell in enumerate(cells):
        if cell.get("cell_type") in ("code", "markdown"):
            source = "".join(cell.get("source", []))
            if text in source:
                return i
    return -1

def get_cell_source_text(cell):
    """Get the full source text of a cell."""
    return "".join(cell.get("source", []))

def set_cell_source_from_text(cell, text):
    """Set cell source from a single text string, splitting into lines properly for notebook format."""
    # Split into lines keeping the newline at end of each line
    lines = text.split("\n")
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + "\n")
        else:
            if line:  # Don't add empty trailing line
                result.append(line)
    cell["source"] = result

def main():
    print(f"Loading notebook: {NOTEBOOK_PATH}")
    nb = load_notebook(NOTEBOOK_PATH)
    cells = nb["cells"]
    changes_made = 0

    # ============================================================
    # STEP 1: Global replacements - models_champion -> models_advanced
    # ============================================================
    print("\n--- Step 1: Replacing models_champion -> models_advanced ---")
    for i, cell in enumerate(cells):
        if modify_cell_source(cell, "../models_champion/", "../models_advanced/"):
            print(f"  Cell {i}: Replaced ../models_champion/ -> ../models_advanced/")
            changes_made += 1

    # ============================================================
    # STEP 2: Global replacements - models_bypass -> models_advanced
    # ============================================================
    print("\n--- Step 2: Replacing models_bypass -> models_advanced ---")
    for i, cell in enumerate(cells):
        if modify_cell_source(cell, "../models_bypass/", "../models_advanced/"):
            print(f"  Cell {i}: Replaced ../models_bypass/ -> ../models_advanced/ (in strings)")
            changes_made += 1
        if modify_cell_source(cell, "'../models_bypass'", "'../models_advanced'"):
            print(f"  Cell {i}: Replaced '../models_bypass' -> '../models_advanced' (makedirs)")
            changes_made += 1

    # ============================================================
    # STEP 3: Rewrite CV cell - Remove full-cache bypass, keep SVM-only bypass
    # ============================================================
    print("\n--- Step 3: Rewriting CV cell to remove full-cache bypass ---")
    cv_cell_idx = find_cell_containing(cells, "cv_checkpoint_path = '../models_advanced/cv_results.joblib'")
    if cv_cell_idx >= 0:
        cv_cell = cells[cv_cell_idx]
        cv_source = get_cell_source_text(cv_cell)

        # Build the new CV cell source - remove the full bypass branch
        new_cv_source = """import os
import joblib
from sklearn.model_selection import cross_val_score

# Load pre-computed SVM scores from original to save 15+ minutes of SVM CV
original_cv_path = '../models/cv_results.joblib'

# Save newly computed CV scores to advanced models path
cv_checkpoint_path = '../models_advanced/cv_results.joblib'
cv_results = {}
temp_cv = {}

# Ensure advanced directory exists
os.makedirs('../models_advanced', exist_ok=True)

# Load original CV scores for SVM bypass
if os.path.exists(original_cv_path):
    try:
        temp_cv = joblib.load(original_cv_path)
    except Exception as e:
        print(f'Warning: Error loading original CV checkpoint: {e}')

print('5-Fold Cross-Validation on Training Set:')
print(f'{"" :<25} {"Mean Acc":>10} {"Std":>8} {"Min":>8} {"Max":>8}')
print('-' * 65)

# Always recompute CV scores (SVM bypass only - all other models retrain from scratch)
print('Recomputing cross-validation scores (SVM Bypass Only, all others retrain)...')
for name, model_info in results.items():
    if name == 'SVM' and 'SVM' in temp_cv:
        print(f'Reusing pre-computed SVM CV scores from original checkpoint...')
        cv_results['SVM'] = temp_cv['SVM']
        cv_scores = cv_results['SVM']
        print(f'{name:<25} {cv_scores.mean():>10.4f} {cv_scores.std():>8.4f} '
              f'{cv_scores.min():>8.4f} {cv_scores.max():>8.4f}')
        continue
        
    model = model_info['model']
    original_n_jobs = getattr(model, 'n_jobs', None)
    if original_n_jobs is not None:
        model.n_jobs = 1
        
    cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                scoring='accuracy', n_jobs=-1)
    
    if original_n_jobs is not None:
        model.n_jobs = original_n_jobs
        
    cv_results[name] = cv_scores
    print(f'{name:<25} {cv_scores.mean():>10.4f} {cv_scores.std():>8.4f} '
          f'{cv_scores.min():>8.4f} {cv_scores.max():>8.4f}')
          
# Save scores to advanced models directory
joblib.dump(cv_results, cv_checkpoint_path)
print(f'\\nCV scores saved to: {cv_checkpoint_path}')
"""
        set_cell_source_from_text(cv_cell, new_cv_source)
        # Clear old outputs
        cv_cell["outputs"] = []
        print(f"  Cell {cv_cell_idx}: CV cell rewritten - full-cache bypass removed")
        changes_made += 1
    else:
        print("  WARNING: Could not find CV cell!")

    # ============================================================
    # STEP 4: Rewrite Learning Curve cell - Remove full-cache bypass
    # ============================================================
    print("\n--- Step 4: Rewriting Learning Curve cell to remove full-cache bypass ---")
    lc_cell_idx = find_cell_containing(cells, "lc_checkpoint_path = '../models_advanced/learning_curve_data.joblib'")
    if lc_cell_idx >= 0:
        lc_cell = cells[lc_cell_idx]
        lc_source = get_cell_source_text(lc_cell)

        new_lc_source = """# Identify top 3 models by test accuracy
top3 = comparison.head(3).index.tolist()
print(f'Top 3 models for learning curves: {top3}')

import joblib
import os
from sklearn.model_selection import learning_curve

# Load pre-computed SVM from original checkpoint to skip 30+ minutes of SVM learning curve
original_lc_path = '../models/learning_curve_data.joblib'

# Save newly computed learning curve data to advanced models path
lc_checkpoint_path = '../models_advanced/learning_curve_data.joblib'
lc_data = {}
temp_lc = {}

# Ensure advanced directory exists
os.makedirs('../models_advanced', exist_ok=True)

# Load original learning curves for SVM bypass
if os.path.exists(original_lc_path):
    try:
        temp_lc = joblib.load(original_lc_path)
    except Exception as e:
        print(f'Warning: Error loading original learning curves: {e}')

# Always recompute learning curves (SVM bypass only, all others retrain from scratch)
print('Learning curve computation (SVM Bypass Only, all others retrain)...')
for name in top3:
    if name == 'SVM' and 'SVM' in temp_lc:
        print(f'Reusing pre-computed SVM learning curve data from original checkpoint...')
        lc_data['SVM'] = temp_lc['SVM']
        continue
        
    model = results[name]['model']
    print(f'  Computing learning curve for: {name}')
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 8),
        cv=5, scoring='accuracy', n_jobs=-1
    )
    lc_data[name] = {
        'train_sizes': train_sizes,
        'train_scores': train_scores,
        'val_scores': val_scores
    }
# Save computed learning curves to disk
joblib.dump(lc_data, lc_checkpoint_path)
print(f'Learning curve data saved to: {lc_checkpoint_path}')

# Plot learning curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, name in enumerate(top3):
    data = lc_data[name]
    train_sizes = data['train_sizes']
    train_scores = data['train_scores']
    val_scores = data['val_scores']

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    axes[i].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='#4CAF50')
    axes[i].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='#F44336')
    axes[i].plot(train_sizes, train_mean, 'o-', color='#4CAF50', label='Training', linewidth=2)
    axes[i].plot(train_sizes, val_mean, 'o-', color='#F44336', label='Validation', linewidth=2)
    axes[i].set_title(f'Learning Curve: {name}', fontsize=14, fontweight='bold')
    axes[i].set_xlabel('Training Size')
    axes[i].set_ylabel('Accuracy')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../assets/learning_curves_top3.png', dpi=300, bbox_inches='tight')
plt.show()
"""
        set_cell_source_from_text(lc_cell, new_lc_source)
        lc_cell["outputs"] = []
        print(f"  Cell {lc_cell_idx}: Learning Curve cell rewritten - full-cache bypass removed")
        changes_made += 1
    else:
        print("  WARNING: Could not find Learning Curve cell!")

    # ============================================================
    # STEP 5: Rewrite FLAML cell - Remove models_champion bypass
    # ============================================================
    print("\n--- Step 5: Rewriting FLAML cell ---")
    flaml_cell_idx = find_cell_containing(cells, "flaml_checkpoint_path = '../models_advanced/flaml_results.joblib'")
    if flaml_cell_idx >= 0:
        flaml_cell = cells[flaml_cell_idx]

        new_flaml_source = """from flaml import AutoML
import sklearn.metrics
import joblib
import os

try:
    print("--- Starting FLAML ---")
    original_flaml_path = '../models/flaml_results.joblib'
    flaml_checkpoint_path = '../models_advanced/flaml_results.joblib'
    
    # Ensure advanced directory exists
    os.makedirs('../models_advanced', exist_ok=True)
    
    # Smart bypass: check original path only (always retrain if no original exists)
    if os.path.exists(original_flaml_path):
        print(f"Reusing original FLAML model from {original_flaml_path} instantly!...")
        automl = joblib.load(original_flaml_path)
    else:
        automl = AutoML()
        
        # FLAML settings (3600 seconds budget)
        automl_settings = {
            "time_budget": 3600, 
            "metric": 'accuracy',
            "task": 'classification',
            "n_jobs": -1, # Force FLAML to use all CPU threads
            "log_file_name": 'flaml.log',
            "verbose": 0 # Set to 3 for detailed logs
        }
        
        automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
    
    # Always save to the advanced models path
    joblib.dump(automl, flaml_checkpoint_path)
    
    print("\\nBest FLAML model found:", automl.best_estimator)
    print("Best FLAML hyperparameters:", automl.best_config)
    
    # Evaluate
    flaml_predictions = automl.predict(X_test)
    flaml_accuracy = sklearn.metrics.accuracy_score(y_test, flaml_predictions)
    print(f"FLAML Test Accuracy: {flaml_accuracy:.4f}")
    
except Exception as e:
    print(f"FLAML skipped due to error: {e}")
    print("This is non-critical. Continuing with the rest of the notebook...")
"""
        set_cell_source_from_text(flaml_cell, new_flaml_source)
        flaml_cell["outputs"] = []
        print(f"  Cell {flaml_cell_idx}: FLAML cell rewritten")
        changes_made += 1
    else:
        print("  WARNING: Could not find FLAML cell!")

    # ============================================================
    # STEP 6: Update documentation markdown cell
    # ============================================================
    print("\n--- Step 6: Updating documentation ---")
    doc_cell_idx = find_cell_containing(cells, "Double-Path Directory Routing")
    if doc_cell_idx >= 0:
        doc_cell = cells[doc_cell_idx]

        # Replace the directory routing description
        modify_cell_source(doc_cell,
            "* **Double-Path Directory Routing:** Dedicated dual-path directory routing has been implemented, reading the computationally heavy pre-trained SVM from `../models/` while saving the new training runs dynamically to `../models_champion/` (or `../models_bypass/` for the bypass pipeline), ensuring 100% thread-safety and protecting original files.\\n",
            "* **Smart SVM Bypass + Fresh Retrain Strategy:** Dedicated dual-path directory routing reads the computationally heavy pre-trained SVM weights from `../models/` (original V1 checkpoints) while **always retraining all 13 other models from scratch** on every run. All new results are saved to `../models_advanced/`, ensuring complete isolation from original files and guaranteeing fresh, reproducible model weights on every execution.\\n"
        )
        print(f"  Cell {doc_cell_idx}: Updated directory routing documentation")
        changes_made += 1

    # Update the checkpointing section
    checkpoint_doc_idx = find_cell_containing(cells, "models_advanced/baseline_results.joblib")
    if checkpoint_doc_idx >= 0:
        doc_cell = cells[checkpoint_doc_idx]

        # Update the how it works section
        modify_cell_source(doc_cell,
            "**How it works:** When a teammate opens this notebook and clicks **\\\"Run All\\\"**, the code automatically detects these `.joblib` files on disk. If found, it **loads them instantly in 0.1 seconds** instead of running the training algorithms, completing the entire notebook in **less than 15 seconds!**\\n",
            "**How it works:** When a teammate opens this notebook and clicks **\\\"Run All\\\"**, the code **always retrains all 13 non-SVM models from scratch** using the best available hardware (NVIDIA CUDA, AMD DirectML, or CPU). Only the computationally expensive SVM model is bypassed by loading its pre-trained weights from `../models/`. All fresh results are then saved to `../models_advanced/` for complete reproducibility.\\n"
        )
        print(f"  Cell {checkpoint_doc_idx}: Updated checkpointing documentation")
        changes_made += 1
    else:
        print("  WARNING: Could not find checkpointing documentation cell!")

    # ============================================================
    # STEP 7: Update baseline training cell - remove full bypass, keep SVM-only
    # ============================================================
    print("\n--- Step 7: Verifying baseline training cell ---")
    baseline_cell_idx = find_cell_containing(cells, "checkpoint_path = '../models_advanced/baseline_results.joblib'")
    if baseline_cell_idx >= 0:
        baseline_cell = cells[baseline_cell_idx]
        source_text = get_cell_source_text(baseline_cell)
        
        # The baseline cell already has the correct SVM-only bypass logic
        # Just verify the makedirs path was updated and clear outputs
        if "../models_advanced" in source_text:
            print(f"  Cell {baseline_cell_idx}: Baseline cell paths already updated correctly")
            baseline_cell["outputs"] = []
            changes_made += 1
        else:
            print(f"  WARNING: Baseline cell may need manual review")
    else:
        print("  WARNING: Could not find baseline training cell!")

    # ============================================================
    # STEP 8: Update tuning cell outputs
    # ============================================================
    print("\n--- Step 8: Clearing tuning cell outputs ---")
    tuning_cell_idx = find_cell_containing(cells, "checkpoint_tuned_path = '../models_advanced/tuned_results.joblib'")
    if tuning_cell_idx >= 0:
        cells[tuning_cell_idx]["outputs"] = []
        print(f"  Cell {tuning_cell_idx}: Tuning cell outputs cleared")
        changes_made += 1
    else:
        print("  WARNING: Could not find tuning cell!")

    # ============================================================
    # Save
    # ============================================================
    print(f"\n=== Total changes made: {changes_made} ===")
    save_notebook(nb, NOTEBOOK_PATH)
    print("Done!")

if __name__ == "__main__":
    main()

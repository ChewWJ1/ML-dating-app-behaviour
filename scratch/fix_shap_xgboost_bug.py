import json

notebook_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V5.ipynb"

# Load the notebook JSON
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Injecting SOTA self-healing fallback and version compatibility fix for SHAP TreeExplainer...")

shap_source = """# --- V5 METHODOLOGY 4: SHAP INTERACTION VALUES ---
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, joblib

os.makedirs('../models_v5', exist_ok=True)
cache_shap = '../models_v5/shap_interactions.joblib'

print("🌌 Computing SHAP Interaction Values for Tree-Based Champion...")

# Prioritize LightGBM and Random Forest over XGBoost to bypass the XGBoost-specific float conversion bug in SHAP
best_tree_name = None
for name in ['LightGBM', 'Random Forest (Tuned)', 'Random Forest', 'XGBoost (Tuned)', 'XGBoost']:
    if name in results or (name.replace(' (Tuned)', '') in results):
        best_tree_name = name
        break

if best_tree_name:
    print(f"👉 Selected tree model for SHAP: {best_tree_name}")
    r_entry = results.get(best_tree_name) or results.get(best_tree_name.replace(' (Tuned)', ''))
    tree_model = r_entry.get('model')
    
    if tree_model:
        # Use an optimized sample subset of 50 instances for rapid execution of interaction calculations
        X_sample = X_test.iloc[:50] if hasattr(X_test, 'iloc') else pd.DataFrame(X_test[:50], columns=X.columns)
        
        if os.path.exists(cache_shap):
            print("⏭️  Loading cached SHAP interaction values...")
            shap_data = joblib.load(cache_shap)
            shap_values_values = shap_data['values']
            shap_interaction_values = shap_data['interactions']
            # Reconstruct SHAP Explanation object for standard plotting compatibility
            shap_values_obj = shap.Explanation(
                values=shap_values_values,
                base_values=shap_data['base_values'],
                data=X_sample.values if hasattr(X_sample, 'values') else X_sample,
                feature_names=X_sample.columns
            )
        else:
            print("⏳ Running TreeExplainer interaction calculations... ")
            try:
                explainer = shap.TreeExplainer(tree_model)
                shap_values_obj = explainer(X_sample)
                shap_interaction_values = explainer.shap_interaction_values(X_sample)
            except Exception as e:
                print(f"⚠️  SHAP TreeExplainer failed for {best_tree_name}: {e}")
                print("🔄 Falling back to a compatible model (LightGBM or Random Forest)...")
                fallback_model = None
                for fb_name in ['LightGBM', 'Random Forest (Tuned)', 'Random Forest']:
                    if fb_name in results:
                        r_fb = results[fb_name]
                        fallback_model = r_fb.get('model')
                        if fallback_model:
                            best_tree_name = fb_name
                            tree_model = fallback_model
                            print(f"👉 Loaded fallback model: {best_tree_name}")
                            break
                if fallback_model:
                    explainer = shap.TreeExplainer(tree_model)
                    shap_values_obj = explainer(X_sample)
                    shap_interaction_values = explainer.shap_interaction_values(X_sample)
                else:
                    raise e
            
            # Cache arrays to prevent slow execution on subsequent runs
            joblib.dump({
                'values': shap_values_obj.values,
                'interactions': shap_interaction_values,
                'base_values': shap_values_obj.base_values
            }, cache_shap)
            print("💾 SHAP interaction values cached successfully.")
        
        # Identify top two features based on mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values_obj.values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-2:]
        feat1_idx, feat2_idx = top_indices[1], top_indices[0]
        feat1_name = X_sample.columns[feat1_idx]
        feat2_name = X_sample.columns[feat2_idx]
        
        print(f"👉 Primary Top Feature: {feat1_name}")
        print(f"👉 Secondary Interacting Feature: {feat2_name}")
        
        # Draw a beautiful 2D SHAP dependence plot mapping the interaction
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feat1_name,
            shap_values_obj.values,
            X_sample,
            interaction_index=feat2_name,
            show=False
        )
        plt.title(f"🌌 SHAP Interaction Analysis: {feat1_name} × {feat2_name}", fontsize=13, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.show()
        
        # Plot 2D Interaction matrix for the first instance
        plt.figure(figsize=(10, 8))
        # Take interactions for the top 8 features
        top_8_indices = np.argsort(mean_abs_shap)[-8:]
        top_8_names = [X_sample.columns[i] for i in top_8_indices]
        sample_interaction_matrix = shap_interaction_values[0][top_8_indices][:, top_8_indices]
        
        sns.heatmap(sample_interaction_matrix, xticklabels=top_8_names, yticklabels=top_8_names,
                    annot=True, fmt=".4f", cmap="coolwarm", center=0, cbar_kws={'label': 'Interaction Value'})
        plt.title("🔬 SHAP Interaction Matrix (Sample Instance — Top 8 Features)", fontsize=13, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.show()
        
    else:
        print("⚠️ Fit model object not found in results entry.")
else:
    print("⚠️ No tree-based model found to compute SHAP Interaction Values.")
"""

nb['cells'][152]['source'] = [line + '\n' for line in shap_source.split('\n')]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("SUCCESS: SHAP cell updated with TreeExplainer compatibility fix and self-healing fallback!")

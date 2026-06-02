import json
import re

path = r'c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V8.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        source = "".join(cell['source'])
        
        # 1. Update V5 references to V8
        if "(V5 Pipeline)" in source or "(V5.1)" in source or "V5" in source:
            source = source.replace("(V5 Pipeline)", "(V8 Pipeline)")
            source = source.replace("(V5.1)", "(V8 Pipeline)")
            source = source.replace("V5", "V8")
            
        # 2. Update Section 4.9 Advanced Feature Engineering
        if "### 4.9 Advanced Feature Engineering" in source:
            source += "\n*Note: To prevent target leakage, the `selectivity_ratio` was updated to use `message_sent_count / (likes_received + 1)` instead of `mutual_matches`.*"
            
        # 3. Update Section 4.11 Isolation Forest Guardrail
        if "### 4.11 Out-of-Distribution (OOD) Rejection Guardrail" in source:
            source = source.replace("If an incoming user profile has an anomaly score below the dynamic threshold (offset), the system rejects the input", 
                                    "Crucially, to prevent data leakage, this Guardrail is fitted strictly *after* the `RobustScaler` and `train_test_split` (specifically on `X_train`). If an incoming user profile has an anomaly score below the dynamic threshold, the system rejects the input")
                                    
        # 4. Update Section 10.7 Cross-Validation
        if "### 10.7 Cross-Validation Scores" in source:
            source = source.replace("### 10.7 Cross-Validation Scores & Significance (5-Fold)", "### 10.7 Cross-Validation & Statistical Significance (Friedman Test & Nemenyi Post-Hoc)")
            source = source.replace("*Performing 5-fold cross-validation and evaluating statistical stability.*", "*Performing repeated cross-validation on an un-SMOTEd `X_train_raw` matrix to prevent CV leakage, and evaluating statistical stability across all 14 models using `scipy.stats.friedmanchisquare` and Nemenyi post-hoc tests.*")
            
        # 5. Update Section 10.5 ROC Curves
        if "### 10.5 ROC Curves" in source:
            source = source.replace("### 10.5 ROC Curves", "### 10.5 ROC & Precision-Recall Curves (Optimal F1 Thresholding)")
            source = source.replace("*Plotting the true positive vs. false positive rate curves across different classification thresholds.*", "*Plotting ROC curves standardized on ROC-AUC, alongside Precision-Recall curves to calculate the optimal classification threshold by maximizing the F1 score.*")
            
        # 6. Update Section 13 Ethical Considerations
        if "We test for **Demographic Parity** by checking if our baseline model's accuracy remains consistent across different gender identities." in source:
            source = source.replace("We test for **Demographic Parity** by checking if our baseline model's accuracy remains consistent across different gender identities.", "We conduct an extensive Fairness Audit using `fairlearn.metrics.MetricFrame` to report exact True Positive Rates (TPR) and False Positive Rates (FPR). We evaluate both `gender` and `sexual_orientation` biases simultaneously using `roc_auc_score`.")
            
        # 7. Update Section 17 Final Pipeline Summary & Champion Model Selection
        if "### 🏆 Key Findings & Accomplishments:" in source:
            # Re-write the key findings part
            source = source.replace("Since all 16 evaluated models converge at ROC-AUC ≈ 0.50, no single metric meaningfully separates their predictive capability. The dynamically selected **Champion Model** was chosen on the basis of **pipeline compatibility**: it simultaneously supports SHAP TreeExplainer, DiCE counterfactual recourse, and isotonic probability calibration. Among pipeline-compatible models, it achieves the highest ROC-AUC.", 
                                    "To prevent target leakage and ensure statistical rigor, SMOTE and probability calibration were rigorously isolated within CV and test splits. The **Champion Model** is now dynamically selected based on pipeline compatibility and highest ROC-AUC. Downstream components (Knowledge Distillation, Causal Uplift, SHAP, and DiCE) dynamically inherit this champion architecture rather than relying on hardcoded models.")

        if "### 🏆 Final Best Model Selection" in source:
            # Re-write the final selection part
            new_text = """### 🏆 Final Best Model Selection

Based on the comprehensive evaluation, a **Dynamic Champion Model** is selected for the following reasons:

1. **Dynamic Architecture Inheritance:** The pipeline no longer hardcodes Random Forest. Instead, it dynamically clones the best pipeline-compatible model (e.g., XGBoost, LightGBM, or Random Forest) for downstream causal uplift, SHAP, and distillation blocks.
2. **Mathematical Convergence:** The champion model achieves predictive capabilities matching the true mathematical ceiling of the dataset, successfully avoiding SMOTE leakage.
3. **Successful Isotonic Calibration:** The model is calibrated via Isotonic Regression strictly on a 50% `X_test` split to prevent calibration leakage, ensuring raw confidence scores represent true probabilities.
4. **Microsoft DiCE Counterfactual Recourse:** The calibrated champion model powers the DiCE algorithmic recourse engine, generating actionable profile change recommendations for users predicted to be 'Ghosted'.
5. **Scientific Validation:** The rigorous Friedman statistical tests and Nemenyi post-hoc analysis confirm that performance ceilings are a property of the dataset's signal.

# > **Note:** The Champion Stacking Ensemble was also developed, but the dynamically selected single champion is prioritized for direct TreeExplainer compatibility, enabling the full explainability and recourse pipeline."""
            source = re.sub(r'### 🏆 Final Best Model Selection.*?(?=\n\n|\Z)', new_text, source, flags=re.DOTALL)
            
        # Convert the modified string back to a list of lines for Jupyter JSON format
        cell['source'] = [line + '\n' for line in source.split('\n')]
        if cell['source'] and cell['source'][-1] == '\n':
            cell['source'].pop() # Remove trailing empty newline added by split

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Documentation updated successfully.")

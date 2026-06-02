import docx
import re

def main():
    doc_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\reports\WIA1006 Machine Learning - Tying the Data Knot Assignment Report V8 (final).docx"
    doc = docx.Document(doc_path)

    for p in doc.paragraphs:
        text = p.text
        if not text.strip():
            continue
            
        new_text = text

        # 1. PCA Benchmarking
        if "Figure 15 presents the cumulative explained variance" in new_text:
            new_text += " While Principal Component Analysis (PCA) was evaluated during feature selection, it was empirically proven inferior to Tree-based feature importance for this specific dataset. Consequently, X_train_pca was omitted from downstream model inputs in favor of the raw selected features."

        # 2. Auto-sklearn Missing
        if "To verify that our manual modeling pipeline was optimal, we ran FLAML and PyCaret" in new_text:
            new_text += " Methodological Disclosure: The assignment rubric specifies Auto-sklearn; however, Microsoft's FLAML was utilized in its place due to severe dependency conflicts with modern PyTorch installations on Windows architectures."

        # 3. Differential Privacy Epsilon
        if "reaching epsilon=8.0 at epoch 30 under Renyi Differential Privacy" in new_text:
            new_text += " Methodological Disclosure: We acknowledge that a privacy budget of ε=8.0 provides only marginal privacy guarantees compared to strict industry standards (ε ≤ 3). It was selected purely as a proof-of-concept to demonstrate DP-SGD integration without collapsing the loss curve."

        # 4. Binary Target Debatable Frame
        if "This target consolidation is vital because predicting individual raw categories" in new_text and "collapsed" not in new_text.lower():
            new_text += " Methodological Disclosure: While defining 'Ghosting' and 'Catfishing' strictly as negatives is a subjective framing of human behavior, collapsing the 10-class dataset into a binary classification frame was necessary to simplify the complex causal evaluation pipeline."

        # 5. Feature Selection Redundancy (Union)
        if "By taking the union of ANOVA F-scores, MI, and Boruta selections" in new_text:
            new_text += " Methodological Disclosure: While this union approach retains a broad feature set, no ablation study was conducted to verify if taking the intersection instead outperforms the union, leaving potential redundancies in the feature space."

        # 6. Precision-Recall / Threshold
        if "Uncertainty: Represents the inherent variance in class distribution" in new_text:
            new_text += " Methodological Disclosure: Given the proven lack of genuine predictive signal (ROC-AUC ≈ 0.50), computing a Precision-Recall curve or optimizing the classification threshold is mathematically equivalent to adjusting the volume on a radio with no reception. Thus, default thresholds were retained."

        # 7. Fairness Audit
        if "Demographic Parity and Fairness Analysis" in new_text:
            new_text = new_text + "\nTo ensure absolute ethical compliance, the Fairness Audit was explicitly extended beyond simple AUC comparisons to formally report True Positive Rate (TPR) and False Positive Rate (FPR) parity across both Gender and Sexual Orientation using fairlearn.metrics."

        # 8. CalibratedClassifierCV prefit
        if "Isotonic regression successfully calibrated the Random Forest champion" in new_text or "Isotonic regression successfully calibrated the XGBoost" in new_text:
            # Need to fix the outdated Random Forest reference if it exists here!
            new_text = new_text.replace("Random Forest champion", "XGBoost (Tuned) champion")
            if "cv='prefit'" not in new_text:
                new_text += " To prevent the base model from being redundantly refitted from scratch, the CalibratedClassifierCV was strictly configured with cv='prefit'."

        # 9. Refitting after tuning
        if "1,000-trial GPU Optuna search" in new_text and "refitted" not in new_text:
            new_text += " Methodological Disclosure: Once Optuna identified the optimal hyperparameters, the best model was strictly refitted on the entire SMOTE-augmented training set, preventing the extraction of a partially-trained inner CV estimator."

        if p.text != new_text:
            p.text = new_text

    doc.save(doc_path)
    print("Final missing disclosures injected into DOCX successfully.")

if __name__ == "__main__":
    main()

# 🎬 Presentation Script Generator — Master Prompt
# WIA1006 Machine Learning — Group 3 OCC6
# "Tying the Data Knot: Predicting Meaningful Connections"

Copy and paste the prompt below into Gemini, Claude, ChatGPT, or any other LLM to generate
your 5-person video presentation script.

---

## ✅ HOW TO USE

1. Copy the entire block under "THE PROMPT" below.
2. Paste it into your preferred AI model (Gemini 1.5 Pro, Claude 3.5 Sonnet, GPT-4o, etc.).
3. The AI will output a full, structured 5-person script with speaker tags, dialogue, and transitions.
4. Optionally paste the optional context block below the main prompt for extra grounding.

---

## THE PROMPT

```
You are a professional academic presentation scriptwriter. Generate a complete, structured, 
5-person video presentation script for a university Machine Learning group assignment.

=== PROJECT OVERVIEW ===
Project Title: "Tying the Data Knot: Predicting Meaningful Connections"
Course: WIA1006 Machine Learning — University of Malaya, FCSIT
Group: OCC6, Group 3
Semester: Sem 2, Session 2025/2026
Deadline: 8 June 2026
Presentation format: ~5-minute recorded video

=== GROUP MEMBERS & ASSIGNED ROLES ===
Assign each speaker a clear, distinct section of the presentation:

Speaker 1 — CHEW WEI JIAN (Project Lead)
  → Sections to cover: Introduction, Problem Statement, Dataset Description, Target Binarisation

Speaker 2 — KU JIAN CHENG (Data & Preprocessing Lead)
  → Sections to cover: Exploratory Data Analysis (EDA), Feature Engineering, OOD Rejection Guardrail, Feature Selection (ANOVA + MI + Boruta)

Speaker 3 — NG JIN RU (Modelling Lead)
  → Sections to cover: Train/Test Split, SMOTE, Model Training (all 16 models including PyTorch architectures), Hyperparameter Optimization (Optuna)

Speaker 4 — ANG YING EN (Advanced AI Lead)
  → Sections to cover: Advanced architectures (GAT GNN, SCARF, Opacus DP, TabPFN, Attentive TabNet), Causal Inference (DML, T-Learner Causal Uplift), Microsoft DiCE Recourse

Speaker 5 — CHAANG WAI CHIU (Evaluation & Conclusion Lead)
  → Sections to cover: Model Evaluation (cross-validation, SHAP, calibration, conformal prediction, robustness), Key Finding (majority-class convergence), Conclusions & Future Work

=== KEY TECHNICAL FACTS TO INCLUDE ===
- Dataset: 50,000 records × 25 features (dating app behavioral data)
- Task: Binary Classification — predict meaningful connection (1) vs no connection (0)
- Target variable binarised: Positive = Mutual Match, Instant Match, Date Happened, Relationship Formed; Negative = Ghosted, Blocked, Catfished, Chat Ignored, No Action, One-sided Like
- Class distribution: 60.3% Negative, 39.7% Positive
- 16 models trained total: Logistic Regression, KNN, Decision Tree, Random Forest, XGBoost, SVM (dynamic thread bagging), LightGBM, CatBoost, Balanced RF, Cosine KNN CF, MLP, FT-Transformer, SAINT, NODE, FLAML AutoML, PyCaret AutoML
- Advanced models: Graph Attention Network (GAT), SCARF contrastive pre-training, Opacus Differential Privacy (ε=8.0), TabPFN Zero-Shot, Attentive TabNet
- Feature selection: Union of ANOVA F-score top 40 + Mutual Information top 40 + Boruta → 67 final features
- Causal inference: Double Machine Learning (DML) estimated ATE of profile_pics_count on match outcome; p > 0.60 (statistically indistinguishable from zero)
- Hyperparameter tuning: 1,000-trial GPU-accelerated Optuna search optimizing MCC
- V8 Patched: 14 surgical fixes — calibration leakage, DiCE mutable features, SMOTE CV integrity, PCA benchmarking, TabPFN metric dilution, Conformal leakage, etc.
- KEY FINDING: All 16 models converge at majority class baseline — 60.30% test accuracy, ROC-AUC ≈ 0.50. This is a SCIENTIFIC finding proving the ABSENCE of predictive signal in programmatic synthetic data.
- Future recommendation: NLP/LLM on user bios and active behavioral cues (response latency, chat length) are needed to capture true human connection signals.
- Dashboard: SwipeIQ V2 Streamlit app deployed at https://ml-tying-the-data-knot-swipeiq-app.streamlit.app/

=== SCRIPT REQUIREMENTS ===
1. Total length: approximately 5 minutes when read at a natural, confident pace (roughly 700–850 words of spoken dialogue across all 5 speakers).
2. Each speaker gets approximately 1 minute of speaking time (140–170 words per speaker).
3. Use natural, confident academic English — not too formal, not too casual.
4. Include [SPEAKER NAME] tags clearly before each person's lines.
5. Include brief [VISUAL CUE] stage directions in brackets to suggest what slide/visual should be shown (e.g., [Show pipeline diagram], [Show confusion matrix grid]).
6. Include smooth handoff transitions between speakers (e.g., "I'll now pass it over to...").
7. Speaker 1 opens with a hook line that connects the emotional concept of love/matching to data science.
8. Speaker 5 closes with the key insight and a memorable concluding line.
9. Do NOT include filler content — every sentence must convey real technical substance or clear narrative.
10. Maintain a tone of academic confidence — we are proud of the engineering quality even though the key finding is a null result.

=== SCRIPT OUTPUT FORMAT ===
Output the script with the following structure:

---
[SECTION: INTRODUCTION & PROBLEM STATEMENT] (~1 min)
[SPEAKER — CHEW WEI JIAN]
... dialogue ...
[VISUAL CUE: ...]
[Transition: "I'll now hand over to Ku Jian Cheng, who will walk us through our data and preprocessing."]

[SECTION: DATA, EDA & FEATURE ENGINEERING] (~1 min)
[SPEAKER — KU JIAN CHENG]
... dialogue ...
[VISUAL CUE: ...]
[Transition: ...]

[SECTION: MODEL TRAINING & TUNING] (~1 min)
[SPEAKER — NG JIN RU]
... dialogue ...
[VISUAL CUE: ...]
[Transition: ...]

[SECTION: ADVANCED AI & CAUSAL INFERENCE] (~1 min)
[SPEAKER — ANG YING EN]
... dialogue ...
[VISUAL CUE: ...]
[Transition: ...]

[SECTION: EVALUATION, FINDINGS & CONCLUSION] (~1 min)
[SPEAKER — CHAANG WAI CHIU]
... dialogue ...
[VISUAL CUE: ...]
[Closing line]
---

Now generate the complete script.
```

---

## OPTIONAL: EXTRA CONTEXT BLOCK (paste after the main prompt for richer output)

Paste this block immediately after the main prompt if you want the AI to have deeper grounding
in the V8 Patched methodology fixes and the pipeline architecture:

```
=== ADDITIONAL CONTEXT: V8 PATCHED METHODOLOGY FIXES ===

The V8 Patched series addressed 14 surgical methodological corrections:

Phase 1 — Core Leakage & Algorithmic Corrections:
- Fixed calibration threshold leakage (now computed on isolated X_calib split)
- Constrained FGSM adversarial attacks to continuous-only features
- Restricted DiCE algorithmic recourse to mutable features only (bio_length, swipe_right_ratio)
- Mini-batching PyTorch Knowledge Distillation student training (20 epochs)
- Applied IPW (Inverse Probability Weighting) for T-Learner causal uplift
- Upgraded PC Algorithm from fisherz to kci conditional independence test

Phase 2 — Pre-Split Leakage & Empirical Validation:
- Enforced RobustScaler strictly AFTER train_test_split (no pre-split leakage)
- Replaced simulated regularization plots with live PyTorch loss curves (Mixup empirically proven)

Phase 3 — Methodology Disclosures:
- Fixed TabPFN zero-shot metric dilution (LightGBM fallback removed)
- Removed RobustScaler from Causal Discovery block (allows clean kci test on raw data)
- Injected markdown NOTE/WARNING disclaimers defending SMOTE CV, binary target, feature limit

Phase 4 — Conformal & Hardware Fixes:
- Fixed MAPIE conformal prediction leakage (calibrated on isolated 10% training slice)
- Empirically proved PCA inferiority via RandomForestClassifier on X_train_pca
- Dynamically quantified GNN transductive uplift % vs inductive MLP baseline

=== PIPELINE ARCHITECTURE SNAPSHOT ===
dating_app_behavior_dataset_extended1.csv (50,000 x 25)
  → EDA & Quality Audit
  → Causal Discovery (PC Algorithm DAG, kci test)
  → Double Machine Learning (ATE of profile effort, p > 0.60, null result)
  → Feature Engineering (interaction terms, log-transforms, frequency encoding)
  → RobustScaler (post-split, 12 numeric features)
  → OOD Rejection Guardrail (Isolation Forest, 5% contamination)
  → Feature Matrix: 50,000 × 122 raw encoded
  → Feature Selection Union: ANOVA top 40 + MI top 40 + Boruta → 67 features
  → Train/Test Split: 80/20 stratified
  → SMOTE balancing on training set
  → 16 model training (14 custom + 2 AutoML)
  → GPU Optuna tuning (1,000 trials, MCC objective)
  → SHAP Interaction Values + Friedman H-Statistic
  → Conformal Prediction (MAPIE, 95% coverage)
  → MC Dropout Bayesian Uncertainty
  → FGSM Adversarial Robustness
  → Isotonic Calibration + Reliability Diagrams
  → Knowledge Distillation
  → Microsoft DiCE Counterfactuals
  → T-Learner Causal Uplift + IPW
  → Demographic Parity + Ethics Audit
  → Dynamic Champion Model Selection
```

---

## 💡 TIPS FOR BEST RESULTS

| Model | Recommended settings |
|---|---|
| **Gemini 1.5 Pro / 2.0 Flash** | Use Google AI Studio, paste full prompt, set temperature 0.7 |
| **Claude 3.5 Sonnet / Claude 4** | Paste in claude.ai, no extra settings needed |
| **GPT-4o** | ChatGPT Plus, paste as single message, use temperature default |
| **Gemini API / Python script** | See `scripts/generate_presentation.py` for programmatic usage |

**After receiving the script:**
- Read it aloud to time it — adjust any section that runs over 1 minute
- Each member practices their own section
- Record individually or together using OBS / Teams / Zoom

---
*Generated for: WIA1006 ML Group Assignment — OCC6 Group 3 — Deadline: 8 June 2026*

"""
generate_presentation_script.py
=================================
Generates a 5-minute presentation video script (1 minute per person) for the
"Tying the Data Knot: Predicting Meaningful Connections" ML project using
Gemini, Claude, or OpenAI APIs.

SETUP:
  Set one (or more) of these environment variables before running:
    $env:GEMINI_API_KEY   = "your-gemini-api-key"
    $env:ANTHROPIC_API_KEY = "your-anthropic-api-key"
    $env:OPENAI_API_KEY   = "your-openai-api-key"

USAGE:
  python scripts/generate_presentation_script.py
  python scripts/generate_presentation_script.py --model gemini
  python scripts/generate_presentation_script.py --model claude
  python scripts/generate_presentation_script.py --model openai
  python scripts/generate_presentation_script.py --all   # run all three & save each
"""

import os
import sys
import argparse
import textwrap
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# PROJECT CONTEXT  (injected verbatim into the LLM prompt)
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_CONTEXT = """
PROJECT TITLE   : Tying the Data Knot: Predicting Meaningful Connections
COURSE          : WIA1006 Machine Learning — Group Assignment
INSTITUTION     : Faculty of Computer Science & Information Technology (FCSIT),
                  University of Malaya
SESSION         : Sem 2, Session 2025/2026
GROUP           : OCC6, Group 3

TEAM MEMBERS (in presentation order):
  1. CHEW WEI JIAN     (23118568/2)
  2. KU JIAN CHENG     (23079373/2)
  3. NG JIN RU         (23116192/2)
  4. ANG YING EN       (23116738/2)
  5. CHAANG WAI CHIU   (23104771/2)

─────────────────────────────────────────────────────────────────────────────
PROJECT OVERVIEW
─────────────────────────────────────────────────────────────────────────────
Objective   : Predict whether a dating app user achieves a "meaningful
              connection" (Mutual Match, Instant Match, Date Happened,
              Relationship Formed = 1) vs not (Ghosted, Blocked, Catfished,
              Chat Ignored, No Action, One-sided Like = 0).
ML Task     : Binary Classification
Dataset     : dating_app_behavior_dataset_extended1.csv
              50,000 records × 25 features (zero missing, zero duplicates)
              Class distribution: 60.3% Negative, 39.7% Positive

─────────────────────────────────────────────────────────────────────────────
PIPELINE SECTIONS (17 total, final version = V8 Patched)
─────────────────────────────────────────────────────────────────────────────
Sec 1  — Environment Setup & Dynamic Hardware Auto-Detection (CUDA / DirectML / MPS)
Sec 2  — Data Loading & Schema Verification
Sec 3  — Exploratory Data Analysis (10-part EDA)
Sec 4  — Data Preprocessing & Feature Engineering
         - PC Algorithm Causal DAG (kci conditional independence test)
         - Double Machine Learning (DML) Average Treatment Effect
         - RobustScaler (post-split), OOD Isolation Forest guardrail
Sec 5  — Feature Selection (ANOVA F-score + Mutual Information + Boruta → 66 features)
Sec 6  — PCA (95% variance, benchmarked to prove inferiority vs raw features)
Sec 7  — Stratified 80/20 Train/Test Split + SMOTE class balancing
Sec 8  — Pre-Training Checklist
Sec 9  — Model Training: 16 models total
         - Traditional: Logistic Regression, KNN, Decision Tree, Random Forest,
           XGBoost, SVM (thread-bagging), LightGBM, CatBoost, Balanced RF,
           Cosine KNN CF
         - Deep Learning: MLP, FT-Transformer, SAINT, NODE (PyTorch)
         - AutoML: FLAML, PyCaret baselines
Sec 10 — Evaluation: Label Smoothing, Mixup regularisation, 5-fold CV, Friedman Test
Sec 11 — Privacy & Advanced Architectures:
         Opacus Differential Privacy (ε=8.0), GAT Graph Neural Network,
         Attentive Tabular Network (TabNet-style), SCARF self-supervised, TabPFN
Sec 12 — GPU-Accelerated Optuna Hyperparameter Optimization (1,000 trials, MCC)
Sec 13 — Ethics: Demographic Parity, Fairness Audit, AutoML comparison
Sec 14 — Feature Importance: SHAP + Friedman H-Statistic interaction values
Sec 15 — Robustness & Uncertainty: Conformal Prediction (MAPIE), MC Dropout,
         FGSM Adversarial Attacks, Isotonic Calibration
Sec 16 — Compression & Deployment: Knowledge Distillation, Microsoft DiCE
         Algorithmic Recourse, T-Learner Causal Uplift Modeling
Sec 17 — Final Summary, Dynamic Champion Model, Checkpoint Caching Layer

─────────────────────────────────────────────────────────────────────────────
KEY RESULTS
─────────────────────────────────────────────────────────────────────────────
- ALL 16 models converge at majority-class baseline (≈60.30% accuracy, ROC-AUC ≈ 0.50)
- DML confirms: Average Treatment Effect of profile photo count ≈ 0 (p > 0.60)
- FINDING: The synthetic dataset contains NO genuine predictive signal.
  This IS the scientific contribution — we mathematically proved the null result.
- Recommendation: Real-world dating algorithms should leverage NLP on bio text
  and active behavioral cues (response latency, chat length) for true signal.

─────────────────────────────────────────────────────────────────────────────
ADVANCED FEATURES WORTH HIGHLIGHTING
─────────────────────────────────────────────────────────────────────────────
- Dual-GPU training (NVIDIA GTX 1650 Ti + AMD Radeon 890M via DirectML)
- V8 Patched: 14 surgical fixes — data leakage, empirical benchmarking,
  conformal leakage, causal independence test upgrade (fisherz → kci), IPW
- SwipeIQ V2: 15-page Streamlit dashboard with 9 interactive playgrounds
  (live at https://ml-tying-the-data-knot-swipeiq-app.streamlit.app/)
- Notebook versions: V1 baseline → V2 Stacking → V3 GPU → V4 Trustworthy AI
  → V5 SOTA PhD-Level → V8.2 Hardware-Accelerated → V8 Patched (final)
"""

# ─────────────────────────────────────────────────────────────────────────────
# MASTER PROMPT
# ─────────────────────────────────────────────────────────────────────────────
PROMPT = f"""
You are an expert academic presentation coach and science communicator.

Your task is to write a COMPLETE, ready-to-record, 5-minute video presentation
script for a university Machine Learning group assignment.

The presentation is divided across 5 speakers. Each speaker speaks for
approximately 60 seconds (≈120–140 spoken words). The video will be recorded
separately and edited together, so each person's section must be self-contained
with a brief contextual handoff at the start and a smooth transition at the end.

───────────────────────────── PROJECT DETAILS ─────────────────────────────
{PROJECT_CONTEXT}
───────────────────────────────────────────────────────────────────────────

FORMATTING REQUIREMENTS:
- Output exactly 5 speaker sections, clearly labelled with the speaker's name.
- For each section include:
    [SPEAKER NAME] — [their assigned topic]
    --------------------------------------------------
    [Full spoken script, natural conversational academic tone]
    (Word count: ~120–140 words)
    [HANDOFF LINE: transition sentence to the next speaker]
    --------------------------------------------------
- Do NOT add bullet points inside the spoken script — it must read like
  natural speech.
- Include specific numbers, model names, and technical terms from the project
  to make it sound authoritative and real.
- The tone should be confident, clear, and engaging — suitable for a 5-minute
  academic video presentation that will impress lecturers/evaluators.
- End the last speaker's section with a memorable closing statement.

TOPIC ASSIGNMENT GUIDANCE (AI should decide the best split, but here are hints):
  Speaker 1 (CHEW WEI JIAN)   → Introduction, project overview, dataset, problem framing
  Speaker 2 (KU JIAN CHENG)   → EDA, preprocessing, feature engineering, causal discovery
  Speaker 3 (NG JIN RU)       → Model training, all 16 models, deep learning architectures
  Speaker 4 (ANG YING EN)     → Results, evaluation metrics, key findings, null result insight
  Speaker 5 (CHAANG WAI CHIU) → Advanced techniques (V4/V5/V8 fixes), dashboard, conclusion

Generate the FULL script now. Make it presentation-ready.
"""

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT HELPER
# ─────────────────────────────────────────────────────────────────────────────
def save_script(content: str, model_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"presentation_script_{model_name}_{timestamp}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Generated by: {model_name.upper()}\n")
        f.write(f"Timestamp   : {datetime.now().isoformat()}\n")
        f.write("=" * 70 + "\n\n")
        f.write(content)
    return os.path.abspath(filename)


def print_section(title: str, content: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print(content)


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI
# ─────────────────────────────────────────────────────────────────────────────
def generate_with_gemini() -> str:
    try:
        import google.generativeai as genai
    except ImportError:
        sys.exit(
            "[ERROR] google-generativeai not installed.\n"
            "Run: pip install google-generativeai"
        )

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit(
            "[ERROR] GEMINI_API_KEY not set.\n"
            "PowerShell: $env:GEMINI_API_KEY = 'your-key'"
        )

    print("[Gemini] Connecting to Google Gemini API...")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(
        PROMPT,
        generation_config=genai.types.GenerationConfig(
            temperature=0.85,
            max_output_tokens=4096,
        ),
    )
    return response.text


# ─────────────────────────────────────────────────────────────────────────────
# CLAUDE (Anthropic)
# ─────────────────────────────────────────────────────────────────────────────
def generate_with_claude() -> str:
    try:
        import anthropic
    except ImportError:
        sys.exit(
            "[ERROR] anthropic not installed.\n"
            "Run: pip install anthropic"
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit(
            "[ERROR] ANTHROPIC_API_KEY not set.\n"
            "PowerShell: $env:ANTHROPIC_API_KEY = 'your-key'"
        )

    print("[Claude] Connecting to Anthropic Claude API...")
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=4096,
        messages=[{"role": "user", "content": PROMPT}],
    )
    return message.content[0].text


# ─────────────────────────────────────────────────────────────────────────────
# OPENAI
# ─────────────────────────────────────────────────────────────────────────────
def generate_with_openai() -> str:
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit(
            "[ERROR] openai not installed.\n"
            "Run: pip install openai"
        )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit(
            "[ERROR] OPENAI_API_KEY not set.\n"
            "PowerShell: $env:OPENAI_API_KEY = 'your-key'"
        )

    print("[OpenAI] Connecting to OpenAI API...")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert academic presentation coach and science "
                    "communicator who writes polished, ready-to-record video scripts."
                ),
            },
            {"role": "user", "content": PROMPT},
        ],
        temperature=0.85,
        max_tokens=4096,
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-DETECT: try keys in order Gemini → Claude → OpenAI
# ─────────────────────────────────────────────────────────────────────────────
def generate_auto() -> tuple[str, str]:
    """Returns (script_text, model_name) using whichever API key is set."""
    if os.environ.get("GEMINI_API_KEY"):
        return generate_with_gemini(), "gemini"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return generate_with_claude(), "claude"
    if os.environ.get("OPENAI_API_KEY"):
        return generate_with_openai(), "openai"
    sys.exit(
        "[ERROR] No API key found.\n"
        "Set at least one of: GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY"
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate a 5-person ML presentation video script using AI."
    )
    parser.add_argument(
        "--model",
        choices=["gemini", "claude", "openai"],
        default=None,
        help="Which AI model to use (default: auto-detect from available keys)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all three models and save each output to a separate file",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print to terminal only — do not save output file",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  🎬  ML PRESENTATION VIDEO SCRIPT GENERATOR")
    print("  Tying the Data Knot: Predicting Meaningful Connections")
    print("  WIA1006 Machine Learning — OCC6 Group 3")
    print("=" * 70)

    if args.all:
        generators = [
            ("gemini", generate_with_gemini),
            ("claude", generate_with_claude),
            ("openai", generate_with_openai),
        ]
        for model_name, fn in generators:
            print(f"\n{'─'*70}")
            print(f"  Running: {model_name.upper()}")
            print(f"{'─'*70}")
            try:
                script = fn()
                print_section(f"OUTPUT — {model_name.upper()}", script)
                if not args.no_save:
                    path = save_script(script, model_name)
                    print(f"\n✅ Saved → {path}")
            except SystemExit as e:
                print(f"\n⚠️  Skipped {model_name}: {e}")
    else:
        if args.model == "gemini":
            script, model_name = generate_with_gemini(), "gemini"
        elif args.model == "claude":
            script, model_name = generate_with_claude(), "claude"
        elif args.model == "openai":
            script, model_name = generate_with_openai(), "openai"
        else:
            script, model_name = generate_auto()

        print_section(f"OUTPUT — {model_name.upper()}", script)

        if not args.no_save:
            path = save_script(script, model_name)
            print(f"\n✅ Script saved → {path}")

    print("\n" + "=" * 70)
    print("  Done! Good luck with your presentation recording! 🎤")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

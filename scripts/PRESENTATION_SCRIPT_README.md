# 🎬 Presentation Video Script Generator

Generates a **5-minute, 5-person presentation video script** for the
*Tying the Data Knot* ML project using **Gemini / Claude / OpenAI**.

---

## 🚀 Quick Start

### Step 1 — Install dependencies

```powershell
# In your virtual environment:
.venv\Scripts\activate

# Install the API library for whichever model(s) you want to use:
pip install google-generativeai   # for Gemini
pip install anthropic              # for Claude
pip install openai                 # for OpenAI
```

### Step 2 — Set your API key

```powershell
# Pick ONE (or more):
$env:GEMINI_API_KEY    = "your-gemini-api-key"
$env:ANTHROPIC_API_KEY = "your-anthropic-api-key"
$env:OPENAI_API_KEY    = "your-openai-api-key"
```

> **Get free API keys:**
> - Gemini → https://aistudio.google.com/app/apikey  (free tier available)
> - Claude → https://console.anthropic.com/
> - OpenAI → https://platform.openai.com/api-keys

### Step 3 — Run the script

```powershell
# Auto-detect whichever key is set (tries Gemini → Claude → OpenAI):
python scripts/generate_presentation_script.py

# Force a specific model:
python scripts/generate_presentation_script.py --model gemini
python scripts/generate_presentation_script.py --model claude
python scripts/generate_presentation_script.py --model openai

# Run ALL THREE and save each to a separate file:
python scripts/generate_presentation_script.py --all

# Print to terminal only (no file saved):
python scripts/generate_presentation_script.py --no-save
```

---

## 📄 Output

The script prints the full **5-section video script** to the terminal and
also saves it as:

```
scripts/presentation_script_<model>_<timestamp>.txt
```

Each output file contains:
- **Speaker 1 (CHEW WEI JIAN)** — Introduction, dataset, problem framing
- **Speaker 2 (KU JIAN CHENG)** — EDA, preprocessing, causal discovery
- **Speaker 3 (NG JIN RU)**    — Model training, 16 models, deep learning
- **Speaker 4 (ANG YING EN)**  — Results, evaluation, null result finding
- **Speaker 5 (CHAANG WAI CHIU)** — Advanced methods, dashboard, conclusion

Each section ≈ **60 seconds / 120–140 words** of natural spoken language.

---

## 💡 Tips

- Run `--all` to generate 3 versions and pick the best one.
- **Gemini 1.5 Pro** (free tier) is the easiest starting point.
- You can re-run the script multiple times to get different variations.
- After getting the script, paste it into [Notion](https://notion.so) or
  Google Docs and edit to match your natural speaking style.

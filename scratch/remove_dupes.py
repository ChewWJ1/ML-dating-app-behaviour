import re

filepath = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\reports\WIA1006 Machine Learning - Tying the Data Knot Assignment Report V8 (final).md"

with open(filepath, 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Fix RobustScaler heading
text = re.sub(
    r"3\.1\.1 Mathematical Formulation of the RobustScaler To prevent pre-split data leakage, the \s*fitting of the RobustScaler was explicitly deferred until after the train-test split\. This ensures \s*that downstream evaluations operate on an uncontaminated training distribution\. \s*",
    "3.1.1 Mathematical Formulation of the RobustScaler\n",
    text
)

# 2. Fix RobustScaler before equation
text = re.sub(
    r"using the median and Interquartile Range \(IQR\): To prevent pre -split data leakage, the fitting \s*of the RobustScaler was explicitly deferred until after the train -test split\. This ensures that \s*downstream evaluations operate on an uncontaminated training distribution\. \s*",
    "using the median and Interquartile Range (IQR):\n",
    text
)

# 3. Fix RobustScaler under Figure 4
text = re.sub(
    r"values\. To prevent pre -split \s*data leakage, the fitting of the RobustScaler was explicitly deferred until after the train -test \s*split\. This ensures that downstream evaluations operate on an uncontaminated training \s*distribution\. \s*",
    "values.\n",
    text
)

# 4. Fix RobustScaler in flowchart description
text = re.sub(
    r"targeted recommendations\. To prevent pre -split \s*data leakage, the fitting of the RobustScaler was explicitly deferred until after the train -test \s*split\. This ensures that downstream evaluations operate on an uncontaminated training \s*distribution\. \s*",
    "targeted recommendations.\n",
    text
)

# 5. Fix RobustScaler in Interactive Workspaces
text = re.sub(
    r"shape\. To prevent pre-split data leakage, the fitting of the RobustScaler was explicitly \s*deferred until after the train -test split\. This ensures that downstream evaluations \s*operate on an uncontaminated training distribution\. \s*",
    "shape.\n",
    text
)

# 6. Fix DML in Abstract (Keep this? Or remove? Actually, keeping it in Abstract is fine, but user said duplicated in 2.1.2 and 2.1.4. I'll remove from Abstract to be safe if it's really repeated 5 times. Wait, Abstract is a summary, so having it there is actually normal! The user specifically complained about 2.1.2 and 2.1.4. Let's just remove from 2.1.2 and keep in 2.1.4, as well as remove from other weird places like Interactive Workspaces).

# Wait, let's look at DML duplicates:
# Abstract (Line 52): Keep, abstract summarizes methodology.
# Section 2.1.2 (Line 345): Remove, interrupts flow.
text = re.sub(
    r"safety\. The DML implementation was upgraded to utilize K -Fold cross-fitting rather than in-\s*sample residualization\. By training the models on K -1 folds and predicting on the held -out \s*fold, we eliminated regularization bias\. The causal engine mathematically confirmed the \s*Average Treatment Effect \(ATE\) is 0\.0104\. \s*",
    "safety.\n",
    text
)

# Section 5.1 (Line 1228):
text = re.sub(
    r"devoid of data leakage\. The DML implementation was upgraded to \s*utilize K-Fold cross-fitting rather than in -sample residualization\. By training the models on \s*K-1 folds and predicting on the held -out fold, we eliminated regularization bias\. The causal \s*engine mathematically confirmed the Average Treatment Effect \(ATE\) is 0\.0104\. \s*",
    "devoid of data leakage.\n",
    text
)

# Section 6.2 (Line 1587):
text = re.sub(
    r"sliders\. The DML implementation was upgraded to \s*utilize K -Fold cross -fitting rather than in -sample residualization\. By training the \s*models on K-1 folds and predicting on the held -out fold, we eliminated regularization \s*bias\. The causal engine mathematically confirmed the Average Treatment Effect \s*\(ATE\) is 0\.0104\. \s*",
    "sliders.\n",
    text
)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(text)

print("Duplicates removed.")

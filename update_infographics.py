import os
import re
import sys
sys.stdout.reconfigure(encoding='utf-8')

pages_dir = r'C:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\streamlit_app_v2\pages'
app_path = r'C:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\streamlit_app_v2\app.py'

mapping = {
    '1_': 'Predictive_Limits_of_Dating_Data.png',
    '2_': 'AI_Model_Explainability_Overview.png',
    '3_': 'Data_Preprocessing_and_Guardrail_Pipeline.png',
    '4_': 'Feature_Selection_and_Dimensionality_Reduction.png',
    '5_': 'Model_Training_and_Statistical_Evaluation.png',
    '6_': 'Advanced_Neural_Network_Architectures.png',
    '7_': 'Hyperparameter_Search_Space_Optimization.png',
    '8_': 'SHAP_Explainability_Model_Analysis.png',
    '9_': 'Trustworthy_AI_Audit_Framework.png',
    '10_': 'Causal_Inference_and_Uplift_Modeling.png',
    '11_': 'Model_Compression_and_Algorithmic_Recourse.png',
    '12_': 'Matchmaking_Prediction_Inference_Stack.png'
}

for root, _, files in os.walk(pages_dir):
    for file in files:
        if not file.endswith('.py'): continue
        path = os.path.join(root, file)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Determine the correct new infographic
        new_img = None
        for k, v in mapping.items():
            if file.startswith(k):
                new_img = v
                break
        
        if new_img:
            # Check if there is an existing st.image with 'NotebookLM'
            if 'NotebookLM' in content and 'st.image' in content:
                # Use regex to find and replace the whole st.image line
                # It might look like: st.image(os.path.join(ROOT_DIR, "assets", "NotebookLM", "section overview", "Old.png"), use_container_width=True)
                content = re.sub(
                    r'st\.image\(os\.path\.join\(ROOT_DIR,\s*["\']assets["\'],\s*["\']NotebookLM["\'][^\)]+\.png["\']\)(?:,\s*[^)]+)?\)',
                    f'st.image(os.path.join(ROOT_DIR, "assets", "New NotebookLM", "Section overview", "{new_img}"), use_container_width=True)',
                    content
                )
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f'Replaced in {file}')
            elif 'st.title' in content:
                # Add it right after st.title and st.markdown('---')
                # Try to find st.markdown('---') or st.markdown("---") after st.title
                # We'll just insert after st.title
                content = re.sub(
                    r'(st\.title\([^)]+\)\n(?:st\.markdown\([\'"]---[\'"]\)\n)?)', 
                    r'\1st.image(os.path.join(ROOT_DIR, "assets", "New NotebookLM", "Section overview", "' + new_img + r'"), use_container_width=True)\n', 
                    content, count=1
                )
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f'Added to {file}')

# For app.py
with open(app_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the specific paths in app.py
# Science_of_Digital_Romance_Infographic.png
content = re.sub(
    r'os\.path\.join\(ROOT_DIR,\s*["\']assets["\'],\s*["\']NotebookLM["\'][^\)]+\.png["\']\)',
    r'os.path.join(ROOT_DIR, "assets", "New NotebookLM", "Section overview", "Predictive_Limits_of_Dating_Data.png")',
    content
)
with open(app_path, 'w', encoding='utf-8') as f:
    f.write(content)
print('Replaced in app.py')

import os

def fix_anomalies(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    original = content
    content = content.replace("09_causal_dag.png", "09_causal_dag.png")
    content = content.replace("09_causal_dag.png", "09_causal_dag.png")
    content = content.replace("09_causal_dag.png", "09_causal_dag.png")
    
    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
            
if __name__ == "__main__":
    search_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\streamlit_app_v2"
    for root, _, files in os.walk(search_dir):
        for file in files:
            if file.endswith('.py'):
                fix_anomalies(os.path.join(root, file))

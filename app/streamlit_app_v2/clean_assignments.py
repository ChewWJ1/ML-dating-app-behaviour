import os

def clean_redundant_assignments(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    new_lines = []
    for line in lines:
            new_lines.append(line)
            
    if len(new_lines) != len(lines):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

if __name__ == "__main__":
    search_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\streamlit_app_v2"
    for root, _, files in os.walk(search_dir):
        for file in files:
            if file.endswith('.py'):
                clean_redundant_assignments(os.path.join(root, file))

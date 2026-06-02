import json

def dump_notebook(filepath, output_path):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    with open(output_path, 'w', encoding='utf-8') as out:
        for cell in nb.get('cells', []):
            if cell['cell_type'] == 'markdown':
                out.write("--- MARKDOWN ---\n")
                out.write("".join(cell.get('source', [])))
                out.write("\n\n")
            elif cell['cell_type'] == 'code':
                out.write("--- CODE ---\n")
                out.write("".join(cell.get('source', [])))
                out.write("\n")
                for output in cell.get('outputs', []):
                    if output.get('output_type') == 'stream':
                        out.write("--- OUTPUT ---\n")
                        out.write("".join(output.get('text', [])))
                        out.write("\n")
                    elif output.get('output_type') == 'execute_result' or output.get('output_type') == 'display_data':
                        if 'data' in output and 'text/plain' in output['data']:
                            out.write("--- OUTPUT ---\n")
                            out.write("".join(output['data']['text/plain']))
                            out.write("\n")
                out.write("\n")

dump_notebook(r'c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V8_patched_v4.ipynb', 'audit_dump.txt')
print("Dumped.")

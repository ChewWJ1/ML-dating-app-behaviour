import json
import sys

with open(sys.argv[1], 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('model_results.txt', 'w', encoding='utf-8') as out_f:
    for i, cell in enumerate(nb.get('cells', [])):
        if cell['cell_type'] == 'markdown':
            out_f.write(f"--- Markdown Cell {i} ---\n")
            out_f.write("".join(cell.get('source', [])))
            out_f.write("\n\n")
        elif cell['cell_type'] == 'code':
            source = "".join(cell.get('source', []))
            if any(k in source.lower() for k in ['model', 'accuracy', 'score', 'predict', 'fit', 'evaluate', 'report', 'classification']):
                out_f.write(f"--- Code Cell {i} ---\n")
                out_f.write("SOURCE:\n")
                out_f.write(source + "\n")
                out_f.write("OUTPUTS:\n")
                for out in cell.get('outputs', []):
                    if out.get('output_type') == 'stream':
                        out_f.write("".join(out.get('text', [])))
                    elif out.get('output_type') == 'execute_result' or out.get('output_type') == 'display_data':
                        data = out.get('data', {})
                        if 'text/plain' in data:
                            out_f.write("".join(data['text/plain']) + "\n")
                out_f.write("="*40 + "\n")

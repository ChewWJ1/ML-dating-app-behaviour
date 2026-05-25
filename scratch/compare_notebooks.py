import json
import os
import re
import sys

# Ensure UTF-8 printing on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

def analyze_notebook(path):
    if not os.path.exists(path):
        return f"File not found: {path}"
    
    with open(path, 'r', encoding='utf-8') as f:
        try:
            nb = json.load(f)
        except Exception as e:
            return f"Error reading JSON from {path}: {e}"
            
    cells = nb.get('cells', [])
    code_cells = [c for c in cells if c.get('cell_type') == 'code']
    markdown_cells = [c for c in cells if c.get('cell_type') == 'markdown']
    
    # Extract headers
    headers = []
    for c in markdown_cells:
        source = "".join(c.get('source', []))
        for line in source.split('\n'):
            if line.strip().startswith('#'):
                headers.append(line.strip())
                
    # Search for model training, metrics, and print statements in outputs
    metrics_and_outputs = []
    
    for i, c in enumerate(code_cells):
        source = "".join(c.get('source', []))
        outputs = c.get('outputs', [])
        
        # Extract printed outputs that might contain metrics
        for out in outputs:
            text = ""
            if 'text' in out:
                text = "".join(out['text'])
            elif 'data' in out and 'text/plain' in out['data']:
                text = "".join(out['data']['text/plain'])
                
            if text:
                # Look for metrics, classification reports, custom printouts
                metric_patterns = [
                    r'accuracy', r'f1', r'precision', r'recall', r'auc', r'roc', 
                    r'mse', r'rmse', r'mae', r'r2', r'confusion matrix', 
                    r'classification report', r'best model', r'best loss', r'best config'
                ]
                if any(re.search(pat, text.lower()) for pat in metric_patterns):
                    metrics_and_outputs.append({
                        'cell_index': i,
                        'code_snippet': source[:150] + "..." if len(source) > 150 else source,
                        'output': text.strip()
                    })
                    
    return {
        'total_cells': len(cells),
        'code_cells_count': len(code_cells),
        'markdown_cells_count': len(markdown_cells),
        'headers': headers,
        'metrics_outputs': metrics_and_outputs,
        'code_cells': code_cells
    }

def main():
    nb1_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\scratch\ML_dating_app_behaviour jr1.ipynb"
    nb2_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour.ipynb"
    
    print("Analyzing Notebook 1 (scratch/ML_dating_app_behaviour jr1.ipynb)...")
    nb1_data = analyze_notebook(nb1_path)
    print("Analyzing Notebook 2 (notebooks/ML_dating_app_behaviour.ipynb)...")
    nb2_data = analyze_notebook(nb2_path)
    
    if isinstance(nb1_data, str) or isinstance(nb2_data, str):
        print("Error analyzing one of the notebooks:", nb1_data, nb2_data)
        return

    # Check for actual differences in the code cells
    diffs = []
    min_len = min(len(nb1_data['code_cells']), len(nb2_data['code_cells']))
    for i in range(min_len):
        src1 = "".join(nb1_data['code_cells'][i].get('source', []))
        src2 = "".join(nb2_data['code_cells'][i].get('source', []))
        if src1.strip() != src2.strip():
            diffs.append({
                'cell_index': i,
                'src1': src1,
                'src2': src2
            })
            
    # Generate markdown report
    report = []
    report.append("# Jupyter Notebook Comparison Report")
    report.append("\n## File Summary")
    report.append("| Metric | JR1 Notebook (Scratch) | Main Notebook (Notebooks) |")
    report.append("| --- | --- | --- |")
    report.append(f"| Total Cells | {nb1_data['total_cells']} | {nb2_data['total_cells']} |")
    report.append(f"| Code Cells | {nb1_data['code_cells_count']} | {nb2_data['code_cells_count']} |")
    report.append(f"| Markdown Cells | {nb1_data['markdown_cells_count']} | {nb2_data['markdown_cells_count']} |")
    report.append(f"| Code Cell Discrepancies | {len(diffs)} | {len(diffs)} |")
    
    report.append("\n## Section Outline Comparison")
    report.append("| JR1 Headers | Main Headers |")
    report.append("| --- | --- |")
    max_headers = max(len(nb1_data['headers']), len(nb2_data['headers']))
    for i in range(max_headers):
        h1 = nb1_data['headers'][i] if i < len(nb1_data['headers']) else ""
        h2 = nb2_data['headers'][i] if i < len(nb2_data['headers']) else ""
        report.append(f"| {h1} | {h2} |")
        
    report.append("\n## Code Cell Differences")
    if not diffs:
        report.append("No differences found in the code of any cells!")
    else:
        report.append(f"Found {len(diffs)} cells with differing code:")
        for d in diffs:
            report.append(f"\n### Code Cell {d['cell_index']}")
            report.append("**JR1 (Scratch):**")
            report.append("```python\n" + d['src1'] + "\n```")
            report.append("**Main (Notebooks):**")
            report.append("```python\n" + d['src2'] + "\n```")

    report.append("\n## Evaluation Metrics / Outputs Comparison")
    report.append("\n### JR1 Metrics & Key Outputs")
    for m in nb1_data['metrics_outputs']:
        report.append(f"\n#### Cell {m['cell_index']} Code Snippet:")
        report.append("```python\n" + m['code_snippet'].strip() + "\n```")
        report.append("**Output:**")
        report.append("```\n" + m['output'] + "\n```")

    report.append("\n### Main Metrics & Key Outputs")
    for m in nb2_data['metrics_outputs']:
        report.append(f"\n#### Cell {m['cell_index']} Code Snippet:")
        report.append("```python\n" + m['code_snippet'].strip() + "\n```")
        report.append("**Output:**")
        report.append("```\n" + m['output'] + "\n```")

    report_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\scratch\comparison_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
        
    print(f"Report written successfully to: {report_path}")
    print(f"Number of code cells that differ: {len(diffs)}")

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()

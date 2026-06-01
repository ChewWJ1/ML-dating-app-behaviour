"""Extract key results from V7_Strict notebook outputs."""
import json

NB_PATH = r'notebooks/ML_dating_app_behaviour V7_Strict.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def get_text_output(cell):
    """Extract all text outputs from a cell."""
    outputs = cell.get('outputs', [])
    text = []
    for out in outputs:
        if out.get('output_type') == 'stream':
            text.extend(out.get('text', []))
        elif out.get('output_type') in ('execute_result', 'display_data'):
            if 'text/plain' in out.get('data', {}):
                text.extend(out['data']['text/plain'])
    return ''.join(text)

# Key sections to extract
searches = {
    'DML Causal': 'CAUSAL DOUBLE MACHINE LEARNING',
    'Train/Test Split': 'train_test_split',
    'SMOTE Balance': 'SMOTE',
    'Baseline Results': 'BASELINE MODEL COMPARISON',
    'Top 3 Selection': 'Top 3 pipeline-compatible',
    'Tuning Results': 'Tuning:',
    'Final Comparison': 'FINAL MODEL COMPARISON',
    'Best Model': 'Best overall model',
    'Cross-Validation': '5-Fold Cross-Validation',
    'Calibration': 'Isotonic Calibration',
    'Conformal': 'Conformal Prediction',
    'SHAP': 'SHAP Interaction',
    'PyCaret': 'PyCaret',
    'FLAML': 'FLAML',
    'Learning Curves': 'Learning curve',
}

for label, keyword in searches.items():
    for i, cell in enumerate(nb['cells']):
        src = ''.join(cell.get('source', []))
        output = get_text_output(cell)
        if keyword in output or keyword in src:
            if output.strip():
                print(f'\n{"="*80}')
                print(f'  {label} (Cell {i})')
                print(f'{"="*80}')
                # Truncate very long outputs
                if len(output) > 2000:
                    print(output[:2000])
                    print(f'\n... [truncated, {len(output)} chars total]')
                else:
                    print(output)
                break

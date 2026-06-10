import json

notebook_path = 'notebooks/ML_dating_app_behaviour V8_patched_v4.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def find_cell(query):
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            source = "".join(cell.get('source', []))
            if query in source:
                return cell
    return None

cell_tabpfn = find_cell("y_test_sub = y_test.values[:test_limit] if hasattr(y_test, 'values') else np.array(y_test)[:test_limit]")
if cell_tabpfn:
    source = "".join(cell_tabpfn['source'])
    old_eval = """# FIX: Evaluate zero-shot transformer performance STRICTLY on the 1000-sample subset to prevent LightGBM dilution
y_test_sub = y_test.values[:test_limit] if hasattr(y_test, 'values') else np.array(y_test)[:test_limit]
acc_tab = accuracy_score(y_test_sub, y_pred_sub)
roc_auc_tab = roc_auc_score(y_test_sub, y_prob_sub[:, 1] if len(y_prob_sub.shape) > 1 else y_prob_sub)
f1_tab = f1_score(y_test_sub, y_pred_sub)
auc_tab = roc_auc_score(y_test_sub, y_prob_sub)"""

    new_eval = """# FIX: Evaluate zero-shot transformer performance STRICTLY on the 1000-sample subset to prevent LightGBM dilution
# Ensure test_limit is defined even if the model was loaded from joblib cache
test_limit_eval = min(1000, len(y_test))
y_test_sub = y_test.values[:test_limit_eval] if hasattr(y_test, 'values') else np.array(y_test)[:test_limit_eval]
y_pred_sub_eval = y_pred_tab[:test_limit_eval]
y_prob_sub_eval = y_prob_tab[:test_limit_eval]

acc_tab = accuracy_score(y_test_sub, y_pred_sub_eval)
roc_auc_tab = roc_auc_score(y_test_sub, y_prob_sub_eval[:, 1] if len(y_prob_sub_eval.shape) > 1 else y_prob_sub_eval)
f1_tab = f1_score(y_test_sub, y_pred_sub_eval)
auc_tab = roc_auc_score(y_test_sub, y_prob_sub_eval)"""

    source = source.replace(old_eval, new_eval)
    cell_tabpfn['source'] = [line + ('\n' if i < len(source.split('\n')) - 1 else '') for i, line in enumerate(source.split('\n'))]

with open('notebooks/ML_dating_app_behaviour V8_patched_v5.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Bug fix applied successfully!")

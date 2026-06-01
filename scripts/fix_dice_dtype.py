import json
import numpy as np

NB_PATH = r'notebooks/ML_dating_app_behaviour V7_Strict.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

src = ''.join(nb['cells'][173].get('source', []))

old_block = '            # Wrap classifier in DiCE Model\n            m_dice = dice_ml.Model(model=model_obj, backend="sklearn")'

new_block = """            # Wrap classifier in a dtype-safe wrapper for XGBoost compatibility.
            # DiCE internally creates perturbed DataFrames with 'object' dtypes,
            # which XGBoost rejects. This wrapper casts all columns to float before prediction.
            from sklearn.base import BaseEstimator, ClassifierMixin
            class DTypeSafeClassifier(BaseEstimator, ClassifierMixin):
                def __init__(self, model):
                    self.model = model
                    self.classes_ = getattr(model, 'classes_', np.array([0, 1]))
                def fit(self, X, y=None):
                    return self
                def predict(self, X):
                    X = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce').fillna(0)
                    return self.model.predict(X)
                def predict_proba(self, X):
                    X = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce').fillna(0)
                    return self.model.predict_proba(X)

            safe_model = DTypeSafeClassifier(model_obj)
            m_dice = dice_ml.Model(model=safe_model, backend="sklearn")"""

new_src = src.replace(old_block, new_block)

nb['cells'][173]['source'] = [line + '\n' for line in new_src.split('\n')]
if nb['cells'][173]['source']:
    nb['cells'][173]['source'][-1] = nb['cells'][173]['source'][-1].rstrip('\n')

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Fixed DiCE cell with DTypeSafeClassifier wrapper')

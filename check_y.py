import sys
import os
import joblib
sys.path.append(os.path.abspath('streamlit_app_v2'))
from utils import data_loader
from utils.model_loader import inject_classes
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

inject_classes()
X, y, _, _ = data_loader.get_preprocessed_data()
_, _, _, y_test = train_test_split(X.copy(), y.copy(), test_size=0.2, random_state=42, stratify=y)

baseline_results = joblib.load('models_v8/baseline_results.joblib')
for model_name, data in baseline_results.items():
    y_prob = data.get('y_prob')
    if y_prob is not None:
        if len(np.array(y_prob).shape) > 1 and np.array(y_prob).shape[1] > 1:
            probs = np.array(y_prob)[:, 1]
        else:
            probs = np.array(y_prob)
        print(f"{model_name}: probs shape={probs.shape}, type={type(probs)}")
        try:
            fpr, tpr, _ = roc_curve(y_test, probs)
            print(f"Success for {model_name}")
        except Exception as e:
            print(f"Failed for {model_name}: {e}")
            print(probs[:5])

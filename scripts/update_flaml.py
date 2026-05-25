import nbformat
import os

notebook_path = 'notebooks/ML_dating_app_behaviour.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

new_source = '''from flaml import AutoML
import sklearn.metrics
import joblib
import os

try:
    print("--- Starting FLAML ---")
    flaml_checkpoint_path = '../models/flaml_results.joblib'
    
    if os.path.exists(flaml_checkpoint_path):
        print("⚡ Loading pre-trained FLAML model from ../models/flaml_results.joblib instantly!...")
        automl = joblib.load(flaml_checkpoint_path)
    else:
        automl = AutoML()
        
        # FLAML settings (120 seconds budget for demonstration)
        automl_settings = {
            "time_budget": 3600, 
            "metric": 'accuracy',
            "task": 'classification',
            "n_jobs": -1, # Force FLAML to use all 24 CPU threads
            "log_file_name": 'flaml.log',
            "verbose": 0 # Set to 3 for detailed logs
        }
        
        automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
        
        # Ensure models directory exists
        os.makedirs('../models', exist_ok=True)
        joblib.dump(automl, flaml_checkpoint_path)
    
    print("\\nBest FLAML model found:", automl.best_estimator)
    print("Best FLAML hyperparameters:", automl.best_config)
    
    # Evaluate
    flaml_predictions = automl.predict(X_test)
    flaml_accuracy = sklearn.metrics.accuracy_score(y_test, flaml_predictions)
    print(f"\\nFLAML Test Accuracy: {flaml_accuracy:.4f}")
except Exception as e:
    print(f"FLAML Error: {e}")'''

for cell in nb.cells:
    if cell.cell_type == 'code':
        if 'from flaml import AutoML' in cell.source:
            cell.source = new_source
    elif cell.cell_type == 'markdown':
        if 'tuned_results.joblib' in cell.source and 'baseline_results.joblib' in cell.source:
            if 'flaml_results.joblib' not in cell.source:
                # Add FLAML to the markdown list
                lines = cell.source.split('\n')
                new_lines = []
                for line in lines:
                    new_lines.append(line)
                    if '- **`tuned_results.joblib`**: Stores all tuned estimators and grid search parameters.' in line:
                        new_lines.append('- **`flaml_results.joblib`**: Stores the trained FLAML AutoML estimator.')
                cell.source = '\n'.join(new_lines)

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print('Notebook updated successfully.')

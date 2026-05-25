import joblib
m = joblib.load('baseline_results.joblib')
print(list(m['Random Forest']['model'].feature_names_in_))

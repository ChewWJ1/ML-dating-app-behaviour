import re

filepath = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\reports\WIA1006 Machine Learning - Tying the Data Knot Assignment Report V8 (final).md"

with open(filepath, 'r', encoding='utf-8') as f:
    text = f.read()

text = re.sub(
    r"dilution\. The DML implementation was upgraded to utilize K -Fold cross-fitting rather than \s*in-sample residualization\. By training the models on K -1 folds and predicting on the held -out \s*fold, we eliminated regularization bias\. Methodological Disclosure:",
    "dilution. Methodological Disclosure:",
    text
)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(text)

print("Abstract duplicate removed.")

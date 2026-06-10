import re

filepath = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\reports\WIA1006 Machine Learning - Tying the Data Knot Assignment Report V8 (final).md"

with open(filepath, 'r', encoding='utf-8') as f:
    text = f.read()

# Fix DML in Core Technology Stack
# Because there's a page break in the middle, regex with \s* is very helpful here
text = re.sub(
    r"library\. The DML \s*implementation was upgraded to utilize K -Fold cross -fitting rather than in -sample \s*residualization\. By training the models on K -1 folds and predicting on the held -out fold, we \s*eliminated regularization bias\. The causal engine mathematically confirmed the Average \s*WIA1006 Machine Learning \| Tying the \(Data\) Knot Group Assignment Report \s*Page \| 52 \s*Treatment Effect \(ATE\) is 0\.0104\. \s*",
    "library.\n\nBecause of the page break, let's just make sure we keep the page break and not break formatting.\n",
    text
)

# Actually let's use a simpler, more robust replacement that preserves the page break.
# Let's read the file again to avoid messing it up with the above test variable.
with open(filepath, 'r', encoding='utf-8') as f:
    text = f.read()

text = re.sub(
    r"library\. The DML \s*implementation was upgraded to utilize K -Fold cross -fitting rather than in -sample \s*residualization\. By training the models on K -1 folds and predicting on the held -out fold, we \s*eliminated regularization bias\. The causal engine mathematically confirmed the Average \s*",
    "library. ",
    text
)

text = re.sub(
    r"Page \| 52 \s*Treatment Effect \(ATE\) is 0\.0104\. To prevent",
    "Page | 52  \n \nTo prevent",
    text
)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(text)

print("6th duplicate removed.")

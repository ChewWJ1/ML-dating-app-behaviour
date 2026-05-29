import codecs
import re

# Read document text
f = codecs.open(r'c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\docx_text.txt', 'r', 'utf-16le')
text = f.read()
f.close()

keywords = ['baseline', 'comparison', 'compare', 'champion', 'best model']
windows = []
for kw in keywords:
    for m in re.finditer(re.escape(kw), text, re.IGNORECASE):
        # Exclude references page matches (which occur at the end of the text, e.g., index > 80000)
        if m.start() > 80000:
            continue
        start = max(0, m.start() - 250)
        end = min(len(text), m.end() + 250)
        snippet = text[start:end].replace('\n', ' ').replace('\r', ' ').strip()
        windows.append(f'Keyword: "{kw}" at index {m.start()}\nSnippet:\n...{snippet}...\n--------------------------------------------------\n')

with open(r'c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\scratch\keyword_windows.txt', 'w', encoding='utf-8') as f_out:
    f_out.write('\n'.join(windows))
print('Written keyword windows:', len(windows))

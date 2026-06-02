import json

path = r'c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V8.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

t1 = '    "        super().__init__()\\n",\n    "        self.d_token = d_token\\n",\n    "        self.depth = depth\\n",'
r1 = '    "        super().__init__()\\n",\n    "        self.depth = depth\\n",'

t2 = '    "    def __init__(self, num_numeric, cat_vocab_sizes, depth=4, n_trees=5):\\n",\n    "        super().__init__()\\n",\n    "        self.d_token = d_token\\n",'
r2 = '    "    def __init__(self, num_numeric, cat_vocab_sizes, depth=4, n_trees=5):\\n",\n    "        super().__init__()\\n",'

content = content.replace(t1, r1).replace(t2, r2)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print('Done')

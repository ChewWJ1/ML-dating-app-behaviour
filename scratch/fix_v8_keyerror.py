import json

notebook_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\notebooks\ML_dating_app_behaviour V8.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell["cell_type"] == "code" or cell["cell_type"] == "markdown":
        new_source = []
        for line in cell["source"]:
            line = line.replace("df['swipe_right_count']", "df['message_sent_count']")
            line = line.replace("df_temp['swipe_right_count']", "df_temp['message_sent_count']")
            new_source.append(line)
        cell["source"] = new_source

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Fixed KeyError by replacing swipe_right_count with message_sent_count!")

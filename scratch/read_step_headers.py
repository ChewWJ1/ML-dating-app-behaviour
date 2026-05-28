import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

subagent_id = "a548e77e-def7-4da9-989d-062fc428b11e"
base_dir = f"C:\\Users\\HP\\.gemini\\antigravity\\brain\\{subagent_id}\\.system_generated\\steps"

steps = sorted([int(d) for d in os.listdir(base_dir) if d.isdigit()])
for step in steps:
    out_file = os.path.join(base_dir, str(step), "output.txt")
    if os.path.exists(out_file):
        print(f"\n=================== STEP {step} ===================")
        with open(out_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        print("".join(lines[:15]))

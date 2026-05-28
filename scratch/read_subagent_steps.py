import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

steps_dir = r"C:\Users\HP\.gemini\antigravity\brain\a548e77e-def7-4da9-989d-062fc428b11e\.system_generated\steps"
if not os.path.exists(steps_dir):
    print("Steps dir not found!")
    exit(1)

for step in sorted(os.listdir(steps_dir), key=int):
    step_path = os.path.join(steps_dir, step)
    output_file = os.path.join(step_path, "output.txt")
    if os.path.exists(output_file):
        print(f"\n==========================================")
        print(f"STEP {step} OUTPUT:")
        print(f"==========================================")
        with open(output_file, "r", encoding="utf-8") as f:
            content = f.read()
        print(content[:2000]) # print first 2000 chars of each step output

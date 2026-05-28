import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

subagent_dir = r"C:\Users\HP\.gemini\antigravity\brain\a548e77e-def7-4da9-989d-062fc428b11e"
if not os.path.exists(subagent_dir):
    print(f"Error: Subagent directory not found at {subagent_dir}!")
    sys.exit(1)

print(f"Listing files in: {subagent_dir}")
for root, dirs, files in os.walk(subagent_dir):
    print(f"\nRoot: {root}")
    print(f"  Dirs: {dirs}")
    print(f"  Files: {files}")

import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

snap_dir = r"C:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\a548e77e-def7-4da9-989d-062fc428b11e\.tempmediaStorage"
if not os.path.exists(snap_dir):
    snap_dir = r"C:\Users\HP\.gemini\antigravity\brain\a548e77e-def7-4da9-989d-062fc428b11e\.tempmediaStorage"

files = [f for f in os.listdir(snap_dir) if f.startswith("snapshot_full_")]
latest_snap = os.path.join(snap_dir, sorted(files)[0])

with open(latest_snap, "r", encoding="utf-8") as f:
    text = f.read()

lines = text.split("\n")
print(f"Searching {latest_snap} for header tags...")
for idx, line in enumerate(lines):
    if "header" in line.lower() or "stheader" in line.lower():
        print(f"  Line {idx + 1}: {line.strip()[:150]}")

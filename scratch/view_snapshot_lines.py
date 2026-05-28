import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

snap_dir = r"C:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\a548e77e-def7-4da9-989d-062fc428b11e\.tempmediaStorage"
if not os.path.exists(snap_dir):
    # Try the main folder too
    snap_dir = r"C:\Users\HP\.gemini\antigravity\brain\a548e77e-def7-4da9-989d-062fc428b11e\.tempmediaStorage"

files = [f for f in os.listdir(snap_dir) if f.startswith("snapshot_full_")]
latest_snap = os.path.join(snap_dir, sorted(files)[0])

with open(latest_snap, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Printing lines 30 to 60 from {latest_snap}:")
for idx in range(29, min(len(lines), 60)):
    print(f"  {idx + 1}: {lines[idx].rstrip()}")

import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

snap_dir = r"C:\Users\HP\.gemini\antigravity\brain\a548e77e-def7-4da9-989d-062fc428b11e\.tempmediaStorage"
if not os.path.exists(snap_dir):
    print("Snapshot dir not found!")
    exit(1)

# Find first snapshot file
files = [f for f in os.listdir(snap_dir) if f.startswith("snapshot_full_")]
if not files:
    print("No files found!")
    exit(1)

latest_snap = os.path.join(snap_dir, sorted(files)[0])
print(f"Reading snapshot: {latest_snap}")
with open(latest_snap, "r", encoding="utf-8") as f:
    text = f.read()

# Print lines containing 'button' or 'toggle' or 'control' or 'Header' or 'sidebar'
lines = text.split("\n")
count = 0
for idx, line in enumerate(lines):
    if any(x in line.lower() for x in ["button", "toggle", "control", "menu", "sidebar"]):
        count += 1
        print(f"  {count}. Line {idx + 1}: {line.strip()[:150]}")

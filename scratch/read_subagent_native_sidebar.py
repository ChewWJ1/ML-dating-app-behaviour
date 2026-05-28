import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

snap_dir = r"C:\Users\HP\.gemini\antigravity\brain\a548e77e-def7-4da9-989d-062fc428b11e\.tempmediaStorage"
if not os.path.exists(snap_dir):
    print("Snapshot dir not found!")
    exit(1)

# Find the first snapshot file that has content
files = [f for f in os.listdir(snap_dir) if f.startswith("snapshot_full_")]
if not files:
    print("No full snapshot files found!")
    exit(1)

latest_snap = os.path.join(snap_dir, sorted(files)[0])
print(f"Reading snapshot from {latest_snap}...")
with open(latest_snap, "r", encoding="utf-8") as f:
    text = f.read()

# Print lines that contain link and url
lines = text.split("\n")
link_count = 0
for idx, line in enumerate(lines):
    if "link " in line and "url=" in line:
        link_count += 1
        print(f"  {link_count}. {line.strip()}")

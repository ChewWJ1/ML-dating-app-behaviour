import json
import os

logs_path = r"C:\Users\HP\.gemini\antigravity\brain\a73e25c0-06e0-44df-a20b-be205aaf020a\.system_generated\logs\transcript.jsonl"
if not os.path.exists(logs_path):
    # Try alternative paths just in case
    alternative_path = r"C:\Users\HP\.gemini\antigravity\brain\a73e25c0-06e0-44df-a20b-be205aaf020a\transcript.jsonl"
    if os.path.exists(alternative_path):
        logs_path = alternative_path
    else:
        print(f"Error: Transcript not found at {logs_path}")
        # List dir contents of the conversation folder to help locate it
        conv_dir = r"C:\Users\HP\.gemini\antigravity\brain\a73e25c0-06e0-44df-a20b-be205aaf020a"
        if os.path.exists(conv_dir):
            print(f"Dir contents of {conv_dir}:")
            for root, dirs, files in os.walk(conv_dir):
                print(f"Root: {root}")
                print(f"Dirs: {dirs}")
                print(f"Files: {files}")
        exit(1)

print(f"Loading logs from {logs_path}...")
with open(logs_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Total entries in log: {len(lines)}")
# Find model responses or user requests talking about app code or structure
for idx, line in enumerate(lines):
    try:
        obj = json.loads(line)
        content = obj.get("content", "")
        # Look for code blocks or design summaries in model outputs
        if obj.get("source") == "MODEL" and ("streamlit_app_v2" in content or "architecture" in content.lower() or "modular" in content.lower()):
            print(f"\n--- Entry {idx} (MODEL) ---")
            print(content[:1500])
            print("...")
    except Exception as e:
        print(f"Error parsing line {idx}: {e}")

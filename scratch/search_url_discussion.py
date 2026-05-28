import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

logs_path = r"C:\Users\HP\.gemini\antigravity\brain\a73e25c0-06e0-44df-a20b-be205aaf020a\.system_generated\logs\transcript.jsonl"
if not os.path.exists(logs_path):
    print("Logs not found!")
    exit(1)

print("Searching logs for href, URL structure, and routing discussion...")
with open(logs_path, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        try:
            obj = json.loads(line)
            content = obj.get("content", "")
            if obj.get("source") == "MODEL" and ("href" in content or "url" in content.lower() or "button" in content.lower()):
                # Let's search if they discuss hyphen vs underscore
                if "causal" in content.lower() or "page" in content.lower() or "sidebar" in content.lower() or "deploy" in content.lower():
                    # Print lines inside content that mention href
                    lines = content.split("\n")
                    for l in lines:
                        if "href" in l or "causal" in l.lower() or "uplift" in l.lower() or "forecaster" in l.lower():
                            print(f"L {idx} | {l.strip()}")
        except Exception as e:
            pass

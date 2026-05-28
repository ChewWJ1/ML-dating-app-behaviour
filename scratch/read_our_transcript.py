import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

logs_path = r"C:\Users\HP\.gemini\antigravity\brain\b0df43fa-5511-4dee-a9e5-9e34688919ec\.system_generated\logs\transcript.jsonl"
if not os.path.exists(logs_path):
    print("Logs not found!")
    exit(1)

print(f"Reading logs from {logs_path}...")
with open(logs_path, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        try:
            obj = json.loads(line)
            stype = obj.get("type", "")
            source = obj.get("source", "")
            content = obj.get("content", "")
            tool_calls = obj.get("tool_calls", [])
            
            # Print if mentions the subagent ID
            if "a548e77e" in str(obj):
                print(f"\n[Step {idx}] Source: {source} | Type: {stype}")
                if content:
                    print(f"  Content: {content[:400].strip()}")
                if tool_calls:
                    print(f"  Tool calls: {json.dumps(tool_calls)[:400]}")
        except Exception as e:
            print(f"Error on line {idx}: {e}")

import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

logs_path = r"C:\Users\HP\.gemini\antigravity\brain\b0df43fa-5511-4dee-a9e5-9e34688919ec\.system_generated\logs\transcript.jsonl"
if not os.path.exists(logs_path):
    print("Logs not found!")
    exit(1)

with open(logs_path, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        try:
            obj = json.loads(line)
            # Look for subagent messages or any step with "a548e77e"
            content = obj.get("content", "")
            if "a548e77e" in str(obj):
                # Print steps where type is "MESSAGE" or similar, or steps containing actual feedback
                if obj.get("type") in ["USER_INPUT", "MESSAGE", "SUBAGENT_RESPONSE", "SUBAGENT_MESSAGE", "PLANNER_RESPONSE"]:
                    print(f"\n[Step {idx}] Type: {obj.get('type')} | Source: {obj.get('source')}")
                    print(f"  Content: {content}")
        except Exception as e:
            pass

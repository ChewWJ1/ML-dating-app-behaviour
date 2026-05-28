import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

logs_path = r"C:\Users\HP\.gemini\antigravity\brain\a548e77e-def7-4da9-989d-062fc428b11e\.system_generated\logs\transcript.jsonl"
if not os.path.exists(logs_path):
    print("Subagent transcript not found yet!")
    exit(1)

print(f"Reading subagent transcript from {logs_path}...")
with open(logs_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Total steps logged by subagent: {len(lines)}")
# Print the last 15 steps of the subagent's trajectory
start_idx = max(0, len(lines) - 20)
for idx in range(start_idx, len(lines)):
    line = lines[idx]
    try:
        obj = json.loads(line)
        stype = obj.get("type", "")
        source = obj.get("source", "")
        status = obj.get("status", "")
        print(f"\n[Step {idx}] Source: {source} | Type: {stype} | Status: {status}")
        
        # Snippet of content or tool calls
        content = obj.get("content", "")
        if content:
            print(f"  Content snippet: {content[:200].strip()}")
            
        tool_calls = obj.get("tool_calls", [])
        if tool_calls:
            print(f"  Tool calls: {json.dumps(tool_calls)[:300]}")
    except Exception as e:
        print(f"Error parsing line {idx}: {e}")

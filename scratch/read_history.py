import json

log_path = r"C:\Users\HP\AppData\Local\Temp\antigravity\brain\2b050392-e90f-4e0b-b778-578fde3f0d96\.system_generated\logs\transcript.jsonl"
# Wait, let's also try standard appdata path from metadata in user info: C:\Users\HP\.gemini\antigravity\brain
log_path = r"C:\Users\HP\.gemini\antigravity\brain\2b050392-e90f-4e0b-b778-578fde3f0d96\.system_generated\logs\transcript.jsonl"

matches = []

try:
    with open(log_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            data = json.loads(line)
            content = data.get('content', '')
            if "uplift" in content.lower() or "dowhy" in content.lower() or "tabnet" in content.lower() or "smoothing" in content.lower():
                matches.append(f"### Step {line_idx} | Source: {data.get('source')} | Type: {data.get('type')}\n\n```\n{content[:2000]}\n```\n\n---\n")
except Exception as e:
    matches.append(f"Error reading transcript: {e}")

with open('scratch/history_snippets.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(matches))
print("Successfully wrote history snippets to scratch/history_snippets.md")

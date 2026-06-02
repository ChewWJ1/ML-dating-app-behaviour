import json
import os

conversations = [
    "135549f5-1de4-48d6-832e-65a5036f094c",
    "6327fd1e-2b5f-4340-9185-60ce66cb35e4",
    "4fa039c6-fede-4f83-b80c-cdb6b70098c0",
    "ea151657-9139-481c-821f-c0460e957d8d",
    "3b5e1c16-1293-4fdd-af9e-4d39c16bdfb0",
    "e24ed79e-d612-43da-a0b8-3b6b2fed2789",
    "c4c44a0f-ab35-4748-acbf-4efc5791f225",
    "232d76e2-b120-41b9-a941-94de4f45823b",
    "2c8f4fe4-a138-42c3-9687-ee498206f2bd",
    "4d64bdef-ffbe-4c1e-af7f-55423d6144e2",
    "ba773c9d-5738-4f2c-8f93-c609e0350c23",
    "5341984a-366d-4662-9fae-6202f386c64f",
    "92dbe1ba-821e-4abb-a13a-13d1413a3d04",
    "16ef0655-d6b6-4cf1-b21a-823f72287637",
    "b08c4527-8803-47d8-83ab-61908bd03395"
]

base_path = r"C:\Users\HP\.gemini\antigravity\brain"

output = []

for cid in conversations:
    transcript_path = os.path.join(base_path, cid, ".system_generated", "logs", "transcript.jsonl")
    if os.path.exists(transcript_path):
        output.append(f"=== Conversation: {cid} ===")
        with open(transcript_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get('type') == 'USER_INPUT':
                        output.append(f"USER: {data.get('content')}")
                    # Also grab the assistant's walkthroughs or summaries if possible,
                    # but for now, just seeing the user's instructions is key.
                except:
                    pass

with open("scratch/conversation_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output))

print("Dumped user intents to scratch/conversation_summary.txt")

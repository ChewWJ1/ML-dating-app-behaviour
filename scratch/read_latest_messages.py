import os
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

msg_dir = r"C:\Users\HP\AppData\Local\Temp" # Wait, no!
msg_dir = r"C:\Users\HP\.gemini\antigravity\brain\b0df43fa-5511-4dee-a9e5-9e34688919ec\.system_generated\messages"

if not os.path.exists(msg_dir):
    print("Messages dir not found!")
    exit(1)

# List all json files and sort by modified time
files = [os.path.join(msg_dir, f) for f in os.listdir(msg_dir) if f.endswith(".json")]
files.sort(key=lambda x: os.path.getmtime(x))

print(f"Total messages: {len(files)}")
# Print the last 5 messages
for f in files[-5:]:
    print(f"\n--- Message file: {os.path.basename(f)} (mtime: {os.path.getmtime(f)}) ---")
    try:
        with open(f, "r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
            print(json.dumps(data, indent=2))
    except Exception as e:
        print(f"Error reading message: {e}")

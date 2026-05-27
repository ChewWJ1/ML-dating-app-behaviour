import os

search_paths = [".", "reports", "assets", "notebooks"]
image_files = []

for root, dirs, files in os.walk("."):
    # Skip hidden and env directories
    dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('node_modules', 'venv', '.venv')]
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            full_path = os.path.join(root, file)
            image_files.append((full_path, os.path.getsize(full_path)))

print("--- Image Files Found ---")
for img, sz in image_files:
    print(f"{img} ({sz} bytes)")

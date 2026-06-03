import os
import sys
import subprocess

app_dir = os.path.abspath("streamlit_app_v2")
target_file = None

for file in os.listdir(app_dir):
    if file.startswith("1_") and file.endswith(".py"):
        target_file = os.path.join(app_dir, file)
        break

if target_file:
    print(f"Starting streamlit on {target_file}")
    subprocess.run([sys.executable, "-m", "streamlit", "run", target_file, "--server.headless=true"])
else:
    print("Could not find the file starting with 1_!")

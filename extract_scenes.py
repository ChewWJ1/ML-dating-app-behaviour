import os
import subprocess
from pathlib import Path

# Add ffmpeg to PATH
ffmpeg_path = r"C:\Users\HP\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin"
os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]

video_dir = Path(r"C:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\assets\video")
output_dir = video_dir / "b_roll_clips"
output_dir.mkdir(exist_ok=True)

for file in video_dir.glob("*.mp4"):
    print(f"Processing {file.name}...")
    video_output_dir = output_dir / file.stem
    video_output_dir.mkdir(exist_ok=True)
    
    # Detect scenes and split the video into clips
    cmd = [
        "scenedetect", 
        "-i", str(file), 
        "detect-content",
        "split-video", 
        "-o", str(video_output_dir)
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully extracted clips for {file.name}")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting clips for {file.name}: {e}")

print("Extraction complete! Check the b_roll_clips folder.")

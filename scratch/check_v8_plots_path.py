import os

assets_dir = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\assets"
print("Assets contents:")
for name in os.listdir(assets_dir):
    p = os.path.join(assets_dir, name)
    print(f"  {name} isDir={os.path.isdir(p)}")
    if os.path.isdir(p) and "plots" in name.lower():
        print(f"  Files in {name}:", os.listdir(p)[:5])

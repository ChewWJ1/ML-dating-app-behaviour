import zipfile
import os

doc_path = r"c:\Users\HP\Documents\GitHub\ML-dating-app-behaviour\reports\WIA1006 Machine Learning - Tying the Data Knot Assignment Report V5.2 (final).docx"
if os.path.exists(doc_path):
    with zipfile.ZipFile(doc_path, 'r') as z:
        media_files = [f for f in z.namelist() if f.startswith('word/media/')]
        print(f"Total media files: {len(media_files)}")
        # Sort them by name
        media_files.sort()
        for f in media_files[:15]:
            print(f"  {f}: size={z.getinfo(f).file_size} bytes")
        if len(media_files) > 15:
            print("  ...")
            for f in media_files[-5:]:
                print(f"  {f}: size={z.getinfo(f).file_size} bytes")
else:
    print("Report not found!")

import os
import shutil
from pathlib import Path

def flatten_dir(base_dir):
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Skipping {base_dir} (not found)")
        return

    # For each track directory
    for track_dir in base_path.iterdir():
        if not track_dir.is_dir():
            continue
        
        print(f"Flattening track: {track_dir.name}")
        
        # Find all files in all subdirectories of track_dir
        for root, dirs, files in os.walk(track_dir):
            if root == str(track_dir):
                continue
            
            for file in files:
                src_path = Path(root) / file
                dst_path = track_dir / file
                
                # Double check for collision just in case
                if dst_path.exists():
                    print(f"COLLISION: {dst_path}. Skipping.")
                    continue
                
                print(f"Moving {src_path} -> {dst_path}")
                shutil.move(src_path, dst_path)

    # Clean up empty directories
    for track_dir in base_path.iterdir():
        if not track_dir.is_dir():
            continue
        for root, dirs, files in os.walk(track_dir, topdown=False):
            if root == str(track_dir):
                continue
            if not os.listdir(root):
                os.rmdir(root)

flatten_dir("interviews/vault/exemplars")
flatten_dir("interviews/vault/visuals")
print("Flattening complete.")

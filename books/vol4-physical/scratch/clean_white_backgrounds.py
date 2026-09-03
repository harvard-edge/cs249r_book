import numpy as np
from PIL import Image
import os

brain_dir = '/Users/VJ/.gemini/antigravity-cli/brain/8b585c3b-011b-489f-9f2f-84cfe7c1da0e'
assets_dir = '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/assets/covers'

def make_pure_white_bg(img_path, out_path, bg_threshold=235):
    img = Image.open(img_path).convert('RGB')
    arr = np.array(img, dtype=np.float32)
    
    # We want to identify background pixels (very light, low saturation)
    # R, G, B channels
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    brightness = (r + g + b) / 3.0
    
    # Maximum difference between channels (saturation proxy)
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    diff = max_c - min_c
    
    # If brightness is above threshold and color difference is small (near neutral), ramp to pure white 255
    # Smooth curve:
    mask = (brightness > bg_threshold) & (diff < 15)
    
    # Linear ramp from bg_threshold to 255 -> 255
    factor = np.clip((brightness - bg_threshold) / (255.0 - bg_threshold), 0.0, 1.0)
    
    # Where mask is True, blend towards pure white (255, 255, 255)
    for c in range(3):
        arr[:,:,c] = np.where(mask, arr[:,:,c] * (1 - factor) + 255.0 * factor, arr[:,:,c])
        
    out_img = Image.fromarray(np.uint8(np.clip(arr, 0, 255)))
    out_img.save(out_path, quality=98)
    print(f"Processed pure white bg for {os.path.basename(out_path)}")

# Process Vol 2 Scaling
make_pure_white_bg(
    f"{brain_dir}/cover_vol2_scaling_clean_1788368954957.jpg",
    f"{brain_dir}/cover_vol2_scaling_pure_white.png",
    bg_threshold=230
)

# Process Agentic Waypoints
make_pure_white_bg(
    f"{brain_dir}/cover_agentic_waypoints_clean_1788368996111.jpg",
    f"{brain_dir}/cover_agentic_waypoints_pure_white.png",
    bg_threshold=225
)

# Process Physical AI Torus Loop (already in assets)
make_pure_white_bg(
    f"{assets_dir}/cover-physical-loop-recommended-art.png",
    f"{brain_dir}/cover_physical_torus_pure_white.png",
    bg_threshold=240
)

# Process Physical AI Gimbal
make_pure_white_bg(
    f"{assets_dir}/cover-physical-gyroscopic-gimbal-art.png",
    f"{brain_dir}/cover_physical_gimbal_pure_white.png",
    bg_threshold=240
)


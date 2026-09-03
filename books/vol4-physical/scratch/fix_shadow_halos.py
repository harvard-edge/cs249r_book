import numpy as np
from PIL import Image
import os

brain_dir = '/Users/VJ/.gemini/antigravity-cli/brain/8b585c3b-011b-489f-9f2f-84cfe7c1da0e'
assets_dir = '/Users/VJ/GitHub/MLSysBook-integrate-physical/physical/book/assets/covers'
vol_assets = '/Users/VJ/GitHub/MLSysBook-integrate-physical/book/quarto/assets/images/covers'

def neutralize_shadow_and_clean_bg(in_p, out_p):
    img = Image.open(in_p).convert('RGB')
    arr = np.array(img, dtype=np.float32)
    h, w, _ = arr.shape
    
    # Bottom 30% of the image contains the shadow
    shadow_start_y = int(h * 0.70)
    
    # 1. Neutralize color in the lower ground area (make shadow neutral monochrome gray)
    lower_region = arr[shadow_start_y:, :, :]
    # Check if a pixel in the lower region is background/shadow (not sculpture)
    r = lower_region[:, :, 0]
    g = lower_region[:, :, 1]
    b = lower_region[:, :, 2]
    brightness = (r + g + b) / 3.0
    
    # In the bottom 30%, any pixel that is a shadow (brightness > 120 and near neutral or cyan/blue tinted)
    # We desaturate it completely to pure grayscale luminance so there's zero cyan/green glow
    for c in range(3):
        lower_region[:, :, c] = np.where(brightness > 100, brightness, lower_region[:, :, c])
    
    arr[shadow_start_y:, :, :] = lower_region
    
    # 2. Ramp any background pixels (brightness > 220 in the whole image) smoothly to 255.0 pure white
    r_all = arr[:, :, 0]
    g_all = arr[:, :, 1]
    b_all = arr[:, :, 2]
    brightness_all = (r_all + g_all + b_all) / 3.0
    diff_all = np.maximum(np.maximum(r_all, g_all), b_all) - np.minimum(np.minimum(r_all, g_all), b_all)
    
    bg_thresh = 220.0
    mask = (brightness_all > bg_thresh) & (diff_all < 20)
    factor = np.clip((brightness_all - bg_thresh) / (255.0 - bg_thresh), 0.0, 1.0)
    
    # Exponential ease for ultra clean transition to pure white
    factor = factor ** 1.5
    
    for c in range(3):
        arr[:, :, c] = np.where(mask, arr[:, :, c] * (1.0 - factor) + 255.0 * factor, arr[:, :, c])
        
    out = Image.fromarray(np.uint8(np.clip(arr, 0, 255)))
    out.save(out_p, quality=98)
    print(f"Fixed shadow halo and cleaned bg for {os.path.basename(out_p)}")

# 1. Fix Vol 2 Scaling
neutralize_shadow_and_clean_bg(
    f"{brain_dir}/v2_scaling_highres_pw.png",
    f"{brain_dir}/v2_scaling_highres_fixed.png"
)

# 2. Fix Agentic Spawning Trajectory
neutralize_shadow_and_clean_bg(
    f"{brain_dir}/agentic_waypoint_arc_pw.png",
    f"{brain_dir}/agentic_waypoint_arc_fixed.png"
)

# 3. Fix Physical AI Breathy Robotic Finger Loop
neutralize_shadow_and_clean_bg(
    f"{brain_dir}/phys_breathy1_pw.png",
    f"{brain_dir}/phys_breathy1_fixed.png"
)

print("All shadow halos successfully neutralized to pure neutral charcoal drop shadows!")

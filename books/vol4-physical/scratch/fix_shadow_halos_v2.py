import numpy as np
from PIL import Image
import os

brain_dir = '/Users/VJ/.gemini/antigravity-cli/brain/8b585c3b-011b-489f-9f2f-84cfe7c1da0e'

def perfect_shadow_and_bg(raw_img_p, out_p):
    img = Image.open(raw_img_p).convert('RGB')
    arr = np.array(img, dtype=np.float32)
    h, w, _ = arr.shape
    
    # 1. Global background cleanup:
    # Identify true background in upper/middle parts
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    
    # Standard luminance
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    
    # In the bottom area (y > 0.62 * h), everything is floor/shadow
    floor_y = int(0.62 * h)
    
    # For all pixels in the floor region:
    # Make them 100% neutral grayscale (eliminate all cyan/blue bounce light)
    for c in range(3):
        arr[floor_y:, :, c] = lum[floor_y:]
        
    # Re-evaluate luminance for entire image
    lum_clean = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    
    # In the floor region, ramp the shadow smoothly to 255 pure white starting from lum = 160 to 240
    # Values below 160 stay untouched (core shadow).
    # Values between 160 and 240 fade smoothly to 255.
    # Values > 240 become exactly 255.
    floor_lum = lum_clean[floor_y:]
    fade_mask = floor_lum >= 150.0
    t = np.clip((floor_lum - 150.0) / (240.0 - 150.0), 0.0, 1.0)
    # Smooth cosine curve
    t_smooth = 0.5 * (1.0 - np.cos(t * np.pi))
    
    new_floor_lum = np.where(fade_mask, floor_lum * (1.0 - t_smooth) + 255.0 * t_smooth, floor_lum)
    for c in range(3):
        arr[floor_y:, :, c] = new_floor_lum
        
    # Upper region background cleanup (y < floor_y):
    # Where brightness is high and saturation is very low, ramp to 255
    upper_r = arr[:floor_y, :, 0]
    upper_g = arr[:floor_y, :, 1]
    upper_b = arr[:floor_y, :, 2]
    upper_lum = lum_clean[:floor_y]
    upper_diff = np.maximum(np.maximum(upper_r, upper_g), upper_b) - np.minimum(np.minimum(upper_r, upper_g), upper_b)
    
    upper_bg_mask = (upper_lum > 225.0) & (upper_diff < 15.0)
    u_t = np.clip((upper_lum - 225.0) / (255.0 - 225.0), 0.0, 1.0)
    for c in range(3):
        arr[:floor_y, :, c] = np.where(upper_bg_mask, arr[:floor_y, :, c] * (1.0 - u_t) + 255.0 * u_t, arr[:floor_y, :, c])
        
    out = Image.fromarray(np.uint8(np.clip(arr, 0, 255)))
    out.save(out_p, quality=98)
    print(f"Processed perfect shadow for {os.path.basename(out_p)}")

# Re-process from raw source generations
perfect_shadow_and_bg(f"{brain_dir}/cover_vol2_scaling_highres_1788375487703.jpg", f"{brain_dir}/v2_scaling_highres_fixed.png")
perfect_shadow_and_bg(f"{brain_dir}/cover_agentic_waypoint_arc_1788371059472.jpg", f"{brain_dir}/agentic_waypoint_arc_fixed.png")
perfect_shadow_and_bg(f"{brain_dir}/cover_phys_breathy_net1_1788411191186.jpg", f"{brain_dir}/phys_breathy1_fixed.png")

print("All shadows processed with perfect mathematical gradient!")

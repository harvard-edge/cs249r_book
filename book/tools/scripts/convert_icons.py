import os
from PIL import Image

icon_dir = "book/quarto/assets/images/icons/callouts/"
# Process all v* files and the main file
files = [f for f in os.listdir(icon_dir) if (f.startswith("icon_callout_war_story") and f.endswith(".png"))]

for f in files:
    img_path = os.path.join(icon_dir, f)
    pdf_path = os.path.join(icon_dir, f.replace(".png", ".pdf"))
    
    try:
        image = Image.open(img_path)
        
        # Create a white background image for PDF to avoid alpha channel issues in some PDF viewers/printers
        # and to match the likely white page background.
        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
            # Convert to RGBA first to handle transparency correctly
            image = image.convert('RGBA')
            # Create white background
            background = Image.new("RGB", image.size, (255, 255, 255))
            # Composite
            background.paste(image, mask=image.split()[3])
            image_to_save = background
        else:
            image_to_save = image.convert('RGB')
            
        # Save as PDF with high resolution (300 DPI)
        image_to_save.save(pdf_path, "PDF", resolution=300.0)
            
        print(f"Converted {f} to {pdf_path} @ 300 DPI")
    except Exception as e:
        print(f"Failed to convert {f}: {e}")

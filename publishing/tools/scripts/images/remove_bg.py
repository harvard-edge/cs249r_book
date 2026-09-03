#!/usr/bin/env python3
"""
Remove white background from PDF icon image
Makes the background transparent while preserving the red PDF label and gray lines
"""

from PIL import Image
import numpy as np
from pathlib import Path

def remove_white_background(input_path, output_path, threshold=250):
    """
    Remove white background from image and make it transparent

    Args:
        input_path: Path to input image
        output_path: Path to save output image
        threshold: RGB value threshold for white (default 250)
    """
    # Open the image
    img = Image.open(input_path)

    # Convert to RGBA if not already
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    # Convert to numpy array
    data = np.array(img)

    # Create alpha channel based on white pixels
    # White pixels are where R, G, and B are all above threshold
    red, green, blue, alpha = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]

    # Find white/near-white pixels
    white_pixels = (red >= threshold) & (green >= threshold) & (blue >= threshold)

    # Set alpha to 0 (transparent) for white pixels
    data[:,:,3] = np.where(white_pixels, 0, 255)

    # Create new image from modified array
    new_img = Image.fromarray(data, 'RGBA')

    # Save with transparency
    new_img.save(output_path, 'PNG')
    print(f"✅ Saved transparent image to: {output_path}")

    return new_img

def process_callout_icons():
    """Process all callout definition icons to remove white backgrounds"""

    # Base directory for icons
    icon_dir = Path("/Users/VJ/GitHub/MLSysBook/quarto/assets/images/icons/callouts")

    # Look for definition icon
    patterns = [
        "callout-definition.png",
        "definition.png",
        "callout-definition.pdf",
        "definition.pdf"
    ]

    found = False
    for pattern in patterns:
        icon_path = icon_dir / pattern
        if icon_path.exists():
            print(f"📄 Found icon: {icon_path}")

            # Create output path with _transparent suffix
            output_path = icon_path.parent / f"{icon_path.stem}_transparent.png"

            # Remove white background
            remove_white_background(icon_path, output_path)

            # Also create a backup of original
            backup_path = icon_path.parent / f"{icon_path.stem}_original.png"
            if not backup_path.exists():
                img = Image.open(icon_path)
                img.save(backup_path)
                print(f"📦 Backed up original to: {backup_path}")

            # Replace original with transparent version
            remove_white_background(icon_path, icon_path)
            print(f"✅ Replaced original with transparent version: {icon_path}")

            found = True
            break

    if not found:
        print("⚠️  Could not find definition icon. Checking other locations...")

        # Check for PDF-specific icons
        pdf_icon_dir = icon_dir / "pdf"
        if pdf_icon_dir.exists():
            for pattern in patterns:
                icon_path = pdf_icon_dir / pattern
                if icon_path.exists():
                    print(f"📄 Found PDF icon: {icon_path}")

                    # Remove white background
                    output_path = icon_path.parent / f"{icon_path.stem}_transparent.png"
                    remove_white_background(icon_path, output_path)

                    # Replace original
                    remove_white_background(icon_path, icon_path)
                    print(f"✅ Replaced original with transparent version: {icon_path}")
                    found = True
                    break

    return found

def main():
    print("🎨 Removing white background from callout-definition icon...")

    # First, let's find where the icons are located
    base_dir = Path("/Users/VJ/GitHub/MLSysBook/quarto/assets/images/icons")

    if not base_dir.exists():
        print(f"❌ Icon directory not found: {base_dir}")
        return

    # List contents to understand structure
    print(f"\n📁 Checking icon directory structure...")
    for item in base_dir.iterdir():
        if item.is_dir():
            print(f"  📂 {item.name}/")
            # List files in subdirectory
            for subitem in item.iterdir()[:10]:  # Limit to first 10
                print(f"    📄 {subitem.name}")

    # Process the icons
    success = process_callout_icons()

    if success:
        print("\n✅ Successfully removed white background from definition icon!")
        print("📝 Note: The icon will now have a transparent background")
        print("🔄 You may need to rebuild the PDF to see the changes")
    else:
        print("\n⚠️  Could not find the definition icon in expected locations")
        print("Please provide the exact path to the icon file")

if __name__ == "__main__":
    main()

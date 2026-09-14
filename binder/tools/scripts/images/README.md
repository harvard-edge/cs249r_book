# Image Management Scripts

Scripts for managing, processing, and validating images in the book.

## Image Processing
- `compress_images.py` - Compress images to reduce file size
- `remove_bg.py` - Remove backgrounds from images

## Figure Generation
- `build_locator_stack.py` - Regenerate the Volume IV chapter locator stack figures (`fig_locator.svg`); needs `lualatex` and `pdftocairo`

## Image Management
- `manage_images.py` - Main image management utility
- `download_external_images.py` - Download external images
- `manage_external_images.py` - Manage external image references
- `rename_auto_images.py` - Rename automatically generated images
- `rename_downloaded_images.py` - Rename downloaded images

## Validation
- `validate_image_references.py` - Ensure all image references are valid
- `analyze_image_sizes.py` - Analyze image sizes and suggest optimizations

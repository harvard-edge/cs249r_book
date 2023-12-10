import os, re, shutil

def process_markdown_files_with_contents_folder():
    # Get all markdown files in the current directory
    markdown_files = [f for f in os.listdir('.') if f.endswith('.qmd')]

    # Create a main 'contents' directory to hold all the folders
    contents_folder = "contents"
    os.makedirs(contents_folder, exist_ok=True)

    for md_file in markdown_files:
        with open(md_file, 'r') as file:
            content = file.read()

        # Find all images in the markdown file (accounting for optional {#...} format)
        images = re.findall(r'!\[.*?\]\((.*?)\)(\{#.*?\})?', content)
        print(*images, sep = "\n")

        if not images:
            continue

        # Create a folder based on the markdown filename (without extension)
        folder_name = os.path.splitext(md_file)[0]
        full_folder_path = os.path.join(contents_folder, folder_name)
        os.makedirs(full_folder_path, exist_ok=True)

        # Create a subdirectory for images within the folder
        images_folder = os.path.join(full_folder_path, "images")
        os.makedirs(images_folder, exist_ok=True)

        # Move all images to the images folder and update their paths in the markdown content
        for image in images:
            image_path = image[0]  # Extract the image path from the tuple
            if not os.path.exists(image_path):
                continue

            new_image_path = os.path.join(images_folder, os.path.basename(image_path))
            shutil.move(image_path, new_image_path)
            content = content.replace(image_path, os.path.join("images", os.path.basename(image_path)))

        # Write the updated content back to the markdown file
        with open(md_file, 'w') as file:
            file.write(content)

        # Move the markdown file into the created directory within 'contents'
        shutil.move(md_file, full_folder_path)

# Note: This script should be executed in the directory where the markdown files are located.
process_markdown_files_with_contents_folder()

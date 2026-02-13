import os
import subprocess

def get_git_files():
    """Get a list of all files tracked by Git."""
    result = subprocess.run(['git', 'ls-files'], capture_output=True, text=True)
    return result.stdout.splitlines()

def generate_rename_script(output_script_path):
    """
    Generates a shell script to rename files with uppercase characters to lowercase.
    """
    git_files = get_git_files()
    image_extensions = {".png", ".jpg", ".jpeg", ".gif"}
    commands = []

    for file_path in git_files:
        directory, filename = os.path.split(file_path)

        # Check if the filename has any uppercase characters and is an image
        if any(char.isupper() for char in filename) and os.path.splitext(filename)[1].lower() in image_extensions:
            lowercase_filename = filename.lower()
            if filename != lowercase_filename:
                new_path = os.path.join(directory, lowercase_filename)
                # Use a temporary name to handle systems that are case-insensitive
                temp_path = os.path.join(directory, f"temp_{lowercase_filename}")

                commands.append(f'git mv -f "{file_path}" "{temp_path}"')
                commands.append(f'git mv -f "{temp_path}" "{new_path}"')

    if commands:
        with open(output_script_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("# This script renames image files to force Git to recognize case changes.\n\n")
            f.write("\n".join(commands))
            f.write("\n")
        print(f"Generated rename script with {len(commands)//2} files to rename at: {output_script_path}")
    else:
        print("No image files with uppercase characters found to rename.")

if __name__ == "__main__":
    generate_rename_script("fix_casing.sh")

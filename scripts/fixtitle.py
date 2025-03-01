import re
import sys
import os

def update_callouts(text):
    # Pattern to match callouts that already have a title attribute (should be ignored)
    callout_with_title_pattern = re.compile(r':::\{.*?title=".*?".*?\}')

    # Pattern to match callouts that do NOT have a title but contain a header (excluding commented-out headers)
    callout_without_title_pattern = re.compile(
        r'(:::\{(?P<class_id>[^\n]+?)\})\n\n(?!<!--)(?P<header>#{1,6} (?P<title>[^\n]+))\n',
        re.MULTILINE
    )

    def replacer(match):
        class_id = match.group('class_id').strip()
        title = match.group('title')

        # Correctly format the callout block with the title inside the `{}` brackets
        updated_callout = ":::{" + class_id + ' title="' + title + '"}\n'
        return updated_callout

    # Ignore callouts that already have a title
    text_without_titled_callouts = callout_with_title_pattern.sub(lambda m: m.group(0), text)

    # Apply transformations only to callouts missing titles
    updated_text = callout_without_title_pattern.sub(replacer, text_without_titled_callouts)
    
    return updated_text

def process_file(filepath):
    """ Reads a .qmd file, processes it, and writes back the modified content. """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    updated_content = update_callouts(content)

    if content != updated_content:  # Only overwrite if there were changes
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        print(f"Updated: {filepath}")
    else:
        print(f"No changes: {filepath}")

def process_directory(directory):
    """ Recursively find and process all .qmd files in a directory. """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".qmd"):
                filepath = os.path.join(root, file)
                process_file(filepath)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 fixtitle.py <directory>")
        sys.exit(1)

    target_directory = sys.argv[1]

    if not os.path.isdir(target_directory):
        print(f"Error: '{target_directory}' is not a valid directory.")
        sys.exit(1)

    process_directory(target_directory)

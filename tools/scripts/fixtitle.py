import re
import sys
import os

def update_callouts(text):
    callout_with_title_pattern = re.compile(r':::\{.*?title=".*?".*?\}')
    callout_without_title_pattern = re.compile(
        r'(:::\{(?P<class_id>[^\n]+?)\})\n\n(?!<!--)(?P<header>#{1,6} (?P<title>[^\n]+))\n',
        re.MULTILINE
    )

    def replacer(match):
        class_id = match.group('class_id').strip()
        title = match.group('title')
        updated_callout = f":::{{{class_id} title=\"{title}\"}}\n"
        return updated_callout

    text_without_titled_callouts = callout_with_title_pattern.sub(lambda m: m.group(0), text)
    updated_text = callout_without_title_pattern.sub(replacer, text_without_titled_callouts)
    
    return updated_text

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    updated_content = update_callouts(content)

    if content != updated_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        print(f"Updated: {filepath}")
    else:
        print(f"No changes: {filepath}")

def process_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".qmd"):
                filepath = os.path.join(root, file)
                process_file(filepath)

def print_usage():
    print("Usage:")
    print("  python3 fixtitle.py -d <directory>  # Process all .qmd files in the directory recursively")
    print("  python3 fixtitle.py -f <file>       # Process a single .qmd file")
    sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print_usage()

    option, path = sys.argv[1], sys.argv[2]

    if option == "-d":
        if not os.path.isdir(path):
            print(f"Error: '{path}' is not a valid directory.")
            sys.exit(1)
        process_directory(path)

    elif option == "-f":
        if not os.path.isfile(path) or not path.endswith(".qmd"):
            print(f"Error: '{path}' is not a valid .qmd file.")
            sys.exit(1)
        process_file(path)

    else:
        print_usage()

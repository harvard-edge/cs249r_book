import os
import re
import argparse


def update_callout(content):
    """
    Updates callout blocks in the content.

    Looks for blocks of the form:

    ::: {.callout-*}
    ## Any Title
    <text>
    :::

    And changes them to:

    ::: {.callout-* title="Any Title"}
    <text>
    :::
    """
    pattern = re.compile(r"(:::\s*)\{(\.callout-\w+)\}\s*(#+)\s*(.+)", re.MULTILINE)
    replacement = r'\1{\2 title="\4"}'
    return pattern.sub(replacement, content)


def process_file(filepath):
    """Reads a .qmd file, updates the callouts, and overwrites the same file."""
    with open(filepath, "r", encoding="utf-8") as file:
        content = file.read()

    updated_content = update_callout(content)

    with open(filepath, "w", encoding="utf-8") as file:
        file.write(updated_content)

    print(f"Updated file: {filepath}")


def process_directory(directory):
    """Processes all .qmd files in a directory recursively."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".qmd"):
                process_file(os.path.join(root, file))


def main():
    parser = argparse.ArgumentParser(description="Update callout blocks in Quarto .qmd files.")
    parser.add_argument("-f", "--file", help="Specify a single .qmd file to process.")
    parser.add_argument("-d", "--directory", help="Specify a directory to process all .qmd files.")
    args = parser.parse_args()

    if args.file:
        process_file(args.file)
    elif args.directory:
        process_directory(args.directory)
    else:
        print("Please specify a file (-f) or directory (-d) to process.")


if __name__ == "__main__":
    main()

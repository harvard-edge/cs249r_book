#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys

def update_bib_file(bib_file):
    """Runs betterbib update on a single .bib file."""
    print(f"Updating: {bib_file}")
    subprocess.run(["betterbib", "update", "-i", bib_file], check=True)

def process_directory(directory):
    """Finds and updates all .bib files in the given directory."""
    if not os.path.isdir(directory):
        print(f"Error: Directory not found: {directory}", file=sys.stderr)
        sys.exit(1)

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".bib"):
                update_bib_file(os.path.join(root, file))

def main():
    parser = argparse.ArgumentParser(description="Update .bib files using betterbib.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--file", help="Specify a single .bib file to update")
    group.add_argument("-d", "--directory", help="Process all .bib files in a directory")

    args = parser.parse_args()

    if args.file:
        if not os.path.isfile(args.file):
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        update_bib_file(args.file)
    elif args.directory:
        process_directory(args.directory)

if __name__ == "__main__":
    main()

import os
import re
import argparse
import sys

def find_issues(file_path, file_extension):
    """Find potential issues in a file.
    
    Args:
        file_path (str): The path to the file to be analyzed.
        file_extension (str): The extension of the file to determine the type of analysis.
        
    Returns:
        list: A list of issues identified in the file.
    """
    issues = []
    # Open the file and read its contents
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Check for issues based on file extension
    if os.path.splitext(file_path)[1] == file_extension:
        # Check for potential Markdown issues
        for i, line in enumerate(lines, start=1):
            # Check for non-ASCII Unicode characters
            non_ascii_chars = re.findall(r'[^\x00-\x7F]', line)
            if non_ascii_chars:
                for char in non_ascii_chars:
                    char_idx = line.index(char)
                    context = line[max(0, char_idx - 10):min(len(line), char_idx + 10)]
                    issues.append(f"Non-ASCII Unicode character '{char}' found at line {i}: '{context}'")

    return issues

def generate_error_report(directory, file_extension):
    """Generate an error report for files with a specific extension in a directory.
    
    Args:
        directory (str): The directory to search for files.
        file_extension (str): The file extension to check (e.g., .tex, .md).
        
    Returns:
        dict: A dictionary containing file paths mapped to lists of issues found in those files.
    """
    error_report = {}
    # Variable to track whether any issues have been found
    has_issues = False
    # Recursively iterate through the directory
    for root, _, files in os.walk(directory):
        for file in files:
            # Check if the file has the specified extension
            file_path = os.path.join(root, file)
            if os.path.splitext(file_path)[1] == file_extension:
                print(f"Processing file: {file_path}")
                issues = find_issues(file_path, file_extension)
                if issues:
                    has_issues = True
                    error_report[file_path] = issues

    return has_issues, error_report

def main():
    # Command line argument parser
    parser = argparse.ArgumentParser(description="Check files for potential issues.")
    parser.add_argument("directory", help="The directory to search for files.")
    parser.add_argument("file_extension", help="The file extension to check (e.g., .tex, .md)")
    
    # Adding help option for argument parsing
    parser.add_argument(
        "-help", 
        action="help", 
        help="Show this help message and exit."
    )

    args = parser.parse_args()

    # Generate error report
    has_issues, error_report = generate_error_report(args.directory, args.file_extension)

    # Print error report
    if error_report:
        print("\nError Report:")
        for file_path, issues in error_report.items():
            print(f"\nFile: {file_path}")
            for issue in issues:
                print(f"- {issue}")
    
    # Check if any issues were found
    if has_issues:
        sys.exit(1)  # Exit with non-zero exit code if issues were found
    else:
        print(f"\nNo potential issues found in {args.file_extension} files.")
        sys.exit(0)  # Exit with zero exit code if no issues were found

if __name__ == "__main__":
    main()

import os
import re
import argparse
from colorama import Fore, Style, init

init(autoreset=True)

def insert_text(content, start_line, end_line, chapter_title):
    # Convert chapter title to a name format
    chapter_name = chapter_title.lower().replace(' ', '-').strip()

    # Define the text to insert
    insertion_text = f"::: {{.content-visible when-format=\"html\"}}\nResources: [Slides](#sec-{chapter_name}-resource), [Labs](#sec-{chapter_name}-resource), [Exercises](#sec-{chapter_name}-resource)\n:::\n"

    # Insert the text between start and end lines
    content.insert(end_line - 1, insertion_text)
    # Insert a line break
    content.insert(end_line, '\n')

    return content

def print_colored_chunk(chunk):
    for line in chunk:
        if line.startswith('+'):
            print(Fore.GREEN + line.rstrip())
        elif line.startswith('-'):
            print(Fore.RED + line.rstrip())
        else:
            print(Style.RESET_ALL + line.rstrip())

def main(directory):
    # Define the regex patterns
    start_pattern = r'# [^{]*'
    end_pattern = r'!\[_DALL·E 3 Prompt:'

    # Find all .qmd files recursively
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.qmd'):
                file_path = os.path.join(root, file)
                print(f'Processing file: {file_path}')

                # Read the content of the file
                with open(file_path, 'r') as f:
                    content = f.readlines()

                # Convert content to string for easier manipulation
                content_str = ''.join(content)

                # Find the start and end line numbers
                start_match = re.search(start_pattern, content_str)
                end_match = re.search(end_pattern, content_str)

                if start_match and end_match:
                    start_line = content_str.count('\n', 0, start_match.start()) + 1
                    end_line = content_str.count('\n', 0, end_match.start()) + 1

                    # Extract chapter title
                    chapter_title = content[start_line - 1].strip().split('{')[0].strip().lstrip('#').strip()

                    print(f'Found start line: {start_line}')
                    print(f'Found end line: {end_line}')

                    # Extract context before and after the core segment
                    start_context = content[max(0, start_line - 4):start_line - 1]
                    core_segment = content[start_line - 1:end_line]
                    end_context = content[end_line:min(len(content), end_line + 3)]

                    # Prepare the updated content
                    updated_content = insert_text(content, start_line, end_line, chapter_title)

                    # Print the before and after contents with colored diffs
                    print(Fore.YELLOW + 'Before:')
                    for line in start_context:
                        print(Style.RESET_ALL + line.rstrip())
                    print_colored_chunk(core_segment)
                    for line in end_context:
                        print(Style.RESET_ALL + line.rstrip())
                    print(Fore.YELLOW + 'After:')
                    updated_start_context = updated_content[max(0, start_line - 4):start_line - 1]
                    updated_core_segment = updated_content[start_line - 1:end_line + 7]
                    updated_end_context = updated_content[end_line + 7:min(len(updated_content), end_line + 10)]
                    for line in updated_start_context:
                        print(Style.RESET_ALL + line.rstrip())
                    print_colored_chunk(updated_core_segment)
                    for line in updated_end_context:
                        print(Style.RESET_ALL + line.rstrip())

                    # Ask for approval
                    while True:
                        choice = input("Do you approve the change? (yes/no): ").lower().strip()
                        if choice in {'yes', 'y', ''}:  # Default choice is 'yes' if Enter is pressed
                            # Write the updated content back to the file
                            with open(file_path, 'w') as f:
                                f.writelines(updated_content)

                            # Append additional content at the end of the file
                            additional_content = f"\n## Resources {{#sec-{chapter_title.lower().replace(' ', '-')}-resource .unnumbered}}\n\n:::{{.callout-slide collapse=\"false\"}}\n# Slides\n\nComing soon.\n:::\n\n:::{{.callout-exercise collapse=\"false\"}}\n# Exercises\n\nComing soon.\n:::\n\n:::{{.callout-lab collapse=\"false\"}}\n# Labs\n\nComing soon.\n:::\n"
                            print(Fore.YELLOW + 'Additional content to be added at the end of the file:')
                            print(additional_content)
                            additional_choice = input("Do you approve the addition of additional content? (yes/no): ").lower().strip()
                            if additional_choice in {'yes', 'y', ''}:  # Default choice is 'yes' if Enter is pressed
                                with open(file_path, 'a') as f:
                                    f.write(additional_content)
                                print('Additional content added successfully.')
                            elif additional_choice in {'no', 'n'}:
                                print('Additional content not approved. Skipping addition.')
                            else:
                                print('Invalid choice. Skipping addition.')
                            break
                        elif choice in {'no', 'n'}:
                            print('Change not approved. Skipping file.')
                            break
                        else:
                            print('Invalid choice. Please enter yes or no.')
                else:
                    print('Start or end pattern not found.')

                print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Recursively search for .qmd files and update them.')
    parser.add_argument('directory', type=str, help='The directory to start searching for .qmd files')
    args = parser.parse_args()
    main(args.directory)


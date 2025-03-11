import os
import re
import argparse

def check_references_in_qmd(files):
    """
    Scans the given QMD files to find all defined figures (#fig-<name>) and tables (#tbl-<name>),
    and checks if they are properly referenced using @fig-<name> and @tbl-<name>.

    Parameters:
        files (list): List of QMD file paths to check.

    Prints:
        - A summary indicating which files are correct and which have missing references.
    """

    fig_pattern = re.compile(r'#fig-([\w-]+)')  # Matches #fig-<name>
    tbl_pattern = re.compile(r'#tbl-([\w-]+)')  # Matches #tbl-<name>
    fig_ref_pattern = re.compile(r'@fig-([\w-]+)')  # Matches @fig-<name>
    tbl_ref_pattern = re.compile(r'@tbl-([\w-]+)')  # Matches @tbl-<name>

    print("\n=== Cross-Reference Check Report ===")

    for filepath in files:
        fig_defs = set()
        tbl_defs = set()
        fig_refs = set()
        tbl_refs = set()

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract defined figures and tables
            fig_defs.update(fig_pattern.findall(content))
            tbl_defs.update(tbl_pattern.findall(content))

            # Extract referenced figures and tables
            fig_refs.update(fig_ref_pattern.findall(content))
            tbl_refs.update(tbl_ref_pattern.findall(content))

            # Identify missing references
            unused_figs = fig_defs - fig_refs
            unused_tbls = tbl_defs - tbl_refs

            if unused_figs or unused_tbls:
                print(f"\n❌ {filepath}")
                if unused_figs:
                    print("   - Figures that are defined but not referenced:")
                    for fig in sorted(unused_figs):
                        print(f"     • #fig-{fig}")
                if unused_tbls:
                    print("   - Tables that are defined but not referenced:")
                    for tbl in sorted(unused_tbls):
                        print(f"     • #tbl-{tbl}")
            else:
                print(f"✅ {filepath}")

        except Exception as e:
            print(f"\n❌ {filepath}")
            print(f"   - Error reading file: {e}")

def get_qmd_files(target_path, is_directory):
    """
    Retrieves a list of QMD files based on the input path.

    Parameters:
        target_path (str): Path to a file or directory.
        is_directory (bool): If True, scans a directory; otherwise, processes a single file.

    Returns:
        list: List of QMD file paths.
    """
    if is_directory:
        qmd_files = []
        for root, _, files in os.walk(target_path):
            for file in files:
                if file.endswith('.qmd'):
                    qmd_files.append(os.path.join(root, file))
        return qmd_files
    elif target_path.endswith('.qmd'):
        return [target_path]
    else:
        print("Error: Provided file is not a QMD file.")
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check cross-references in QMD files.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--file", help="Path to a specific QMD file.")
    group.add_argument("-d", "--directory", help="Path to a directory containing QMD files.")

    args = parser.parse_args()

    # Determine files to check
    if args.file:
        qmd_files = get_qmd_files(args.file, is_directory=False)
    elif args.directory:
        qmd_files = get_qmd_files(args.directory, is_directory=True)
    
    if qmd_files:
        check_references_in_qmd(qmd_files)
    else:
        print("No valid QMD files found. Exiting.")

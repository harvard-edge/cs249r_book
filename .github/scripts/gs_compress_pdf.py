import argparse
import subprocess
import sys
import os
import shutil

def get_ghostscript_command():
    """Determine the correct Ghostscript command based on the platform."""
    if os.name == 'nt':
        # Try 64-bit and then 32-bit Ghostscript command names
        for cmd in ['gswin64c', 'gswin32c']:
            if shutil.which(cmd):
                return cmd
        print("❌ Ghostscript executable not found. Install it and ensure it's in your PATH (e.g., gswin64c.exe).", file=sys.stderr)
        sys.exit(1)
    else:
        # On Linux/macOS, the command is usually 'gs'
        if shutil.which('gs'):
            return 'gs'
        print("❌ Ghostscript (gs) not found. Install it and ensure it's in your PATH.", file=sys.stderr)
        sys.exit(1)

def convert_pdf(input_file, output_file, settings='/printer', compatibility='1.4', debug=False):
    gs_command = get_ghostscript_command()

    command = [
        gs_command,
        '-sDEVICE=pdfwrite',
        '-dNOPAUSE',
        '-dQUIET' if not debug else '-dQUIET=false',
        '-dBATCH',
        f'-dPDFSETTINGS={settings}',
        f'-dCompatibilityLevel={compatibility}',
        f'-sOutputFile={output_file}',
        input_file
    ]

    if debug:
        print(f"Running command: {' '.join(command)}")

    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if debug:
            print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr.decode()}", file=sys.stderr)
        sys.exit(e.returncode)

def main():
    parser = argparse.ArgumentParser(description="Convert PDF using Ghostscript with various options.")
    parser.add_argument('-i', '--input', required=True, help="Input PDF file")
    parser.add_argument('-o', '--output', required=True, help="Output PDF file")
    parser.add_argument('-s', '--settings', default='/printer', help="PDF settings (default: /printer)")
    parser.add_argument('-c', '--compatibility', default='1.4', help="PDF compatibility level (default: 1.4)")
    parser.add_argument('-d', '--debug', action='store_true', help="Enable debug mode")

    args = parser.parse_args()

    convert_pdf(args.input, args.output, settings=args.settings, compatibility=args.compatibility, debug=args.debug)

if __name__ == "__main__":
    main()

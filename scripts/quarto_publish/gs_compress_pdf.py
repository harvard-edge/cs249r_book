import argparse
import subprocess
import sys

def convert_pdf(input_file, output_file, settings='/screen', compatibility='1.4', debug=False):
    command = [
        'gs',
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
    parser.add_argument('-s', '--settings', default='/screen', help="PDF settings (default: /screen)")
    parser.add_argument('-c', '--compatibility', default='1.4', help="PDF compatibility level (default: 1.4)")
    parser.add_argument('-d', '--debug', action='store_true', help="Enable debug mode")

    args = parser.parse_args()

    convert_pdf(args.input, args.output, settings=args.settings, compatibility=args.compatibility, debug=args.debug)

if __name__ == "__main__":
    main()


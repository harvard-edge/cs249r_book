import os
import sys
import re
import subprocess
import argparse
from zipfile import ZipFile
from PIL import Image
import tempfile
import shutil
import time

DEFAULT_COMPRESSION_QUALITY = 60

def compress_image(image_path, quality=DEFAULT_COMPRESSION_QUALITY):
    try:
        img = Image.open(image_path)
        img.save(image_path, optimize=True, quality=quality)
    except Exception as e:
        print(f"Error compressing image {image_path}: {e}")





def quarto_pdf_render():
    """
    Install Quarto's TinyTeX and render the book to PDF.

    Returns:
        str: Path to the generated PDF file.
    """
    print("Rendering book to PDF...")
    print("Installing Quarto TinyTeX")
    try:
        process = subprocess.run(['quarto', 'render', '--no-clean', '--to', 'pdf'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print("Error:", e)
        sys.exit(1)

    pdf_path = None
    output_lines = process.stdout.splitlines()
    for line in output_lines:
        match = re.search(r'Output created: (.+\.pdf)', line)
        if match:
            pdf_path = match.group(1)
            break

    if not pdf_path:
        output_lines_err = process.stderr.splitlines()
        for line in output_lines_err:
            match = re.search(r'Output created: (.+\.pdf)', line)
            if match:
                pdf_path = match.group(1)
                break

    if not pdf_path:
        print("Error: PDF file path not found.")
        sys.exit(1)

    print(f"Quarto render process return value: {process.returncode}")

    return pdf_path

def quarto_render_html():
    """
    Publish the rendered book using Quarto.
    """
    print("Publishing the rendered book using Quarto")
    try:
        subprocess.run(['quarto', 'render', '--no-clean', '--to', 'html'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while publishing with Quarto: {e}")
        raise RuntimeError("Failed to publish with Quarto")

def quarto_publish():
    """
    Publish the rendered book using Quarto.
    """
    print("Publishing the rendered book using Quarto")
    try:
        subprocess.run(['quarto', 'publish', '--no-render', 'gh-pages'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while publishing with Quarto: {e}")
        raise RuntimeError("Failed to publish with Quarto")

def compress_pdf_ghostscript(input_path, output_path):
    """
    Compress a PDF file using ghostscript.

    Args:
        input_path (str): Path to the input PDF file.
        output_path (str): Path to the output compressed PDF file.
    """
    print(f"Compressing PDF '{input_path}' using ghostscript")

    # Measure input file size
    input_size_before = os.path.getsize(input_path)
    print(f"Input file size: {convert_bytes_to_human_readable(input_size_before)}")

    try:
        # Command for file conversion
        command = ['gs', '-sDEVICE=pdfwrite', '-dCompatibilityLevel=1.4', '-dPDFSETTINGS=/ebook', '-dNOPAUSE', '-dQUIET', '-dBATCH', '-sOutputFile=' + output_path, input_path]
        process = subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print("Error:", e)
        sys.exit(1)

    print(f"Ghostscript render process return value: {process.returncode}")

    # Measure output file size
    output_size_after = os.path.getsize(output_path)
    print(f"Output file size: {convert_bytes_to_human_readable(output_size_after)}")
    print(f"Compression ratio: {(1.0 - (output_size_after / input_size_before)) * 100:.2f}%")

def convert_bytes_to_human_readable(size_in_bytes):
    """
    Convert bytes to human-readable format.

    Args:
        size_in_bytes (int): Size in bytes.

    Returns:
        str: Size in human-readable format.
    """
    if size_in_bytes < 1024:
        return f"{size_in_bytes} bytes"
    elif size_in_bytes < 1024 * 1024:
        return f"{size_in_bytes / 1024:.2f} KB"
    elif size_in_bytes < 1024 * 1024 * 1024:
        return f"{size_in_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_in_bytes / (1024 * 1024 * 1024):.2f} GB"


def main():
    """
    Main function to parse command-line arguments and execute the program.
    """
    parser = argparse.ArgumentParser(description="Convert a book to PDF and optionally reduce its size")
    parser.add_argument('--compress', action='store_true', default=True, help='Compress the PDF file (default: %(default)s)')
    parser.add_argument('--quality', type=int, default=DEFAULT_COMPRESSION_QUALITY, help='Compression quality (default: %(default)s)')
    parser.add_argument('--pdf', action='store_true', default=True, help='Render to PDF (default: %(default)s)')
    parser.add_argument('--publish', action='store_true', default=True, help='Publish to gh-pages (default: %(default)s)')
    parser.add_argument('--html', action='store_true', default=True, help='Build HTML (default: %(default)s)')
    parser.add_argument('--no-pdf', dest='pdf', action='store_false', help="Don't render to PDF")
    parser.add_argument('--no-html', dest='html', action='store_false', help="Don't render to HTML")
    parser.add_argument('--no-publish', dest='publish', action='store_false', help="Don't publish")
    args = parser.parse_args()

    if args.pdf:
        output_pdf_path = quarto_pdf_render()
        output_dir = tempfile.mkdtemp()
        output_pdf_temp_path = os.path.join(output_dir, os.path.basename(output_pdf_path))

        if args.compress:
            print("Compressing PDF using", args.quality)
            compress_pdf_ghostscript(output_pdf_path, output_pdf_temp_path)
            print(f"Compression of {output_pdf_path} completed. Output saved to {output_pdf_temp_path}")

            # Replace the original file with the temporary file
            shutil.move(output_pdf_temp_path, output_pdf_path)
            print(f"Replaced original PDF file with compressed version: {output_pdf_path}")
        else:
            print(f"Output saved to {output_pdf_path}")



    if args.html:
        quarto_render_html()

    quarto_publish()

if __name__ == "__main__":
    main()

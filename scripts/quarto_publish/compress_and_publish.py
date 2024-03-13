import os
import subprocess
import argparse
import PyPDF4
import sys
from PIL import Image
import io
import ghostscript  # Ensure ghostscript is installed and available

# Default input and output paths
DEFAULT_INPUT_PATH = './_book/Machine-Learning-Systems.pdf'
DEFAULT_OUTPUT_PATH = './_book/Machine-Learning-Systems_reduced.pdf'

def quarto_render():
    """
    Install Quarto's TinyTeX and render the book to PDF.
    """
    print("Installing Quarto TinyTeX")
    subprocess.run(['quarto', 'install', 'tinytex'])
    process = subprocess.run(['quarto', 'render', '--to', 'pdf'], check=True)
    print(f"Quarto render process return value: {process.returncode}")

def compress_pdf_pypdf(input_path, output_path):
    """
    Compress a PDF file using PyPDF4 by copying its contents to a new file.

    Args:
        input_path (str): Path to the input PDF file.
        output_path (str): Path to the output compressed PDF file.
    """
    if not os.path.exists(input_path):
        print("Input file does not exist:", input_path)
        return

    with open(input_path, 'rb') as input_file:
        reader = PyPDF4.PdfFileReader(input_file)
        writer = PyPDF4.PdfFileWriter()
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            writer.addPage(page)
        with open(output_path, 'wb') as output_file:
            writer.write(output_file)

def compress_pdf_ghostscript(input_path, output_path):
    """
    Compress a PDF file using ghostscript.

    Args:
        input_path (str): Path to the input PDF file.
        output_path (str): Path to the output compressed PDF file.
    """
    command = [
        'gs',  # or 'ghostscript' depending on your installation
        '-sDEVICE=pdfwrite',
        '-dCompatibilityLevel=1.4',
        '-dPDFSETTINGS=/screen',  # Lower quality, smaller size.
        '-dNOPAUSE',
        '-dQUIET',
        '-dBATCH',
        '-sOutputFile=' + output_path,
        input_path
    ]
    subprocess.run(command, check=True)

def main():
    """
    Main function to parse command-line arguments and execute the program.
    """
    parser = argparse.ArgumentParser(description="Convert a book to PDF and optionally reduce its size")
    parser.add_argument('-c', '--compress', nargs='?', const='ghostscript', default=None, choices=['pypdf', 'ghostscript'], help='Compress the PDF file. Default method: ghostscript')
    parser.add_argument('input_path', nargs='?', default=DEFAULT_INPUT_PATH, help='Path to the rendered book file (default: {})'.format(DEFAULT_INPUT_PATH))
    parser.add_argument('output_path', nargs='?', default=DEFAULT_OUTPUT_PATH, help='Path to the output PDF file (default: {})'.format(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()

    # Run rendering unless compression is specified
    if args.compress is None:
        quarto_render()

    full_input_path = os.path.abspath(args.input_path)
    full_output_path = os.path.abspath(args.output_path)

    # Compress if specified
    if args.compress:
        print("Compressing", full_input_path, "to", full_output_path, "using", args.compress)
        if args.compress == 'ghostscript':
            compress_pdf_ghostscript(full_input_path, full_output_path)
        elif args.compress == 'pypdf':  # This option allows for future expansion
            compress_pdf_pypdf(full_input_path, full_output_path)

if __name__ == "__main__":
    main()

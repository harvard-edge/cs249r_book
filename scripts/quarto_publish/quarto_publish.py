import os
import subprocess

import ghostscript
from absl import app
from absl import flags
from absl import logging

_BOOK_PATH = flags.DEFINE_string("book_path",
                                 '../../_book/Machine-Learning-Systems.pdf',
                                 # Ensure this is a PostScript file
                                 "Path to the rendered book file")
_OUTPUT_PATH = flags.DEFINE_string("output_path",
                                   '../../_book/Machine-Learning-Systems_reduced.pdf',
                                   "Path to the output PDF file")


def quarto_render():
  logging.info("Installing quarto tinytex")
  subprocess.run(['quarto', 'install', 'tinytex'])

  process = subprocess.run(['quarto', 'render', '--to', 'pdf'], check=True)
  logging.info("Quarto render process return value: %s", process.returncode)


def main(_):
  quarto_render()

  full_book_path = os.path.abspath(_BOOK_PATH.value)
  full_output_path = os.path.abspath(_OUTPUT_PATH.value)
  logging.info("Converting %s to %s", full_book_path, full_output_path)

  command = ['ps2pdf', '-dQUIET', '-dBATCH', '-sDEVICE=pdfwrite',
             '-dPDFSETTINGS=/ebook',
             '-dNOPAUSE',
             f'-sOutputFile={full_output_path}',
             full_book_path]
  ghostscript.Ghostscript(*command)


if __name__ == "__main__":
  app.run(main)

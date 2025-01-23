"""
Description:
This script converts the first page of each PDF file in a specified directory into a PNG image. It utilizes the pdf2image library to perform the conversion.

Usage:
- Run the script with `python script_name.py`.
- Specify the path to the directory containing PDF files using the `--path_to_PDF` argument.
- Specify the path to save the converted images using the `--path_to_saved_images` argument.

Example:
python script_name.py --path_to_PDF /path/to/PDF/files --path_to_saved_images /path/to/save/images
"""
# Import the necessary functions and modules.
# convert_from_path: Function to convert PDF pages to images.
# argparse: Module for parsing command-line arguments.
# os: Module for interacting with the operating system.
# Image: Module from PIL (Python Imaging Library) for image processing.
from pdf2image import convert_from_path
import argparse
import os
from PIL import Image

# Create an ArgumentParser object to handle command-line arguments.
parser = argparse.ArgumentParser()

# Add an argument for the path to the PDF files.
parser.add_argument('--path_to_PDF', type=str, required=True, help='Path to the directory containing PDF files.')

# Add an argument for the path to save the converted images.
parser.add_argument('--path_to_saved_images', type=str, required=True, help='Path to the directory to save the converted images.')

parser.add_argument('--PDF_page', type=int, default=1, help='Pge of the important information')
# Parse the command-line arguments and store them in the 'args' variable.
args = parser.parse_args()

# Check if the directory to save the images exists.
folder_exist = os.path.exists(args.path_to_saved_images)

# If the directory does not exist, create it.
if not folder_exist:
    os.makedirs(args.path_to_saved_images)
    print("A new directory to save the images has been created!")

poppler_path = r'C:\Users\victo\Downloads\poppler-24.07.0\Library\bin'
for file in os.listdir(args.path_to_PDF):
    if file.endswith(".pdf"):
        file_name, _ = os.path.splitext(file)

        # Convert PDF to image
        convert_from_path(
            os.path.join(args.path_to_PDF, file),
            output_folder=args.path_to_saved_images,
            fmt='png',
            first_page=args.PDF_page,
            last_page=args.PDF_page,
            output_file=file_name,
            poppler_path= poppler_path
        )
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import os
import subprocess
import sys
import argparse
from pathlib import Path
from pypdf import PdfReader
from utils.constants import (
    DEFAULT_CSV_PATH,
    DEFAULT_PDF_DIR,
    DEFAULT_OUTPUT_FILENAME,
    PAPERS_DOWNLOAD_URL
)

class TextCorpusGenerator:
    def __init__(self, csv_path: str, output_dir: Path = None):
        self.csv_path = Path(csv_path)
        self.output_dir = output_dir or Path.cwd() / "data"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_and_extract_papers(self):
        try:
            print("Downloading paper list...")
            result = subprocess.run(
                ["wget", PAPERS_DOWNLOAD_URL, "--content-disposition"],
                check=True,
                capture_output=True,
                text=True
            )
            
            print("Extracting papers...")
            subprocess.run(["unzip", "PaperList.zip"], check=True)
            print("Successfully downloaded and extracted papers")
            
        except subprocess.CalledProcessError as e:
            print(f"Error during download or extraction: {e}")
            sys.exit(1)

    def process_papers(self):
        try:
            if not self.csv_path.exists():
                raise FileNotFoundError(f"CSV file not found at {self.csv_path}")
                
            df = pd.read_csv(self.csv_path)
            
            download_list = df['Link'].tolist()
            print(f"Found {len(download_list)} paper links")
            if download_list:
                print(f"First download link: {download_list[0]}")

            pdf_directory = Path(DEFAULT_PDF_DIR)
            if not pdf_directory.exists():
                raise FileNotFoundError(f"PDF directory not found at {pdf_directory}")
            
            filenames = os.listdir(pdf_directory)
            pdf_files = [f for f in filenames if f.endswith('.pdf')]
            
            if not pdf_files:
                raise FileNotFoundError("No PDF files found in the directory")
            
            text = ""
            print(f"Processing {len(pdf_files)} PDF files...")
            for filename in pdf_files:
                try:
                    reader = PdfReader(pdf_directory / filename)
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                    print(f"Processed {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue
            
            print(f"Total text length: {len(text)}")
            if text:
                print(f"First 1000 characters: {text[0:1000]}")
            
            output_file = self.output_dir / DEFAULT_OUTPUT_FILENAME
            with open(output_file, "w", encoding='utf-8') as file:
                file.write(text)
            print(f"Text has been saved to {output_file}")

        except Exception as e:
            print(f"Error during paper processing: {e}")
            sys.exit(1)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Process papers from a CSV file containing paper links.'
    )
    parser.add_argument(
        '--csv-path',
        type=str,
        default=DEFAULT_CSV_PATH,
        help='Path to the CSV file containing paper links'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Directory to save output files',
        default=None
    )
    return parser.parse_args()

def main():
    print("Starting paper processing pipeline...")
    args = parse_arguments()
    
    generator = TextCorpusGenerator(
        csv_path=args.csv_path,
        output_dir=Path(args.output_dir) if args.output_dir else None
    )
    
    generator.download_and_extract_papers()
    generator.process_papers()
    print("Pipeline completed successfully")

if __name__ == "__main__":
    main()


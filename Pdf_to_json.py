import os
import json
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime


def preprocess_image(image):
    """Preprocess image for better OCR results."""
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply thresholding
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    
    return denoised


def ocr_pdf(pdf_path, output_dir):
    """
    Perform OCR on a PDF file and save results to JSON.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save the JSON output
    """
    try:
        print(f"Processing: {pdf_path}")
        
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=300)
        
        # Store results for all pages
        results = {
            "source_file": str(pdf_path),
            "total_pages": len(images),
            "processed_date": datetime.now().isoformat(),
            "pages": []
        }
        
        # Process each page
        for page_num, image in enumerate(images, start=1):
            print(f"  Processing page {page_num}/{len(images)}")
            
            # Preprocess image
            processed_img = preprocess_image(image)
            
            # Perform OCR with page segmentation mode for newspaper/column layout
            text = pytesseract.image_to_string(
                processed_img,
                config='--psm 1'  # Automatic page segmentation with OSD
            )
            
            # Get additional data (bounding boxes, confidence)
            data = pytesseract.image_to_data(
                processed_img,
                output_type=pytesseract.Output.DICT,
                config='--psm 1'
            )
            
            # Calculate average confidence (excluding -1 values)
            confidences = [conf for conf in data['conf'] if conf != -1]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            page_result = {
                "page_number": page_num,
                "text": text,
                "word_count": len(text.split()),
                "avg_confidence": round(avg_confidence, 2)
            }
            
            results["pages"].append(page_result)
        
        # Create output filename based on input PDF name
        pdf_name = Path(pdf_path).stem
        output_path = Path(output_dir) / f"{pdf_name}.json"
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Saved to: {output_path}\n")
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {pdf_path}: {str(e)}\n")
        return False


def process_directory(input_dir, output_dir=None):
    """
    Process all PDFs in a directory and its subdirectories.
    
    Args:
        input_dir: Directory containing PDFs
        output_dir: Directory to save JSON files (default: input_dir/ocr_output)
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Directory '{input_dir}' does not exist.")
        return
    
    # Set output directory
    if output_dir is None:
        output_path = input_path / "ocr_output"
    else:
        output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all PDF files recursively
    pdf_files = list(input_path.rglob("*.pdf")) + list(input_path.rglob("*.PDF"))
    
    if not pdf_files:
        print(f"No PDF files found in '{input_dir}'")
        return
    
    print(f"Found {len(pdf_files)} PDF files")
    print(f"Output directory: {output_path}\n")
    print("=" * 60)
    
    # Process each PDF
    success_count = 0
    for pdf_file in pdf_files:
        if ocr_pdf(pdf_file, output_path):
            success_count += 1
    
    print("=" * 60)
    print(f"\nCompleted: {success_count}/{len(pdf_files)} files processed successfully")


if __name__ == "__main__":
    import sys
    
    # Usage examples
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_directory> [output_directory]")

        sys.exit(1)
    
    input_directory = sys.argv[1]
    output_directory = sys.argv[2] if len(sys.argv) > 2 else None
    
    process_directory(input_directory, output_directory)
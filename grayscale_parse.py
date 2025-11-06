import cv2
import numpy as np
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import json
from pathlib import Path
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def segment_text_blocks(image):
    """
    Segment the image into text blocks using contour detection.
    Returns list of bounding boxes (x, y, w, h).
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply binary threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological operations to connect text regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    dilated = cv2.dilate(binary, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and sort contours by area
    text_blocks = []
    min_area = 5000  # Minimum area to consider as a text block
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        if area > min_area:
            text_blocks.append({
                'bbox': (x, y, w, h),
                'area': area
            })
    
    # Sort by position (top to bottom, left to right)
    text_blocks.sort(key=lambda b: (b['bbox'][1] // 100, b['bbox'][0]))
    
    return text_blocks

def visualize_blocks(image, text_blocks, output_path):
    """
    Visualize the detected text blocks on the image.
    """
    fig, ax = plt.subplots(1, figsize=(12, 16))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    for i, block in enumerate(text_blocks):
        x, y, w, h = block['bbox']
        rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                 edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y-10, f"Block {i+1}", color='red', fontsize=12, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {output_path}")

def perform_ocr_on_block(image_crop, model):
    """
    Perform OCR on a single image crop using docTR.
    """
    # Save temporary image
    temp_path = "temp_block.png"
    cv2.imwrite(temp_path, image_crop)
    
    # Load with docTR
    doc = DocumentFile.from_images(temp_path)
    
    # Perform OCR
    result = model(doc)
    
    # Extract text and structure
    block_data = {
        'pages': []
    }
    
    for page in result.pages:
        page_data = {
            'blocks': []
        }
        
        for block in page.blocks:
            block_text = []
            for line in block.lines:
                line_text = ' '.join([word.value for word in line.words])
                block_text.append(line_text)
            
            page_data['blocks'].append({
                'text': '\n'.join(block_text),
                'confidence': float(np.mean([word.confidence for line in block.lines for word in line.words]))
            })
        
        block_data['pages'].append(page_data)
    
    # Combine all text
    full_text = '\n\n'.join([
        block['text'] 
        for page in block_data['pages'] 
        for block in page['blocks']
    ])
    
    return {
        'text': full_text,
        'structured_data': block_data
    }

def process_newspaper_pdf(pdf_path, output_dir="output"):
    """
    Main function to process a newspaper PDF.
    """
    print(f"Processing PDF: {pdf_path}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Convert PDF to image (first page only for test)
    print("Converting PDF to image...")
    images = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=300)
    
    if not images:
        print("Error: Could not convert PDF to image")
        return
    
    # Convert PIL image to OpenCV format
    image = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
    print(f"Image size: {image.shape}")
    
    # Segment into text blocks
    print("Segmenting text blocks...")
    text_blocks = segment_text_blocks(image)
    print(f"Found {len(text_blocks)} text blocks")
    
    # Visualize blocks
    vis_path = output_path / "text_blocks_visualization.png"
    visualize_blocks(image, text_blocks, vis_path)
    
    # Initialize docTR OCR model
    print("Loading docTR OCR model...")
    model = ocr_predictor(pretrained=True)
    
    # Process each text block
    print("\nPerforming OCR on each text block...")
    for i, block in enumerate(text_blocks):
        print(f"Processing block {i+1}/{len(text_blocks)}...")
        
        x, y, w, h = block['bbox']
        
        # Extract image crop
        crop = image[y:y+h, x:x+w]
        
        # Perform OCR
        ocr_result = perform_ocr_on_block(crop, model)
        
        # Save block image
        block_img_path = output_path / f"block_{i+1:02d}.png"
        cv2.imwrite(str(block_img_path), crop)
        
        # Prepare JSON output
        json_output = {
            'block_id': i + 1,
            'bbox': {
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h)
            },
            'area': int(block['area']),
            'text': ocr_result['text'],
            'structured_ocr': ocr_result['structured_data'],
            'block_image': str(block_img_path)
        }
        
        # Save JSON
        json_path = output_path / f"block_{i+1:02d}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)
        
        print(f"  Saved: {json_path}")
        print(f"  Text preview: {ocr_result['text'][:100]}...")
    
    print(f"\n✓ Processing complete! Output saved to: {output_path}")
    print(f"  - {len(text_blocks)} JSON files")
    print(f"  - {len(text_blocks)} block images")
    print(f"  - 1 visualization image")

if __name__ == "__main__":
    # Example usage - replace with your PDF path
    pdf_file = "wyubdi/2004/08/31_01/wyubdi_20040831_0001.pdf" 
    
    # Process the PDF
    process_newspaper_pdf(pdf_file, output_dir="newspaper_output")
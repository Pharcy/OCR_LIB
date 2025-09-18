import os
import json
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# Initialize the model once
model = ocr_predictor(pretrained=True)

# Input and output directories
input_dir = "/mnt/datahive/Data/UWYO_Library/wyubdi/2004/08/31_01/"
output_dir = "output_jsons"
os.makedirs(output_dir, exist_ok=True)

# Loop through files in directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(input_dir, filename)
        print(f"Processing {pdf_path}...")

        # Load PDF
        doc = DocumentFile.from_pdf(pdf_path)

        # Run OCR
        result = model(doc)

        # Export to JSON
        json_output = result.export()
        json_filename = os.path.splitext(filename)[0] + ".json"
        json_path = os.path.join(output_dir, json_filename)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_output, f, ensure_ascii=False, indent=2)

        print(f"Saved OCR result to {json_path}")
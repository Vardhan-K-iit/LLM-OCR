# requirements:
#   pip install torch transformers pillow regex

import os
import re
import json
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# 1. Load BLIP-2 model & processor
processor = BlipProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model     = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

def extract_full_text(image_path: str) -> str:
    """
    Run the multimodal LLM to get a free‐text transcription/caption of the prescription.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    out    = model.generate(**inputs)
    text   = processor.decode(out[0], skip_special_tokens=True)
    return text

def parse_prescription(text: str) -> list[dict]:
    """
    Simple regex‐based parser for medicine entries.
    Returns a list of dicts with keys: medicine, dosage, frequency.
    """
    pattern = re.compile(
        r"(?P<medicine>[A-Za-z]+)\s+"
        r"(?P<dosage>\d+mg)\s+"
        r"(?P<frequency>\d+/\d+\s*(?:day|daily|week|weekly))",
        flags=re.IGNORECASE
    )
    return [m.groupdict() for m in pattern.finditer(text)]

def process_all_images(input_dir: str, output_path: str):
    """
    Walks through input_dir, processes each image, saves JSON with:
      { image: filename, text: full_transcript, structured: [ … ] }
    """
    results = []
    for fn in sorted(os.listdir(input_dir)):
        if fn.lower().endswith((".png", ".jpg", ".jpeg", ".tiff")):
            path = os.path.join(input_dir, fn)
            txt  = extract_full_text(path)
            struct = parse_prescription(txt)
            results.append({
                "image": fn,
                "text": txt,
                "structured": struct
            })
            print(f"Processed {fn}: found {len(struct)} entries")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    # <-- your directory with all prescription images -->
    IMG_DIR  = r"C:\Users\Mavisoft\Downloads\Image\data"
    # <-- wherever you want to save the JSON results -->
    OUT_JSON = r"C:\Users\Mavisoft\Downloads\Image\data\predictions.json"

    process_all_images(IMG_DIR, OUT_JSON)


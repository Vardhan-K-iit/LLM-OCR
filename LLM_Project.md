# LLM-OCR
Technologies to Use
Python (main language)

Open-source Multimodal LLM: e.g., LLaVA, MiniGPT-4, # BLIP-2, or any similar model capable of image + text input.

OCR Library: Tesseract, EasyOCR, or similar (optional for fallback).

Frameworks: PyTorch/Transformers for model interaction.

Jupyter Notebooks or .py scripts for demonstration

**Pipeline Steps**
# Data Loading

Read prescription image files from input directory.

# Pre-processing

Image resizing and denoising.

Optional: Contrast enhancement for handwritten clarity.

# Inference with Multimodal LLM

Input images directly into the selected multimodal LLM. 

# Post-processing

Clean up extracted text fields (remove artifacts, standardize field names).

# Output

Save structured output as CSV/JSON.
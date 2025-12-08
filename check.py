import importlib

modules = [
    "streamlit", "pandas", "numpy", "torch", "chromadb",
    "sentence_transformers", "fitz", "docx", "cv2", "pytesseract",
    "PIL", "tqdm", "nltk", "spacy"
]

print("\nChecking installed modules:\n")

for m in modules:
    try:
        importlib.import_module(m)
        print(f"✅ {m} is installed")
    except ImportError:
        print(f"❌ {m} is NOT installed")

import os
import fitz  # PyMuPDF

from ingestion.pdf.cleaner import clean_pdf_text
from ingestion.common.schema import build_result


def extract_pdf_text(file_path: str) -> str:
    """
    Open a PDF and extract raw text from every page.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    doc = fitz.open(file_path)

    pages_text = []

    for page in doc:
        pages_text.append(page.get_text())

    doc.close()

    return "\n".join(pages_text)


def ingest_pdf(file_path: str, original_filename: str) -> dict:
    """
    Full PDF ingestion pipeline:

    PDF file
        ↓
    Text extraction
        ↓
    Text cleaning
        ↓
    Build standard output format
    """

    # Step 1: Extract text
    raw_text = extract_pdf_text(file_path)

    # Step 2: Clean extracted text
    cleaned_text = clean_pdf_text(raw_text)

    # Step 3: Generate user-friendly title
    title = os.path.splitext(original_filename)[0]

    # Step 4: Return common JSON format
    return build_result(
        source_type="pdf",
        title=title,
        text=cleaned_text,
        source=file_path,
    )


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python extractor.py <path_to_pdf>")

    else:
        pdf_path = sys.argv[1]

        result = ingest_pdf(
            file_path=pdf_path,
            original_filename=os.path.basename(pdf_path)
        )

        print(json.dumps(result, indent=2))
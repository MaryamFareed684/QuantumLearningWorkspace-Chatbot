from ingestion.common.cleaner import clean_text


def clean_pdf_text(raw_text: str) -> str:
    """PDF extraction often leaves broken line-wraps and page-break
    artifacts; the shared cleaner already normalizes spacing/lines,
    so we just call it directly here."""
    return clean_text(raw_text)

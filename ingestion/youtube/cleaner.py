from ingestion.common.cleaner import clean_text


def clean_youtube_text(raw_text: str) -> str:
    """Transcripts have filler words / repeated segments; the shared
    cleaner's dedup + whitespace normalization handles most of it."""
    return clean_text(raw_text)

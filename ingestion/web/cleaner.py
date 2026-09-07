from ingestion.common.cleaner import clean_text


def clean_article_text(raw_text: str) -> str:
    """trafilatura already strips ads/nav/boilerplate; the shared
    cleaner normalizes whitespace and removes duplicate lines."""
    return clean_text(raw_text)

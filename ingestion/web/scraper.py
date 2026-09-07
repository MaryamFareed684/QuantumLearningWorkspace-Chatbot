"""
Member 3: Web Article Ingestion
----------------------------------
Free library used: trafilatura — pip install trafilatura
Chosen over BeautifulSoup/newspaper3k because it strips ads, nav
menus, and boilerplate automatically with no extra rules needed.
100% local parsing, no API key, no cost.
"""

import trafilatura

from ingestion.web.cleaner import clean_article_text
from ingestion.common.schema import build_result


def download_html(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        raise RuntimeError(f"Could not download page: {url}")
    return downloaded


def extract_article(html: str) -> dict:
    """Returns dict with 'text', 'title', 'author', 'date' if found."""
    text = trafilatura.extract(html, include_comments=False, include_tables=False)
    metadata = trafilatura.extract_metadata(html)

    return {
        "text": text or "",
        "title": metadata.title if metadata and metadata.title else "",
        "author": metadata.author if metadata and metadata.author else "",
        "date": metadata.date if metadata and metadata.date else "",
    }


def ingest_article(url: str) -> dict:
    """
    Full pipeline: receive URL -> download HTML -> extract main
    content (strip ads/nav/etc) -> clean -> return common-format dict.
    """
    html = download_html(url)
    extracted = extract_article(html)

    cleaned_text = clean_article_text(extracted["text"])

    return build_result(
        source_type="article",
        title=extracted["title"] or url,
        text=cleaned_text,
        author=extracted["author"],
        date=extracted["date"],
        source=url,
    )


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python scraper.py <article_url>")
    else:
        result = ingest_article(sys.argv[1])
        print(json.dumps(result, indent=2))

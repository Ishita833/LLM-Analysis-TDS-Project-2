from langchain_core.tools import tool
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from urllib.parse import urljoin


def extract_all_urls(html: str, base_url: str) -> list:
    soup = BeautifulSoup(html, "html.parser")
    urls = set()

    # Extract href attributes
    for tag in soup.find_all(href=True):
        urls.add(urljoin(base_url, tag["href"]))

    # Extract src attributes (images, scripts, videos, etc.)
    for tag in soup.find_all(src=True):
        urls.add(urljoin(base_url, tag["src"]))

    return list(urls)


def clean_text(html: str, base_url: str = "") -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Remove script/style contents
    for tag in soup(["script", "style"]):
        tag.decompose()
    # Convert <a href> → "text (URL)"
    for a in soup.find_all("a", href=True):
            href = urljoin(base_url, a["href"])
            a.replace_with(f"{a.get_text(strip=True)} ({href})")

    # Convert <img src> → "[Image: URL]"
    for img in soup.find_all("img", src=True):
        src = urljoin(base_url, img["src"])
        img.replace_with(f"[Image: {src}]")

    # Extract visible text
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def get_rendered_html(url: str) -> dict:
    print("\nFetching and rendering:", url)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            page.goto(url, wait_until="networkidle")
            raw_html = page.content()
            browser.close()

            # 1️⃣ Extract ALL file URLs before cleaning
            all_urls = extract_all_urls(raw_html, url)

            # 2️⃣ Clean readable text for the LLM
            cleaned_text = clean_text(raw_html, url)

            # Optionally truncate long pages
            if len(cleaned_text) > 300000:
                cleaned_text = cleaned_text[:300000] + "... [TRUNCATED]"

            return {
                "url": url,
                "text": cleaned_text,
                "files": all_urls,
            }

    except Exception as e:
        return {"error": f"Error fetching/rendering: {str(e)}"}

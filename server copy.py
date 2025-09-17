from fastmcp import FastMCP
from playwright.sync_api import sync_playwright

mcp = FastMCP("chrome-agent")
_browser = None
_page = None

def _get_page():
    global _browser, _page
    if _page is None:
        playwright = sync_playwright().start()
        _browser = playwright.chromium.launch(headless=False)
        _page = _browser.new_page()
    return _page

@mcp.tool
def open_url(url: str) -> str:
    page = _get_page()
    page.goto(url)
    return f"Opened {url} with title: {page.title()}"

@mcp.tool
def extract_content() -> dict:
    page = _get_page()
    return {
        "title": page.title(),
        "h1": [h.inner_text() for h in page.query_selector_all("h1")],
        "h2": [h.inner_text() for h in page.query_selector_all("h2")],
        "paragraphs": [p.inner_text() for p in page.query_selector_all("p")],
    }

@mcp.tool
def find_text(keyword: str, context_chars: int = 80) -> list:
    page = _get_page()
    full_text = page.inner_text("body")
    matches = []
    idx = 0
    while (idx := full_text.lower().find(keyword.lower(), idx)) != -1:
        start = max(0, idx - context_chars)
        end = min(len(full_text), idx + len(keyword) + context_chars)
        snippet = full_text[start:end].replace("\n", " ")
        matches.append(snippet)
        idx += len(keyword)
    return matches if matches else [f"No matches found for '{keyword}'"]

@mcp.tool
def click_link(link_text: str) -> str:
    page = _get_page()
    links = page.query_selector_all("a")
    for link in links:
        text = (link.inner_text() or "").strip()
        if text.lower() == link_text.lower():
            href = link.get_attribute("href")
            link.click()
            return f"Clicked link '{link_text}' → {href}"
    return f"No link found with text '{link_text}'"

if __name__ == "__main__":
    mcp.run(transport="stdio")

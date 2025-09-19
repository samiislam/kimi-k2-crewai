from fastmcp import FastMCP
from playwright.async_api import async_playwright

mcp = FastMCP("chrome-agent")
_browser = None
_page = None
_playwright = None

async def _get_page():
    global _browser, _page, _playwright
    if _page is None:
        _playwright = await async_playwright().start()
        _browser = await _playwright.chromium.launch(headless=True)
        _page = await _browser.new_page()
    return _page

@mcp.tool
async def open_url(url: str) -> str:
    page = await _get_page()
    await page.goto(url)
    title = await page.title()
    return f"Opened {url} with title: {title}"

@mcp.tool
async def extract_content() -> dict:
    page = await _get_page()
    title = await page.title()
    h1_elements = await page.query_selector_all("h1")
    h2_elements = await page.query_selector_all("h2")
    p_elements = await page.query_selector_all("p")
    h1 = [await h.inner_text() for h in h1_elements]
    h2 = [await h.inner_text() for h in h2_elements]
    paragraphs = [await p.inner_text() for p in p_elements]
    return {
        "title": title,
        "h1": h1,
        "h2": h2,
        "paragraphs": paragraphs,
    }

@mcp.tool
async def find_text(keyword: str, context_chars: int = 80) -> list:
    page = await _get_page()
    full_text = await page.inner_text("body")
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
async def click_link(link_text: str) -> str:
    page = await _get_page()
    links = await page.query_selector_all("a")
    for link in links:
        text = (await link.inner_text() or "").strip()
        if text.lower() == link_text.lower():
            href = await link.get_attribute("href")
            await link.click()
            return f"Clicked link '{link_text}' → {href}"
    return f"No link found with text '{link_text}'"

if __name__ == "__main__":
    mcp.run(transport="stdio")

"""
Scrape The Batch issues (1..320) and save a minimal JSON:
- issue_url
- title
- text
- images: list[str] of image src URLs only

Usage (when you want to run it):
  python scrape_min.py --start 1 --end 320 --out the_batch_articles_min.json --delay 0.5
"""

from __future__ import annotations

import json
import time
from argparse import ArgumentParser
from typing import Iterable, List, Dict, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, Tag, Comment

BASE_URL = "https://www.deeplearning.ai"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def iter_issue_urls(start: int = 1, end: int = 320) -> Iterable[str]:
    for i in range(start, end + 1):
        yield f"{BASE_URL}/the-batch/issue-{i}/"


def _has_post_content_class(classes) -> bool:
    """Detect the main content div by its class names.

    Look for a class that starts with 'post_postContent__' along with 'prose--styled'.
    """
    if not classes:
        return False
    if isinstance(classes, str):
        classes = classes.split()
    return any(c.startswith("post_postContent__") for c in classes) and (
        "prose--styled" in classes
    )


def fetch_soup(url: str, timeout: int = 30) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def resolve_img_src(img: Tag) -> str:
    """Resolve an <img> URL using multiple attributes and srcset candidates."""
    src = (
        img.get("src")
        or img.get("data-src")
        or img.get("data-lazy-src")
        or img.get("data-original")
        or ""
    )
    if not src:
        srcset = img.get("srcset") or img.get("data-srcset") or ""
        if srcset:
            candidates = [c.strip().split()[0] for c in srcset.split(",") if c.strip()]
            if candidates:
                src = candidates[-1]  # typically largest
    if src:
        src = urljoin(BASE_URL, src)
    return src


def _absolutize_resources(soup: BeautifulSoup, base: str = BASE_URL) -> None:
    """Make resource URLs absolute within a soup fragment.

    Absolutizes href/src attributes and rewrites srcset URLs to absolute.
    """
    # href-bearing tags
    for tag in soup.find_all(["a", "link"]):
        href = tag.get("href")
        if href:
            tag["href"] = urljoin(base, href)

    # src-bearing tags
    for tag in soup.find_all(["img", "script"]):
        src = tag.get("src")
        if src:
            tag["src"] = urljoin(base, src)

    # Handle srcset on <img> and <source>
    for tag in soup.find_all(["img", "source"]):
        srcset = tag.get("srcset")
        if not srcset:
            continue
        parts = []
        for c in srcset.split(","):
            c = c.strip()
            if not c:
                continue
            bits = c.split()
            url = urljoin(base, bits[0])
            if len(bits) > 1:
                parts.append(f"{url} {bits[1]}")
            else:
                parts.append(url)
        tag["srcset"] = ", ".join(parts)


def _inline_hidden_media(soup: BeautifulSoup) -> None:
    """Inline hidden media so that <img> tags become discoverable.

    - Replace <noscript> blocks with their parsed HTML contents.
    - Expand HTML comments that contain Ghost/Koenig card HTML or obvious media.
    """
    # Inline <noscript> contents
    for nos in list(soup.find_all("noscript")):
        try:
            frag = BeautifulSoup(nos.decode_contents(), "html.parser")
            # Insert contents before the noscript to preserve order
            for child in list(frag.contents):
                nos.insert_before(child)
            nos.decompose()
        except Exception:
            # If anything goes wrong, just remove noscript to avoid duplication
            nos.decompose()

    # Expand media inside HTML comments
    for cm in list(soup.find_all(string=lambda s: isinstance(s, Comment))):
        content = str(cm)
        if (
            ("kg-card-begin" in content)
            or ("<img" in content)
            or ("<figure" in content)
        ):
            try:
                inner = content.replace("<!--kg-card-begin: html-->", "").replace(
                    "<!--kg-card-end: html-->", ""
                )
                frag = BeautifulSoup(inner, "html.parser")
                # Insert parsed nodes after the comment, then remove the comment
                # Insert in reverse order to keep original sequence
                for child in reversed(list(frag.contents)):
                    cm.insert_after(child)
                cm.extract()
            except Exception:
                # If parsing fails, drop the comment to avoid noise
                cm.extract()


def _collect_images_from_soup(
    soup_fragment: BeautifulSoup, images: list, added_srcs: set
):
    # Prefer figures with captions
    for fig in soup_fragment.find_all("figure"):
        img = fig.find("img")
        if not img:
            continue
        src = resolve_img_src(img)
        if not src or src in added_srcs:
            continue
        images.append({"src": src})
        added_srcs.add(src)

    # Standalone images not inside figure
    for img in soup_fragment.find_all("img"):
        if img.find_parent("figure") is not None:
            continue
        src = resolve_img_src(img)
        if not src or src in added_srcs:
            continue
        images.append({"src": src})
        added_srcs.add(src)


def extract_article_content(blocks: List[Tag]):
    """From a list of Tag blocks, extract concatenated text and image metadata.

    Captures figures and images, plus images embedded via <noscript> and in Ghost/Koenig
    HTML cards placed inside HTML comments.
    """
    paragraphs: List[str] = []
    images: List[Dict] = []
    added_srcs: set[str] = set()

    for b in blocks:
        # Gather text from common text-bearing elements
        for t in b.find_all(
            ["p", "li", "blockquote", "pre", "figcaption", "h2", "h3", "h4"]
        ):
            txt = t.get_text(" ", strip=True)
            if txt:
                paragraphs.append(txt)
        if getattr(b, "name", None) in {"p", "li", "blockquote", "pre", "figcaption"}:
            txt = b.get_text(" ", strip=True)
            if txt:
                paragraphs.append(txt)

        # 1) Parse figures and images present in normal DOM
        _collect_images_from_soup(b, images, added_srcs)

        # 2) Images inside <noscript>
        for nos in b.find_all("noscript"):
            try:
                frag = BeautifulSoup(nos.decode_contents(), "html.parser")
                _collect_images_from_soup(frag, images, added_srcs)
            except Exception:
                pass

        # 3) Images embedded inside HTML comments (Ghost kg-card)
        for cm in b.find_all(string=lambda s: isinstance(s, Comment)):
            content = str(cm)
            if (
                ("kg-card-begin" in content)
                or ("<img" in content)
                or ("<figure" in content)
            ):
                try:
                    inner = content.replace("<!--kg-card-begin: html-->", "").replace(
                        "<!--kg-card-end: html-->", ""
                    )
                    frag = BeautifulSoup(inner, "html.parser")
                    _collect_images_from_soup(frag, images, added_srcs)
                except Exception:
                    pass

    return "\n\n".join(paragraphs).strip(), images


def split_articles_by_heading(content_div: Tag):
    """Split the content of an issue page into separate articles at H1/H2/H3.

    Fixes media association: any media blocks (figures/images/wrappers) immediately
    preceding a heading are moved to the following section so that images above
    the title belong to that article, not the previous one.
    """
    sections = []
    current = {"title": "", "blocks": []}

    def push_current():
        nonlocal current
        if current["title"] or current["blocks"]:
            sections.append(current)
            current = {"title": "", "blocks": []}

    def is_media_block(node) -> bool:
        # Treat figures/images/pictures or wrappers containing them as media
        if isinstance(node, Tag):
            if node.name in {"figure", "img", "picture"}:
                return True
            # common wrappers like div.kg-card that contain media
            if node.find(["figure", "img", "picture"]):
                return True
        # If it's a fragment (BeautifulSoup doc), search within
        find = getattr(node, "find", None)
        if callable(find) and find(["figure", "img", "picture"]):
            return True
        return False

    for child in content_div.children:
        if isinstance(child, Comment):
            try:
                inner = (
                    str(child)
                    .replace("<!--kg-card-begin: html-->", "")
                    .replace("<!--kg-card-end: html-->", "")
                )
                frag = BeautifulSoup(inner, "html.parser")
                current["blocks"].append(frag)
            except Exception:
                pass
            continue
        if not isinstance(child, Tag):
            continue
        if child.name in {"h1", "h2", "h3"}:
            # Move trailing media (immediately before the heading) into the new section
            moved_media = []
            while current["blocks"] and is_media_block(current["blocks"][-1]):
                moved_media.append(current["blocks"].pop())
            moved_media.reverse()

            # Push previous section only if it has content/title after moving media
            if current["title"] or current["blocks"]:
                sections.append(current)

            # Start a new section with the heading and any carried media
            current = {"title": child.get_text(strip=True), "blocks": moved_media}
        else:
            current["blocks"].append(child)

    # Finalize last section
    push_current()

    # If no headings found, fallback: one article with best-effort title
    if not sections:
        page_h1 = content_div.find("h1")
        title = page_h1.get_text(strip=True) if page_h1 else ""
        if not title:
            first_p = content_div.find(["p", "h3"])  # lead text
            title = (
                (first_p.get_text(strip=True)[:80] + "...") if first_p else "(untitled)"
            )
        sections = [{"title": title, "blocks": [content_div]}]

    sections = [s for s in sections if s["title"].strip()]
    return sections


def parse_issue(issue_url: str) -> List[Dict]:
    """Parse a single issue page, splitting into articles by H1/H2/H3.

    Returns list of dicts with keys: issue_url, title, text, images (list of src only).

    Logic mirrors the notebook's robust approach:
    - Build a per-article container from section blocks.
    - Inline hidden media from <noscript> and HTML comments.
    - Absolutize URLs and srcset entries.
    - Collect image URLs from the reconstructed container.
    """
    soup = fetch_soup(issue_url)

    content_div = soup.find("div", class_=_has_post_content_class)
    if not content_div:
        content_div = soup.select_one("div.prose--styled")
    if not content_div:
        return []

    sections = split_articles_by_heading(content_div)

    # Exclude blocks by title containing 'deeplearning.ai' (robust to spacing/punctuation/case)
    def _normalize_title(t: str) -> str:
        return "".join(ch for ch in t.lower() if ch.isalnum())

    brand_key = _normalize_title("deeplearning.ai")
    news_key = _normalize_title("news")
    filtered_sections = []
    for s in sections:
        norm = _normalize_title(s["title"]) if s["title"] else ""
        if not norm:
            continue
        if brand_key in norm:
            continue
        if norm == news_key:
            continue
        filtered_sections.append(s)

    out: List[Dict] = []
    for sec in filtered_sections:
        # Prepare a fresh container and copy blocks into it, with hidden media inlined
        article_doc = BeautifulSoup('<div class="article"></div>', "html.parser")
        container = article_doc.select_one("div.article")

        for b in sec["blocks"]:
            # Work on a cloned fragment to avoid mutating original soup tree
            frag = BeautifulSoup(str(b), "html.parser")
            _inline_hidden_media(frag)
            _absolutize_resources(frag, BASE_URL)
            for child in list(frag.contents):
                container.append(child)

        # Text via existing extractor for consistency
        text, _ = extract_article_content(sec["blocks"])

        # Collect image URLs from the reconstructed container
        seen = set()
        img_srcs: List[str] = []
        for img in container.find_all("img"):
            src = resolve_img_src(img)
            if src and src not in seen:
                seen.add(src)
                img_srcs.append(src)

        out.append(
            {
                "issue_url": issue_url,
                "title": sec["title"],
                "text": text,
                "images": img_srcs,
            }
        )
    return out


def scrape_issues_min(
    start: int = 1,
    end: int = 320,
    delay: float = 0.1,
    limit_issues: Optional[int] = None,
) -> List[Dict]:
    """Scrape issues in [start, end] and return a flat list of minimal article dicts.

    If limit_issues is provided (> 0), stop after scraping at most that many issues.
    """
    all_articles: List[Dict] = []
    processed = 0
    for url in iter_issue_urls(start, end):
        try:
            print(f"Fetching issue: {url}")
            issue_articles = parse_issue(url)
            print(f"  → Found {len(issue_articles)} articles in this issue")
            all_articles.extend(issue_articles)
        except requests.HTTPError as http_err:
            print(f"HTTP error for {url}: {http_err}")
        except Exception as e:
            print(f"Error for {url}: {e}")
        time.sleep(delay)
        processed += 1
        if limit_issues and limit_issues > 0 and processed >= limit_issues:
            break
    return all_articles


def save_articles_json_min(
    articles: List[Dict], path: str = "the_batch_articles_min.json"
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"articles": articles}, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(articles)} articles to {path}")


def build_arg_parser() -> ArgumentParser:
    p = ArgumentParser(description="Scrape The Batch issues into minimal JSON")
    p.add_argument(
        "--start", type=int, default=1, help="Start issue number (inclusive)"
    )
    p.add_argument("--end", type=int, default=320, help="End issue number (inclusive)")
    p.add_argument(
        "--delay", type=float, default=0.5, help="Delay between requests in seconds"
    )
    p.add_argument(
        "--limit-issues",
        type=int,
        default=None,
        help=(
            "Maximum number of issues to scrape within the [start, end] range. "
            "If omitted, all issues in the range are scraped."
        ),
    )
    p.add_argument(
        "--out",
        type=str,
        default="the_batch_articles_min.json",
        help="Output JSON file path",
    )
    return p


def main():
    args = build_arg_parser().parse_args()
    articles = scrape_issues_min(
        start=args.start,
        end=args.end,
        delay=args.delay,
        limit_issues=args.limit_issues,
    )
    save_articles_json_min(articles, args.out)


if __name__ == "__main__":
    # Script entrypoint. Will only run when this file is executed directly.
    main()

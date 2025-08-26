# scrape_gmaps_reviews_sync.py
from playwright.sync_api import sync_playwright
import argparse, csv, json, re, time, hashlib
from urllib.parse import quote_plus, urlsplit, urlunsplit

HEADLESS = False
SCROLL_PAUSE = 1.2
MAX_REVIEWS_DEFAULT = 20

def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def rating_category(val: str) -> str:
    try:
        v = float(val)
        return "positive" if v >= 4 else "neutral" if v >= 3 else "negative"
    except:
        return ""

def ensure_en(url: str) -> str:
    """Append ?hl=en (or &hl=en) so UI strings are English."""
    parts = list(urlsplit(url))
    q = parts[3]
    parts[3] = (q + "&hl=en") if q else "hl=en"
    return urlunsplit(parts)

def open_maps_and_get_place_url(page, query: str) -> str:
    """Search in Maps, click the first result that shows ratings in its aria-label."""
    maps_search = f"https://www.google.com/maps/search/{quote_plus(query)}?hl=en"
    page.goto(maps_search, wait_until="domcontentloaded")

    # Cookie/consent (best-effort)
    for label in ["Accept all", "I agree", "Accept"]:
        try:
            page.get_by_role("button", name=re.compile(label, re.I)).click(timeout=1500)
            break
        except:
            pass

    # Wait for results feed to render
    page.wait_for_selector('div[role="feed"]', timeout=8000)

    # Prefer entries that clearly have ratings in aria-label (e.g. "... 4.3 stars ...")
    # If none, fall back to the first visible place link.
    preferred = page.locator(
        'div[role="feed"] a.hfpxzc[aria-label*="stars"], '
        'div[role="feed"] a.hfpxzc[aria-label*="reviews"]'
    )
    candidate = preferred.first if preferred.count() else page.locator('div[role="feed"] a.hfpxzc[aria-label]').first

    # Try up to a few candidates in case the first is a “Menu”/collection/etc.
    for idx in range(min(candidate.count() or 1, 5)):
        try:
            (candidate if candidate.count() else page.locator('div[role="feed"] a.hfpxzc[aria-label]').nth(idx)).click(timeout=5000)
            page.wait_for_timeout(2500)
            url = page.url.split("?")[0]
            # Heuristic: place pages include /maps/place/
            if "/maps/place/" in url:
                return url
            # else go back and try next
            page.go_back(wait_until="domcontentloaded")
            page.wait_for_timeout(1200)
        except:
            # try next
            try:
                page.go_back(wait_until="domcontentloaded")
                page.wait_for_timeout(1200)
            except:
                pass

    # Last resort: any visible /maps/place/ link in the DOM
    try:
        page.locator('a[href*="/maps/place/"]').first.click(timeout=5000)
        page.wait_for_timeout(2500)
        return page.url.split("?")[0]
    except:
        return ""

def click_reviews(page):
    """Open the Reviews tab reliably."""
    # 1) Safest: the tab has aria-label like: 'Reviews for <Place Name>'
    try:
        reviews_tab = page.locator('button[role="tab"][aria-label^="Reviews for"]')
        if reviews_tab.count():
            reviews_tab.first.scroll_into_view_if_needed()
            reviews_tab.first.click(timeout=2500)
            return True
    except:
        pass

    # 2) Next best: click the 'Reviews' tab inside the tablist (avoids hitting 'Menu')
    try:
        tablist = page.locator('div[role="tablist"]').first
        tab = tablist.get_by_role("tab", name=re.compile(r"^Reviews$", re.I))
        tab.scroll_into_view_if_needed()
        tab.click(timeout=2500)
        return True
    except:
        pass

    # 3) Fallbacks: Google’s 'More reviews' control or the rating chip
    try:
        page.locator('button[jsaction*="pane.reviewChart.moreReviews"]').first.click(timeout=2000)
        return True
    except:
        pass
    try:
        page.locator('div[aria-label*="out of 5"]').first.click(timeout=2000)
        return True
    except:
        pass

    return False

def get_business_name(page):
    # h1 usually holds business name
    try:
        return page.locator("h1").first.inner_text(timeout=2000)
    except:
        return ""

def scrape_reviews_from_place(page, place_url: str, max_reviews: int):
    page.goto(ensure_en(place_url), wait_until="domcontentloaded")

    # Dismiss consent
    for label in ["Accept all", "I agree", "Accept"]:
        try: page.get_by_role("button", name=re.compile(label, re.I)).click(timeout=1500); break
        except: pass

    clicked = click_reviews(page)
    # Wait for the reviews list to render (several fallbacks)
    page.wait_for_selector(
        'div.m6QErb.DxyBCb.kA9KIf.dS8AEf.XiKgde, div[aria-label^="Reviews"], div[aria-label*="Google reviews"]',
        timeout=8000
    )
    business_name = get_business_name(page)

    # Reviews container (Google rotates classes, use broad matches)
    # Preferred scroll container:
    scroll_container = None
    candidates = [
        'div.m6QErb.DxyBCb.kA9KIf.dS8AEf.XiKgde',                  # common reviews + list wrapper
        'div.m6QErb.DxyBCb.kA9KIf.dS8AEf',                         # sometimes without XiKgde
        'div[aria-label^="Reviews"], div[aria-label*="Google reviews"]'
    ]
    for sel in candidates:
        loc = page.locator(sel)
        if loc.count():
            scroll_container = loc.last  # the LAST one is usually the scrolling list
            break

    # As a safety, if still None, try to find the parent of a review tile
    if not scroll_container:
        tile = page.locator('div.jJc9Ad, div[data-review-id]').first
        if tile.count():
            scroll_container = tile.locator('xpath=ancestor::div[contains(@class,"m6QErb")]').first

    # If we never opened reviews (no button), we might still get inline tiles—continue anyway.
    rows, seen = [], set()

    # Helper to expand visible "More" buttons inside reviews
    def expand_more():
        try:
            more = page.locator("button", has_text=re.compile(r"\bmore\b", re.I))
            for i in range(min(more.count(), 20)):
                try: more.nth(i).click(timeout=400)
                except: pass
        except: pass

    # Helper to parse all currently visible review elements via locators
    def harvest_current():
        newly = 0
        # Each review tile typically has class jJc9Ad; also look for data-review-id
        review_tiles = page.locator('div.jJc9Ad, div[data-review-id]')
        count = review_tiles.count()
        for i in range(count):
            tile = review_tiles.nth(i)
            try:
                # Author
                author = ""
                try: author = tile.locator('div.d4r55').first.inner_text(timeout=800)
                except: pass

                # Rating (primary: kvMYJc has aria-label)
                rating_number = ""
                rating_el = tile.locator('span.kvMYJc').first
                if rating_el.count():
                    aria = rating_el.get_attribute('aria-label') or ""
                    m = re.search(r'([0-9.]+)', aria)
                    if m: rating_number = m.group(1)
                else:
                    # alt pattern text like "4/5"
                    alt = tile.locator('span.fzvQIb').first
                    if alt.count():
                        m = re.search(r'(\d+(?:\.\d+)?)/5', alt.inner_text())
                        if m: rating_number = m.group(1)

                # Text
                review_text = ""
                text_loc = tile.locator('span.wiI7pd').first
                if text_loc.count():
                    try: review_text = text_loc.inner_text(timeout=800)
                    except: pass
                if not review_text:
                    # Fallback: the largest visible span
                    spans = tile.locator('span')
                    for j in range(min(spans.count(), 8)):
                        t = spans.nth(j).inner_text() or ""
                        if len(t.strip()) > len(review_text):
                            review_text = t.strip()

                # Photo (if present)
                photo = ""
                img = tile.locator("img").first
                if img.count():
                    src = img.get_attribute("src") or img.get_attribute("data-src") or ""
                    if src and "http" in src:
                        photo = src

                if not review_text:
                    continue

                key = sha(f"{author}|{review_text}")
                if key in seen:
                    continue
                seen.add(key)

                rows.append({
                    "business_name": business_name,
                    "author_name": author,
                    "text": review_text,
                    "photo": photo,
                    "rating": rating_number,
                    "rating_category": rating_category(rating_number),
                    "place_url": place_url
                })
                newly += 1
                if len(rows) >= max_reviews:
                    break
            except:
                pass
        return newly

    # Main loop: expand, harvest, scroll, repeat
    stagnant_loops = 0
    while len(rows) < max_reviews and stagnant_loops < 4:
        expand_more()
        before = len(rows)
        harvest_current()
        after = len(rows)

        if after == before:
            stagnant_loops += 1
        else:
            stagnant_loops = 0

        # Scroll the drawer if we found it; else scroll page
        try:
            if scroll_container:
                handle = scroll_container.element_handle()
                page.evaluate("(el) => el.scrollTop = el.scrollHeight", handle)
            else:
                page.mouse.wheel(0, 1800)
        except:
            pass

        time.sleep(SCROLL_PAUSE)

    return rows

def save_outputs(rows, csv_path, json_path):
    if not rows:
        print("No reviews found.")
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"Saved to {csv_path} and {json_path}")

if __name__ == "__main__":
    import sys
    # Quick-and-dirty CLI:
    # python scrape_gmaps_reviews_sync.py "<place-url or search term>" 120
    target = sys.argv[1] if len(sys.argv) > 1 else "Restaurants"
    max_reviews = int(sys.argv[2]) if len(sys.argv) > 2 else MAX_REVIEWS_DEFAULT

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        context = browser.new_context(locale="en-US", extra_http_headers={"Accept-Language": "en-US,en;q=0.9"})
        page = context.new_page()

        if target.startswith("http"):
            place_url = target.split("?")[0]
        else:
            place_url = open_maps_and_get_place_url(page, target)

        print("Using place URL:", place_url)
        rows = scrape_reviews_from_place(page, place_url, max_reviews)
        print(f"Collected {len(rows)} reviews")
        save_outputs(rows, "reviews.csv", "reviews.json")

        browser.close()
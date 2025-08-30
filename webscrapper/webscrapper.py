# scrape_gmaps_reviews_sync.py
from playwright.sync_api import sync_playwright
from playwright._impl._errors import TimeoutError
from pathlib import Path
import argparse, csv, json, re, time, hashlib, os
from urllib.parse import quote_plus, urlsplit, urlunsplit

HEADLESS = False
SCROLL_PAUSE = 1.5
MAX_REVIEWS_DEFAULT = 20

# use python tester.py --per-place-max-reviews 50 --max-places 1000 to reuse same checkpoint

# ---------- NEW: grid centers around Singapore ----------
# (lat, lng, zoom)
GRID_CENTERS = [
    (1.3521, 103.8198, 12),   # Central SG (broad sweep)
    (1.29027, 103.85196, 14), # CBD / Marina Bay
    (1.3000, 103.8550, 14),   # Orchard / River Valley
    (1.3333, 103.8480, 14),   # Toa Payoh / Balestier
    (1.3691, 103.8454, 14),   # Ang Mo Kio
    (1.3721, 103.9493, 14),   # Punggol
    (1.3437, 103.8738, 14),   # Serangoon
    (1.3611, 103.8860, 14),   # Hougang
    (1.4043, 103.9020, 14),   # Yishun
    (1.4380, 103.7886, 14),   # Woodlands
    (1.4491, 103.8239, 14),   # Sembawang
    (1.4253, 103.7420, 14),   # Kranji
    (1.3854, 103.7448, 14),   # Choa Chu Kang
    (1.3401, 103.7074, 14),   # Jurong West
    (1.3151, 103.7649, 14),   # Clementi
    (1.3496, 103.8736, 14),   # Bishan / AMK fringe
    (1.3526, 103.9447, 14),   # Pasir Ris
    (1.3456, 103.9568, 14),   # Loyang
    (1.3667, 103.9100, 14),   # Paya Lebar
    (1.3450, 103.9320, 14),   # Tampines
    (1.3430, 103.9530, 14),   # Simei
    (1.3236, 103.9273, 14),   # Bedok
    (1.3890, 103.9870, 14),   # Changi
]

# ---------- GLOBAL checkpoint (across all viewports) ----------
CHECKPOINT_PATH = "gmaps_checkpoint_global.json"
CHECKPOINT_SAVE_EVERY = 1  # save after each place by default

def _safe_write_json(path: str, data: dict):
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)

def _empty_global_cp(base_query: str) -> dict:
    return {
        "base_query": base_query,          # stable query across centers (e.g., "Restaurants in Singapore")
        "seen_place_urls": [],             # GLOBAL dedupe across all centers
        "seen_place_names": [],            # GLOBAL dedupe across all centers
        "viewports": {}                    # per-viewport progress: key -> {"idx": int}
    }

def load_global_checkpoint(base_query: str) -> dict:
    p = Path(CHECKPOINT_PATH)
    if not p.exists():
        return _empty_global_cp(base_query)
    try:
        cp = json.loads(p.read_text(encoding="utf-8"))
        if cp.get("base_query") != base_query:
            # New/different base query -> start fresh
            return _empty_global_cp(base_query)
        cp.setdefault("seen_place_urls", [])
        cp.setdefault("seen_place_names", [])
        cp.setdefault("viewports", {})
        return cp
    except Exception:
        return _empty_global_cp(base_query)

def save_global_checkpoint(cp: dict):
    _safe_write_json(CHECKPOINT_PATH, cp)

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

    preferred = page.locator(
        'div[role="feed"] a.hfpxzc[aria-label*="stars"], '
        'div[role="feed"] a.hfpxzc[aria-label*="reviews"]'
    )
    candidate = preferred.first if preferred.count() else page.locator('div[role="feed"] a.hfpxzc[aria-label]').first

    # Try a few candidates in case the first is a “Menu”/collection/etc.
    for idx in range(min(candidate.count() or 1, 5)):
        try:
            (candidate if candidate.count() else page.locator('div[role="feed"] a.hfpxzc[aria-label]').nth(idx)).click(timeout=5000)
            page.wait_for_timeout(2500)
            url = page.url.split("?")[0]
            if "/maps/place/" in url:
                return url
            page.go_back(wait_until="domcontentloaded")
            page.wait_for_timeout(1200)
        except:
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
    # 1) aria-label starts with 'Reviews for ...'
    try:
        reviews_tab = page.locator('button[role="tab"][aria-label^="Reviews for"]')
        if reviews_tab.count():
            reviews_tab.first.scroll_into_view_if_needed()
            reviews_tab.first.click(timeout=2500)
            return True
    except: pass

    # 2) Exact 'Reviews' tab in tablist
    try:
        tablist = page.locator('div[role="tablist"]').first
        tab = tablist.get_by_role("tab", name=re.compile(r"^Reviews$", re.I))
        tab.scroll_into_view_if_needed()
        tab.click(timeout=2500)
        return True
    except: pass

    # 3) Fallbacks
    try:
        page.locator('button[jsaction*="pane.reviewChart.moreReviews"]').first.click(timeout=2000)
        return True
    except: pass
    try:
        page.locator('div[aria-label*="out of 5"]').first.click(timeout=2000)
        return True
    except: pass

    return False

def get_business_name(page, fallback: str = "") -> str:
    """Robustly read the business name from the place header."""
    try:
        h1 = page.locator("h1.DUwDvf.lfPIob").first
        if h1.count():
            txt = h1.inner_text(timeout=2000).strip()
            if txt and txt.lower() != "results":
                return txt
            # If spans are empty and name is a bare text node, read only text nodes:
            txt2 = h1.evaluate(
                """el => Array.from(el.childNodes)
                        .filter(n => n.nodeType === Node.TEXT_NODE)
                        .map(n => n.textContent.trim())
                        .filter(Boolean)
                        .join(' ')"""
            ).strip()
            if txt2 and txt2.lower() != "results":
                return txt2
    except:
        pass
    # Fallback to page title heuristic
    try:
        t = (page.title() or "").replace(" - Google Maps", "").strip()
        if t and t.lower() != "results":
            return t
    except:
        pass
    return fallback or ""

def scrape_reviews_from_place(page, place_url: str, max_reviews: int, fallback_name: str = ""):
    page.goto(ensure_en(place_url), wait_until="domcontentloaded")
    page.wait_for_timeout(400)

    # Dismiss consent
    for label in ["Accept all", "I agree", "Accept"]:
        try:
            page.get_by_role("button", name=re.compile(label, re.I)).click(timeout=1500)
            break
        except: pass

    click_reviews(page)
    page.wait_for_timeout(400)

    # Wait for reviews area, but don't crash if slow
    try:
        page.wait_for_selector(
            'div.m6QErb.DxyBCb.kA9KIf.dS8AEf.XiKgde, div[aria-label^="Reviews"], div[aria-label*="Google reviews"]',
            timeout=8000
        )
    except TimeoutError:
        # Nudge once more
        click_reviews(page)
        page.wait_for_timeout(800)

    # Early exit if clearly no reviews
    if page.locator('text=/No reviews yet/i').count():
        return []

    business_name = get_business_name(page, fallback=fallback_name)

    # Pick the scroll container
    scroll_container = None
    candidates = [
        'div.m6QErb.DxyBCb.kA9KIf.dS8AEf.XiKgde',
        'div.m6QErb.DxyBCb.kA9KIf.dS8AEf',
        'div[aria-label^="Reviews"], div[aria-label*="Google reviews"]'
    ]
    for sel in candidates:
        loc = page.locator(sel)
        if loc.count():
            scroll_container = loc.last
            break
    if not scroll_container:
        tile = page.locator('div.jJc9Ad, div[data-review-id]').first
        if tile.count():
            scroll_container = tile.locator('xpath=ancestor::div[contains(@class,"m6QErb")]').first

    rows, seen = [], set()

    def expand_more():
        try:
            more = page.locator("button", has_text=re.compile(r"\bmore\b", re.I))
            for i in range(min(more.count(), 20)):
                try: more.nth(i).click(timeout=400)
                except: pass
        except: pass

    def harvest_current():
        newly = 0
        review_tiles = page.locator('div.jJc9Ad, div[data-review-id]')
        count = review_tiles.count()
        for i in range(count):
            tile = review_tiles.nth(i)
            try:
                author = ""
                try: author = tile.locator('div.d4r55').first.inner_text(timeout=800)
                except: pass

                rating_number = ""
                rating_el = tile.locator('span.kvMYJc').first
                if rating_el.count():
                    aria = rating_el.get_attribute('aria-label') or ""
                    m = re.search(r'([0-9.]+)', aria)
                    if m: rating_number = m.group(1)
                else:
                    alt = tile.locator('span.fzvQIb').first
                    if alt.count():
                        m = re.search(r'(\d+(?:\.\d+)?)/5', alt.inner_text())
                        if m: rating_number = m.group(1)

                review_text = ""
                text_loc = tile.locator('span.wiI7pd').first
                if text_loc.count():
                    try: review_text = text_loc.inner_text(timeout=800)
                    except: pass
                if not review_text:
                    spans = tile.locator('span')
                    for j in range(min(spans.count(), 8)):
                        t = spans.nth(j).inner_text() or ""
                        if len(t.strip()) > len(review_text):
                            review_text = t.strip()

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

    def scroll_once():
        try:
            if scroll_container:
                handle = scroll_container.element_handle()
                page.evaluate("(el) => { el.scrollBy(0, Math.max(400, el.clientHeight*0.9)); }", handle)
            else:
                page.mouse.wheel(0, 1800)
        except:
            pass

    stagnant_loops = 0
    while len(rows) < max_reviews and stagnant_loops < 6:
        expand_more()
        before = len(rows)
        try:
            page.wait_for_selector('div.jJc9Ad, div[data-review-id]', timeout=5000)
        except TimeoutError:
            # Nudge list: click reviews again and scroll a bit
            click_reviews(page)
            for _ in range(2):
                scroll_once()
                time.sleep(0.5)
        harvest_current()
        after = len(rows)

        stagnant_loops = stagnant_loops + 1 if after == before else 0
        for _ in range(3):
            scroll_once()
            time.sleep(SCROLL_PAUSE)

    return rows

# ---------- APPEND OUTPUT HELPERS ----------
def _csv_has_header(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0

def append_outputs(rows, csv_path: str, json_path: str):
    if not rows:
        return
    csv_p = Path(csv_path)
    need_header = not _csv_has_header(csv_p)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        if need_header:
            w.writeheader()
        w.writerows(rows)
    # JSON append (read-modify-write)
    data = []
    if Path(json_path).exists():
        try:
            data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        except:
            data = []
    data.extend(rows)
    Path(json_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------- UPDATED: SCRAPE MANY FROM FEED with checkpoint ----------
def scrape_query_feed(page,
                      query: str,
                      max_places: int,
                      per_place_max_reviews: int,
                      csv_path: str,
                      json_path: str,
                      center=None,
                      global_cp: dict = None):
    if center:
        lat, lng, zoom = center
        maps_search = f"https://www.google.com/maps/search/{quote_plus(query)}/@{lat},{lng},{zoom}z?hl=en"
    else:
        maps_search = f"https://www.google.com/maps/search/{quote_plus(query)}?hl=en"
    page.goto(maps_search, wait_until="domcontentloaded")

    # Cookie/consent (best-effort)
    for label in ["Accept all", "I agree", "Accept"]:
        try:
            page.get_by_role("button", name=re.compile(label, re.I)).click(timeout=1500)
            break
        except: pass

    page.wait_for_selector('div[role="feed"]', timeout=10000)
    feed = page.locator('div[role="feed"]').first

    # ----- resume GLOBAL checkpoint with per-viewport progress -----
    processed_count = 0
    if global_cp is None:
        global_cp = load_global_checkpoint(query)

    # a stable key per viewport (center) so we can resume that feed independently
    if center:
        lat, lng, zoom = center
        viewport_key = f"{lat:.5f},{lng:.5f},z{zoom}"
    else:
        viewport_key = "default"

    vp_state = global_cp["viewports"].get(viewport_key, {"idx": 0})
    idx = int(vp_state.get("idx", 0))

    # GLOBAL dedupe sets shared across centers
    seen_place_urls = set(global_cp.get("seen_place_urls", []))
    seen_place_names = set(global_cp.get("seen_place_names", []))

    print(f"Resuming at index {idx} (GLOBAL seen: {len(seen_place_names)} names, {len(seen_place_urls)} urls)")
    processed_since_save = 0

    while idx < max_places:
        # Heartbeat checkpoint so we can resume mid-iteration
        vp_state["idx"] = idx
        global_cp["viewports"][viewport_key] = vp_state
        global_cp["seen_place_urls"] = list(seen_place_urls)
        global_cp["seen_place_names"] = list(seen_place_names)
        save_global_checkpoint(global_cp)

        entries = feed.locator('a.hfpxzc[aria-label]')
        count_now = entries.count()

        if idx >= count_now:
            # Load more items in feed
            try:
                handle = feed.element_handle()
                page.evaluate("(el) => el.scrollBy(0, el.scrollHeight)", handle)
            except: pass
            page.wait_for_timeout(1800)

            entries = feed.locator('a.hfpxzc[aria-label]')
            if entries.count() == count_now:
                print("No more entries loaded; stopping.")
                break
            continue

        entry = entries.nth(idx)

        # Name from feed (fallback name)
        try:
            place_name = entry.get_attribute("aria-label") or f"place_{idx}"
        except:
            place_name = f"place_{idx}"

        # Skip duplicate by name (cheap pre-check)
        if place_name in seen_place_names:
            print(f"[{idx+1}/{max_places}] Skipping duplicate name: {place_name}")
            idx += 1
            vp_state["idx"] = idx
            global_cp["viewports"][viewport_key] = vp_state
            save_global_checkpoint(global_cp)
            continue

        print(f"\n[{idx+1}/{max_places}] Visiting: {place_name}")

        # Open place detail
        try:
            entry.scroll_into_view_if_needed()
            time.sleep(0.2)  # small human-like pause
            entry.click(timeout=5000)
        except:
            try:
                handle = feed.element_handle()
                page.evaluate("(el) => el.scrollBy(0, 600)", handle)
            except: pass
            idx += 1
            vp_state["idx"] = idx
            global_cp["viewports"][viewport_key] = vp_state
            save_global_checkpoint(global_cp)
            continue

        page.wait_for_timeout(2500)
        place_url = page.url.split("?")[0]
        if "/maps/place/" not in place_url:
            # Not a place detail (menu/collection)
            try:
                page.go_back(wait_until="domcontentloaded")
                page.wait_for_timeout(1000)
            except: pass
            idx += 1
            vp_state["idx"] = idx
            global_cp["viewports"][viewport_key] = vp_state
            save_global_checkpoint(global_cp)
            continue

        # Skip duplicate by URL
        if place_url in seen_place_urls:
            print(f"  -> Skip duplicate URL already scraped: {place_url}")
            idx += 1
            vp_state["idx"] = idx
            global_cp["viewports"][viewport_key] = vp_state
            save_global_checkpoint(global_cp)
            try:
                page.go_back(wait_until="domcontentloaded")
                page.wait_for_timeout(1200)
            except:
                page.goto(maps_search, wait_until="domcontentloaded")
                page.wait_for_selector('div[role="feed"]', timeout=8000)
                feed = page.locator('div[role="feed"]').first
            time.sleep(0.8)
            continue
        else:
            # Scrape
            try:
                rows = scrape_reviews_from_place(page, place_url, per_place_max_reviews, fallback_name=place_name)
            except Exception as e:
                print(f"  -> Error scraping this place: {e}")
                rows = []
            print(f"  -> Got {len(rows)} reviews")
            append_outputs(rows, csv_path, json_path)

            # Mark as seen + update checkpoint in-memory
            seen_place_urls.add(place_url)
            seen_place_names.add(place_name)

            # advance viewport index immediately after a processed place
            idx_next = idx + 1
            vp_state["idx"] = idx_next
            global_cp["viewports"][viewport_key] = vp_state

            # write GLOBAL sets (dedupe across centers)
            global_cp["seen_place_urls"] = list(seen_place_urls)
            global_cp["seen_place_names"] = list(seen_place_names)

            processed_since_save += 1
            processed_count += 1
            save_global_checkpoint(global_cp)
            idx = idx_next  # move local idx forward

        # Return to feed (or reopen search if needed)
        try:
            page.go_back(wait_until="domcontentloaded")
            page.wait_for_timeout(1200)
        except:
            page.goto(maps_search, wait_until="domcontentloaded")
            page.wait_for_selector('div[role="feed"]', timeout=8000)
            feed = page.locator('div[role="feed"]').first

        time.sleep(0.8)  # pacing

    # Final save
    vp_state["idx"] = idx
    global_cp["viewports"][viewport_key] = vp_state
    global_cp["seen_place_urls"] = list(seen_place_urls)
    global_cp["seen_place_names"] = list(seen_place_names)
    save_global_checkpoint(global_cp)
    print(f"Checkpoint saved for viewport {viewport_key} at idx={idx}.")
    return processed_count, (idx >= max_places)

# ---------- NEW: iterate across grid centers ----------
def scrape_across_centers(page, query: str, centers, max_places: int, per_place_max_reviews: int, csv_path: str, json_path: str):
    total_processed = 0
    global_cp = load_global_checkpoint(query)  # single global cp shared across centers

    for ci, center in enumerate(centers, 1):
        lat, lng, zoom = center
        print(f"\n=== Viewport {ci}/{len(centers)} @ {lat:.5f},{lng:.5f}, z={zoom} ===")
        processed, _ = scrape_query_feed(
            page, query, max_places, per_place_max_reviews, csv_path, json_path,
            center=center, global_cp=global_cp
        )
        # reload after each center to persist any external changes (optional)
        global_cp = load_global_checkpoint(query)
        total_processed += processed

    print(f"\nDone across centers. Total places processed: {total_processed}")

def save_outputs(rows, csv_path, json_path):
    if not rows:
        print("No reviews found.")
        return
    # Overwrite version (used for single-place run)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"Saved to {csv_path} and {json_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Google Maps reviews scraper (Playwright, no proxy).")
    ap.add_argument("target", nargs="?", default="Restaurants in Singapore",
                    help="Either a direct /maps/place/... URL or a search query (default: 'Restaurants in Singapore').")
    ap.add_argument("--per-place-max-reviews", type=int, default=MAX_REVIEWS_DEFAULT,
                    help="Max reviews to collect per place (default: 20).")
    ap.add_argument("--max-places", type=int, default=50,
                    help="Max number of places to visit when target is a query (default: 50).")
    ap.add_argument("--csv", default="reviews.csv", help="CSV output path (appends in multi-place mode).")
    ap.add_argument("--json", default="reviews.json", help="JSON output path (appends in multi-place mode).")
    ap.add_argument("--reset", action="store_true", help="Delete checkpoint and start fresh for this query.")
    args = ap.parse_args()

    # Optional reset
    if args.reset and Path(CHECKPOINT_PATH).exists():
        Path(CHECKPOINT_PATH).unlink(missing_ok=True)
        print("Checkpoint reset.")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        context = browser.new_context(locale="en-US", extra_http_headers={"Accept-Language": "en-US,en;q=0.9"})
        page = context.new_page()

        try:
            # If it's a direct URL, scrape only that place (overwrite outputs)
            if args.target.startswith("http"):
                place_url = args.target.split("?")[0]
                print("Using place URL:", place_url)
                rows = scrape_reviews_from_place(page, place_url, args.per_place_max_reviews)
                print(f"Collected {len(rows)} reviews")
                save_outputs(rows, args.csv, args.json)
            else:
                # Multi-place mode: iterate feed for this query and append outputs
                query = args.target
                print("Searching query across grid centers:", query)
                scrape_across_centers(page, query, GRID_CENTERS, args.max_places, args.per_place_max_reviews, args.csv, args.json)
        finally:
            print("Exit: latest checkpoint (if any) is saved.")
            browser.close()
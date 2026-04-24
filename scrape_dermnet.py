import os
import re
import time
import hashlib
import requests
import pandas as pd
from tqdm import tqdm
from urllib.parse import urljoin
from playwright.sync_api import sync_playwright

BASE_DIR = "/home/ubuntu/proj/dermnet"
IMG_DIR = os.path.join(BASE_DIR, "images")
META_PATH = os.path.join(BASE_DIR, "dermnet_metadata.xlsx")

SEARCH_TERM = "dermoscopy"
START_URL = "https://dermnetnz.org/images"

MAX_PAGES = 20          # test
SLEEP_SECONDS = 2       #

os.makedirs(IMG_DIR, exist_ok=True)


def clean_filename(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:80]


def download_image(url, save_path):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()

    with open(save_path, "wb") as f:
        f.write(r.content)


def guess_ext(url):
    url = url.split("?")[0].lower()
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        if url.endswith(ext):
            return ext
    return ".jpg"


def scrape_current_page(page, page_num):
    """
    visible image cards.
    return list of dicts: title, image_url, source_page_url
    """

    #
    page.wait_for_timeout(1500)

    items = page.evaluate("""
    () => {
        const results = [];

        const imgs = Array.from(document.querySelectorAll("img"));

        for (const img of imgs) {
            const src = img.currentSrc || img.src;
            if (!src) continue;

            const alt = img.alt || "";

            
            let node = img;
            let title = "";

            for (let i = 0; i < 6; i++) {
                if (!node.parentElement) break;
                node = node.parentElement;
                const text = node.innerText || "";
                const lines = text.split("\\n").map(x => x.trim()).filter(Boolean);

                // 
                const candidate = lines.find(x =>
                    x.length > 3 &&
                    !x.toLowerCase().includes("advert") &&
                    !x.toLowerCase().includes("showing") &&
                    !x.toLowerCase().includes("all images") &&
                    !x.toLowerCase().includes("galleries")
                );

                if (candidate) {
                    title = candidate;
                    break;
                }
            }

            if (!title && alt) title = alt;

            // logo / icon / divider / ad 
            const low = (title + " " + src).toLowerCase();
            if (
                low.includes("logo") ||
                low.includes("divider") ||
                low.includes("advert") ||
                low.includes("icon") ||
                src.startsWith("data:")
            ) {
                continue;
            }

            // DermNet image titles usually include useful caption text
            if (title.length < 4) continue;

            results.push({
                title: title,
                image_url: src,
                page_url: window.location.href
            });
        }

        //
        const seen = new Set();
        return results.filter(x => {
            const key = x.image_url;
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
        });
    }
    """)

    for item in items:
        item["page_num"] = page_num

    return items


def click_next_page(page):
    """
pagination aria-label / text / button。
    """
    candidates = [
        "a[aria-label='Next']",
        "button[aria-label='Next']",
        "a:has-text('Next')",
        "button:has-text('Next')",
        "a:has-text('›')",
        "button:has-text('›')",
        "a:has-text('→')",
        "button:has-text('→')",
    ]

    for selector in candidates:
        try:
            loc = page.locator(selector)
            if loc.count() > 0 and loc.first.is_visible():
                loc.first.click()
                page.wait_for_load_state("networkidle", timeout=15000)
                page.wait_for_timeout(1500)
                return True
        except Exception:
            pass

    # fallback
    try:
        buttons = page.locator("button, a")
        n = buttons.count()
        for i in range(n):
            txt = buttons.nth(i).inner_text(timeout=1000).strip()
            aria = buttons.nth(i).get_attribute("aria-label") or ""
            if "next" in aria.lower() or txt in [">", "→", "›"]:
                buttons.nth(i).click()
                page.wait_for_load_state("networkidle", timeout=15000)
                page.wait_for_timeout(1500)
                return True
    except Exception:
        pass

    return False


def main():
    all_rows = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1600, "height": 1000},
            user_agent="Mozilla/5.0"
        )
        page = context.new_page()

        print("Opening DermNet images page...")
        page.goto(START_URL, wait_until="networkidle", timeout=60000)

        print(f"Searching: {SEARCH_TERM}")
        search_box = page.locator("input[type='search'], input[placeholder*='Search'], input").first
        search_box.fill(SEARCH_TERM)
        page.keyboard.press("Enter")
        page.wait_for_load_state("networkidle", timeout=20000)
        page.wait_for_timeout(3000)

        for page_num in range(1, MAX_PAGES + 1):
            print(f"Scraping page {page_num}...")

            items = scrape_current_page(page, page_num)
            print(f"Found {len(items)} items on page {page_num}")

            all_rows.extend(items)

            ok = click_next_page(page)
            if not ok:
                print("No next page found. Stopping.")
                break

            time.sleep(SLEEP_SECONDS)

        browser.close()

    # remove duplicates
    unique = {}
    for row in all_rows:
        unique[row["image_url"]] = row
    rows = list(unique.values())

    print(f"Total unique images found: {len(rows)}")

    final_rows = []

    for idx, row in enumerate(tqdm(rows), start=1):
        image_id = f"dermnet_{idx:05d}"
        title = row["title"]
        img_url = row["image_url"]
        ext = guess_ext(img_url)

        filename = f"{image_id}_{clean_filename(title)}{ext}"
        save_path = os.path.join(IMG_DIR, filename)

        status = "downloaded"
        error = ""

        try:
            if not os.path.exists(save_path):
                download_image(img_url, save_path)
        except Exception as e:
            status = "failed"
            error = str(e)

        final_rows.append({
            "image_id": image_id,
            "filename": filename,
            "title": title,
            "image_url": img_url,
            "source_page_url": row["page_url"],
            "search_term": SEARCH_TERM,
            "page_num": row["page_num"],
            "status": status,
            "error": error,
            "source": "DermNet",
            "image_type": "dermoscopy"
        })

    df = pd.DataFrame(final_rows)
    df.to_excel(META_PATH, index=False)

    print(f"Saved images to: {IMG_DIR}")
    print(f"Saved metadata to: {META_PATH}")


if __name__ == "__main__":
    main()
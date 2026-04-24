import os
import re
import csv
import time
import urllib.request
from html.parser import HTMLParser
from urllib.parse import urljoin

BASE_DIR = "/home/ubuntu/proj/dermnet"
IMG_DIR = os.path.join(BASE_DIR, "images")
CSV_PATH = os.path.join(BASE_DIR, "dermnet_metadata.csv")

os.makedirs(IMG_DIR, exist_ok=True)

START_URL = "https://dermnetnz.org/images"
SEARCH_TERM = "dermoscopy"
MAX_PAGES = 10


class ImageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = []
        self.current_img = None
        self.current_text = []
        self.in_text = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        if tag == "img":
            src = attrs.get("src") or attrs.get("data-src")
            alt = attrs.get("alt", "")
            if src:
                self.current_img = {
                    "src": src,
                    "alt": alt,
                    "title": alt
                }

    def handle_data(self, data):
        text = data.strip()
        if text and self.current_img:
            if len(text) > 3:
                self.current_img["title"] = text
                self.images.append(self.current_img)
                self.current_img = None


def clean_filename(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:80]


def download_url(url):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.read()


def save_image(url, path):
    data = download_url(url)
    with open(path, "wb") as f:
        f.write(data)


def guess_ext(url):
    u = url.lower().split("?")[0]
    if u.endswith(".png"):
        return ".png"
    if u.endswith(".webp"):
        return ".webp"
    if u.endswith(".jpeg"):
        return ".jpeg"
    return ".jpg"


def parse_page(url):
    html = download_url(url).decode("utf-8", errors="ignore")

    parser = ImageParser()
    parser.feed(html)

    results = []
    seen = set()

    for item in parser.images:
        src = item["src"]
        title = item["title"]

        if not src or src.startswith("data:"):
            continue

        low = (src + " " + title).lower()
        if "logo" in low or "icon" in low or "advert" in low:
            continue

        if "dermoscopy" not in title.lower():
            continue

        full_url = urljoin(url, src)

        if full_url in seen:
            continue
        seen.add(full_url)

        results.append({
            "title": title,
            "image_url": full_url
        })

    return results


def main():
    all_rows = []
    image_count = 0

    for page_num in range(1, MAX_PAGES + 1):
        url = f"{START_URL}?q={SEARCH_TERM}&page={page_num}"
        print(f"Reading page {page_num}: {url}")

        try:
            items = parse_page(url)
        except Exception as e:
            print(f"Failed page {page_num}: {e}")
            continue

        print(f"Found {len(items)} images")

        for item in items:
            image_count += 1
            image_id = f"dermnet_{image_count:05d}"
            title = item["title"]
            img_url = item["image_url"]

            ext = guess_ext(img_url)
            filename = f"{image_id}_{clean_filename(title)}{ext}"
            save_path = os.path.join(IMG_DIR, filename)

            status = "downloaded"
            error = ""

            try:
                save_image(img_url, save_path)
            except Exception as e:
                status = "failed"
                error = str(e)

            all_rows.append({
                "image_id": image_id,
                "filename": filename,
                "title": title,
                "image_url": img_url,
                "search_page": url,
                "page_num": page_num,
                "status": status,
                "error": error,
                "source": "DermNet",
                "image_type": "dermoscopy"
            })

        time.sleep(2)

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id", "filename", "title", "image_url",
                "search_page", "page_num", "status", "error",
                "source", "image_type"
            ]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Done.")
    print(f"Images saved to: {IMG_DIR}")
    print(f"Metadata saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()
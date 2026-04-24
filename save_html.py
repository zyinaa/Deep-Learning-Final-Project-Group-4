import os
import subprocess
import time

BASE_DIR = "/home/ubuntu/proj/dermnet"
HTML_DIR = os.path.join(BASE_DIR, "html_pages")
os.makedirs(HTML_DIR, exist_ok=True)

SEARCH_TERM = "dermoscopy"
MAX_PAGES = 10

# chrome/chromium
BROWSER_CANDIDATES = [
    "google-chrome",
    "chromium",
    "chromium-browser"
]

browser = None
for b in BROWSER_CANDIDATES:
    result = subprocess.run(["which", b], capture_output=True, text=True)
    if result.returncode == 0:
        browser = result.stdout.strip()
        break

if browser is None:
    raise RuntimeError("No Chrome/Chromium browser found.")

print("Using browser:", browser)

for page_num in range(1, MAX_PAGES + 1):
    url = f"https://dermnetnz.org/images?search={SEARCH_TERM}&page={page_num}"
    out_path = os.path.join(HTML_DIR, f"dermnet_dermoscopy_page_{page_num:02d}.html")

    print(f"Saving page {page_num}: {url}")

    cmd = [
        browser,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--virtual-time-budget=8000",
        "--dump-dom",
        url
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    if result.returncode != 0:
        print("Failed:", result.stderr[:500])
        continue

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result.stdout)

    print("Saved:", out_path)
    time.sleep(2)

print("Done.")
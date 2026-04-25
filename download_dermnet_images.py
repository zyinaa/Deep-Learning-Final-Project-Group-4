import os
import json
import csv
import urllib.request
from urllib.parse import urlparse

JSON_DIR = "/home/ubuntu/proj/dermnet/jsonfile"
IMG_DIR = "/home/ubuntu/proj/dermnet/images"
CSV_PATH = os.path.join(IMG_DIR, "dermnet_metadata.csv")

os.makedirs(IMG_DIR, exist_ok=True)


def get_ext(url):
    path = urlparse(url).path.lower()
    if path.endswith(".png"):
        return ".png"
    if path.endswith(".webp"):
        return ".webp"
    if path.endswith(".jpeg"):
        return ".jpeg"
    return ".jpg"


def download_image(url, save_path):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        data = response.read()

    with open(save_path, "wb") as f:
        f.write(data)


def main():
    all_items = []

    json_files = sorted(
        [f for f in os.listdir(JSON_DIR) if f.endswith(".json")],
        key=lambda x: int("".join(filter(str.isdigit, x)) or 0)
    )

    print(f"Found {len(json_files)} json files.")

    for jf in json_files:
        json_path = os.path.join(JSON_DIR, jf)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            item["json_file"] = jf
            all_items.append(item)

    seen = set()
    unique_items = []

    for item in all_items:
        image_url = item.get("image_url", "").strip()
        title = item.get("title", "").strip()

        if not image_url:
            continue

        if image_url in seen:
            continue

        seen.add(image_url)

        unique_items.append({
            "image_url": image_url,
            "title": title,
            "json_file": item.get("json_file", "")
        })

    print(f"Total records before dedup: {len(all_items)}")
    print(f"Total unique images: {len(unique_items)}")

    rows = []

    for idx, item in enumerate(unique_items, start=1):
        image_id = f"dermnet_{idx:05d}"
        image_url = item["image_url"]
        title = item["title"]
        ext = get_ext(image_url)

        filename = f"{image_id}{ext}"
        save_path = os.path.join(IMG_DIR, filename)

        status = "downloaded"
        error = ""

        try:
            if not os.path.exists(save_path):
                download_image(image_url, save_path)
        except Exception as e:
            status = "failed"
            error = str(e)

        rows.append({
            "image_id": image_id,
            "filename": filename,
            "title": title,
            "image_url": image_url,
            "json_file": item["json_file"],
            "source": "DermNet",
            "image_type": "dermoscopy",
            "status": status,
            "error": error
        })

        if idx % 50 == 0:
            print(f"Processed {idx}/{len(unique_items)}")

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image_id",
            "filename",
            "title",
            "image_url",
            "json_file",
            "source",
            "image_type",
            "status",
            "error"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Done.")
    print(f"Images saved to: {IMG_DIR}")
    print(f"CSV saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()
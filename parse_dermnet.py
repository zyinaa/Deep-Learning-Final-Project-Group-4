import os
import re
import csv
import json
import html
import urllib.request
from urllib.parse import urljoin

BASE_DIR = "/home/ubuntu/proj/dermnet"
HTML_PATH = os.path.join(BASE_DIR, "dermnet_source.html")  #
IMG_DIR = os.path.join(BASE_DIR, "images")
CSV_PATH = os.path.join(BASE_DIR, "dermnet_metadata.csv")

BASE_URL = "https://dermnetnz.org"

os.makedirs(IMG_DIR, exist_ok=True)


def clean_filename(text):
    text = text or "untitled"
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:80]


def download_file(url, out_path):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        data = response.read()
    with open(out_path, "wb") as f:
        f.write(data)


def extract_objects(text):
    """
    """
    pattern = re.compile(
        r'\{\s*"a"\s*:\s*.*?\s*"q"\s*:\s*(?:true|false|null)\s*\}',
        re.DOTALL
    )

    objects = []
    for m in pattern.finditer(text):
        raw = m.group(0)

        try:
            obj = json.loads(raw)
            objects.append(obj)
        except Exception:
            #
            try:
                obj = json.loads(html.unescape(raw))
                objects.append(obj)
            except Exception:
                pass

    return objects


def is_dermoscopy(obj):
    fields = [
        obj.get("a"),
        obj.get("b"),
        obj.get("c"),
        obj.get("d"),
        obj.get("n"),
        obj.get("p"),
        obj.get("m"),
    ]
    joined = " ".join([str(x).lower() for x in fields if x])
    return ("dermoscopy" in joined) or ("dermatoscope" in joined)


def main():
    if not os.path.exists(HTML_PATH):
        raise FileNotFoundError(
            f"Cannot find {HTML_PATH}. Save the DermNet view-source HTML as dermnet_source.html first."
        )

    with open(HTML_PATH, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    objects = extract_objects(text)
    print("Total image-like objects found:", len(objects))

    dermoscopy_objects = [obj for obj in objects if is_dermoscopy(obj)]
    print("Dermoscopy-related objects found:", len(dermoscopy_objects))

    rows = []

    for idx, obj in enumerate(dermoscopy_objects, start=1):
        image_id = f"dermnet_{idx:05d}"

        condition = obj.get("a")
        title = obj.get("b") or obj.get("a") or obj.get("n")
        keywords = obj.get("c")
        description = obj.get("d")
        copyright_owner = obj.get("f")
        source_site = obj.get("g")
        license_text = obj.get("h")
        original_filename = obj.get("n")
        dermnet_numeric_id = obj.get("o")
        slug = obj.get("p")

        image_path = obj.get("l")
        detail_path = obj.get("m")

        image_url = urljoin(BASE_URL, image_path) if image_path else ""
        detail_url = urljoin(BASE_URL, detail_path) if detail_path else ""

        ext = os.path.splitext(original_filename or image_path or "")[1]
        if not ext:
            ext = ".jpg"

        filename = f"{image_id}_{clean_filename(title)}{ext}"
        out_path = os.path.join(IMG_DIR, filename)

        status = "downloaded"
        error = ""

        try:
            if image_url and not os.path.exists(out_path):
                download_file(image_url, out_path)
        except Exception as e:
            status = "failed"
            error = str(e)

        rows.append({
            "image_id": image_id,
            "filename": filename,
            "condition": condition,
            "title": title,
            "keywords": keywords,
            "description": description,
            "image_url": image_url,
            "detail_url": detail_url,
            "original_filename": original_filename,
            "dermnet_numeric_id": dermnet_numeric_id,
            "slug": slug,
            "copyright_owner": copyright_owner,
            "source_site": source_site,
            "license_text": license_text,
            "source": "DermNet",
            "image_type": "dermoscopy",
            "status": status,
            "error": error
        })

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image_id", "filename", "condition", "title", "keywords",
            "description", "image_url", "detail_url", "original_filename",
            "dermnet_numeric_id", "slug", "copyright_owner", "source_site",
            "license_text", "source", "image_type", "status", "error"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Done.")
    print("Images saved to:", IMG_DIR)
    print("Metadata saved to:", CSV_PATH)


if __name__ == "__main__":
    main()
import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

ARCHIVE_DIR = "../data/archive"
QUALITY = 80

def compress_image(filename):
    if not filename.endswith(".png"):
        return

    full_path = os.path.join(ARCHIVE_DIR, filename)
    new_filename = filename.replace(".png", ".jpeg")
    new_path = os.path.join(ARCHIVE_DIR, new_filename)

    try:
        with Image.open(full_path) as img:
            # Convert to RGB (PNG is RGBA, JPEG doesn't support transparency)
            rgb_img = img.convert('RGB')
            rgb_img.save(new_path, "JPEG", quality=QUALITY)
        
        # If successful, delete the huge PNG
        os.remove(full_path)
        print(f"Compressed: {filename}")
    except Exception as e:
        print(f"❌ Failed: {filename} ({e})")

files = os.listdir(ARCHIVE_DIR)
pngs = [f for f in files if f.endswith(".png")]

print(f"Found {len(pngs)} PNGs to compress. This will save huge space...")

# Run on 4 threads to go fast
with ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(compress_image, pngs)

print("Compression Complete!")


import os
from pathlib import Path
from datetime import datetime, timezone
import shutil

# Set the path to your directory containing .txt and .fits files
directory = Path("/mnt/fuuu/glint/20250513/dmptt")  # <-- Change this to your path

# Get all txt files in the directory
txt_files = sorted(directory.glob("*.txt"))

for txt_file in txt_files:
    with open(txt_file, "r") as f:
        # Skip comment lines
        for line in f:
            if not line.strip().startswith("#"):
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        timestamp = float(parts[3])
                        break
                    except ValueError:
                        continue

    # Convert timestamp to date string in UTC
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    date_str = dt.strftime("%Y%m%d")

    # Make new directory if needed
    date_folder = directory / date_str
    date_folder.mkdir(exist_ok=True)

    # Get base filename
    base = txt_file.stem
    fits_file = directory / (base + ".fits")

    # Move both files
    shutil.move(txt_file, date_folder / txt_file.name)
    if fits_file.exists():
        shutil.move(fits_file, date_folder / fits_file.name)
    else:
        print(f"⚠️ FITS file not found for {base}")

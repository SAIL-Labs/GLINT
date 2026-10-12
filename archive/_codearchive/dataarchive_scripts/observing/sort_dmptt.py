import os
from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+

# Replace with your target directory
directory = "/mnt/fuuu/glint/20250513/dmptt"
aest = ZoneInfo("Australia/Sydney")

# Get all files in the directory with their full paths
files = [os.path.join(directory, f) for f in os.listdir(directory)
         if os.path.isfile(os.path.join(directory, f))]

# Sort files by modification time
files_sorted = sorted(files, key=os.path.getctime)

# Print sorted list with AEST modification times
for f in files_sorted:
    ctime = datetime.fromtimestamp(os.path.getmtime(f), tz=aest)
    print(f"{f} -- Last modified: {ctime.strftime('%Y-%m-%d %H:%M:%S %Z')}")

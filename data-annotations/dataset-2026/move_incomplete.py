import os
import shutil

# Folder containing your images
source_folder = "dataset"

# Folder where the selected images will be moved
destination_folder = "incomplete"

# Range: 932.jpg through 7220.jpg, inclusive
start = 932
end = 7220

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

moved = 0
not_found = 0

for number in range(start, end + 1):
    filename = f"{number}.jpg"
    source = os.path.join(source_folder, filename)
    destination = os.path.join(destination_folder, filename)

    if os.path.isfile(source):
        shutil.move(source, destination)
        moved += 1
    else:
        not_found += 1

print(f"Done.")
print(f"Moved: {moved}")
print(f"Not found: {not_found}")
print(f"Range: {start}.jpg -> {end}.jpg")
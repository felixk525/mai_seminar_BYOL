import os
import random
import shutil

# directory for SSL4EO-12 dataset / subset dataset creation path
source_root = r"D:\todelete\0k_251k_uint8_jpeg_tif\rgb"
destination_dir = r"C:\PCodetmp\mai_seminar_byol\rgb_subset"

# Collect .tif files with 10k limit (to avoid waiting for excessive os walks)
subset_size = 10000 # Subset size
max_files_to_collect = 20000
all_images = []
for root, dirs, files in os.walk(source_root):
    for file in files:
        if file.lower().endswith(('.tif', '.tiff')):
            all_images.append(os.path.join(root, file))
            if len(all_images) >= max_files_to_collect:
                break
    if len(all_images) >= max_files_to_collect:
        break

print(f"Collected {len(all_images)} files.")


subset = random.sample(all_images, subset_size)

os.makedirs(destination_dir, exist_ok=True)

# Copy with unique prefix to avoid overwriting
for i, src_path in enumerate(subset):
    filename = os.path.basename(src_path)
    # Create unique filename: index + original name
    unique_name = f"{i:05d}_{filename}"
    dst_path = os.path.join(destination_dir, unique_name)
    shutil.copy2(src_path, dst_path)
    if i % 200 == 0:
        print(f"Copied {i} / {subset_size}")

print(f"Copied {subset_size} unique files to: {destination_dir}")



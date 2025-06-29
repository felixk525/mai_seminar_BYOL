import os

# Dataset test - load an image

rgb_path = r"D:\todelete\0k_251k_uint8_jpeg_tif\rgb"
image_files = []
max_samples = 10 

for root, dirs, files in os.walk(rgb_path):
    for file in files:
        if file.lower().endswith(('.tif', '.tiff')):
            image_files.append(os.path.join(root, file))
            if len(image_files) >= max_samples:
                break
    if len(image_files) >= max_samples:
        break

print(f"Found {len(image_files)} files.")

from PIL import Image
import matplotlib.pyplot as plt

# Use the first image path
sample_path = image_files[0]

img = Image.open(sample_path)

# Check number of channels
print("Mode:", img.mode)
print("Size:", img.size)

plt.imshow(img)
plt.title("Sample Image")
plt.axis('off')
plt.show()


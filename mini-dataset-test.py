import os
from PIL import Image
import matplotlib.pyplot as plt

# Test of local subset dataset
subset_path = r"C:\PCodetmp\mai_seminar_byol\rgb_subset"

# File list
image_files = [f for f in os.listdir(subset_path) if f.lower().endswith(('.tif', '.tiff'))]

# Load one image
sample_file = image_files[20]
sample_path = os.path.join(subset_path, sample_file)

img = Image.open(sample_path)

# Print metadata
print("Loaded image:", sample_file)
print("Mode:", img.mode)
print("Size:", img.size)

plt.imshow(img)
plt.title(sample_file)
plt.axis('off')
plt.show()

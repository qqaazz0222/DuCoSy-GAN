import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob

os.makedirs('debug', exist_ok=True)
target_file_dir = "/workspace/ct-dual-energy/DuCoSy-GAN/data/output/masked/Kyunghee_Univ/KP-0007/POST VUE"
dcm_files = glob(f"{target_file_dir}/*.dcm")
if not dcm_files:
    raise FileNotFoundError(f"No DICOM files found in {target_file_dir}")

for file in dcm_files:
    print(file)
    ds = pydicom.dcmread(file)
    instance_number = ds.InstanceNumber
    pixel_array = ds.pixel_array
    hu_image = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
    hu_image_normalized = 255 * (hu_image - np.min(hu_image)) / (np.max(hu_image) - np.min(hu_image))
    hu_image_normalized = hu_image_normalized.astype(np.uint8)
    img = Image.fromarray(hu_image_normalized)
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title('DICOM Image Visualization', fontsize=16)
    plt.savefig(f'debug/{instance_number}.png', dpi=300, bbox_inches='tight')

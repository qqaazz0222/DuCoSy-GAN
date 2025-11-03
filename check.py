import os
import numpy as np

target_dir = 'data/output/anonymized_pixel'

file_list = [f for f in os.listdir(target_dir) if f.endswith('.npy')]

for file_name in file_list:
    file_path = os.path.join(target_dir, file_name)
    volume = np.load(file_path)
    print(f"{file_name}: {volume.dtype}")
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# mapping_file = 'data/output/anonymization_mapping.csv'
# data_dir = 'data/output/anonymized_pixel'
# vis_dir = 'data/temp_vis'
# os.makedirs(vis_dir, exist_ok=True)

# mapping = {}
# with open(mapping_file, 'r') as f:
#     reader = csv.reader(f)
#     next(reader)  # Skip header
#     for row in reader:
#         if row[2] in mapping:
#             mapping[row[2]][row[0]] = row[3]
#         else:
#             mapping[row[2]] = {row[0]: row[3]}

# for pid in tqdm(mapping, desc="Generating Visualizations"):
#     cur_vis_dir = os.path.join(vis_dir, pid)
#     os.makedirs(cur_vis_dir, exist_ok=True)
    
#     original_volume_path = os.path.join(data_dir, f"{mapping[pid]['original']}.npy")
#     generated_volume_path = os.path.join(data_dir, f"{mapping[pid]['generated']}.npy")

#     original_volume = np.load(original_volume_path)
#     generated_volume = np.load(generated_volume_path)
    
#     num_slices = original_volume.shape[0]
    
#     for i in range(num_slices):
#         fig, axes = plt.subplots(1, 3, figsize=(15, 6))
#         # 원본 슬라이스
#         axes[0].imshow(original_volume[i], cmap='gray')
#         axes[0].set_title('Original Slice {}'.format(i))
#         axes[0].axis('off')
#         # 생성 슬라이스
#         axes[1].imshow(generated_volume[i], cmap='gray')
#         axes[1].set_title('Generated Slice {}'.format(i))
#         axes[1].axis('off')
#         # 오버랩 슬라이스
#         axes[2].imshow(original_volume[i], cmap='gray', alpha=0.5)
#         axes[2].imshow(generated_volume[i], cmap='jet', alpha=0.5)
#         axes[2].set_title('Overlay Slice {}'.format(i))
#         axes[2].axis('off')

#         plt.tight_layout()
#         plt.savefig(os.path.join(cur_vis_dir, '{}_slice_{}.png'.format(pid, i)))
#         plt.close()
        
from modules.nmodel.inference import nomalize_volume
os.makedirs('temp_diff', exist_ok=True)

data_list = []
for root, dirs, files in os.walk('data/working/test/'):
    for file in files:
        if file.endswith('.npy'):
            data_list.append(np.load(os.path.join(root, file)))

for idx, data_file in enumerate(data_list):
    data = nomalize_volume(data_file)

    for i in range(0, data.shape[0], 10):
        plt.imshow(data[i], cmap='gray')
        plt.tight_layout()
        plt.savefig(f'temp_diff/data_{idx}_slice_{i:03d}.png')
        plt.close()

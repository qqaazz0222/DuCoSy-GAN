import os
import numpy as np
from matplotlib import pyplot as plt

ncct_file = '/workspace/ct-dual-energy/DuCoSy-GAN/data/output/calculated/data/Kyunghee_Univ_KP-0007_vue.npy'
cect_file = '/workspace/ct-dual-energy/DuCoSy-GAN/data/output/calculated/data/Kyunghee_Univ_KP-0007_std.npy'

ncct_data = np.load(ncct_file)
cect_data = np.load(cect_file)

# best_match = None
# best_similarity = float('inf')

# instances = cect_data.copy()
# num_instances = cect_data.shape[0]

# print(f"Starting comparison between NCCT and CECT data, NCCT slices: {ncct_data.shape[0]}, CECT slices: {cect_data.shape[0]}")
# while num_instances > ncct_data.shape[0] - 32:
#     for shift in range(0, ncct_data.shape[0] - num_instances + 1):
#         check_data = ncct_data[shift:shift + num_instances]
#         similarity = np.mean(np.abs(check_data - instances))
#         print(f"Num Instances: {num_instances}, Shift: {shift}, Similarity: {similarity}")
#         if similarity < best_similarity:
#             best_similarity = similarity
#             best_match = (num_instances, shift)
#     instances = instances[1:-1]
#     num_instances = instances.shape[0]
    
# print(f"Best Match - Num Instances: {best_match[0]}, Shift: {best_match[1]}, Similarity: {best_similarity}")


for idx, (ncct_slice, cect_slice) in enumerate(zip(ncct_data, cect_data)):
    ncct_slice_image = ncct_slice.astype(np.int16)
    clipped_ncct_slice = np.clip(ncct_slice_image, -250, 450)
    cect_slice_image = cect_slice.astype(np.int16)
    clipped_cect_slice = np.clip(cect_slice_image, -250, 450)
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(clipped_ncct_slice, cmap='gray')
    axs[0].set_title(f'NCCT Slice {idx+1}')
    axs[0].axis('off')

    axs[1].imshow(clipped_cect_slice, cmap='gray')
    axs[1].set_title(f'CECT Slice {idx+1}')
    axs[1].axis('off')
    
    plt.tight_layout()
    plt.imsave(f'temp/{idx+1:03d}.png', np.hstack((ncct_slice_image, cect_slice_image)), cmap='gray')
    
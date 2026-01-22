"""
자동 마스크 생성 기능 테스트 스크립트
"""
import os
import sys
import torch
import pydicom
import numpy as np
import matplotlib.pyplot as plt

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.mask_generator import generate_anatomical_masks

def test_mask_generation():
    """단일 DICOM 파일에서 마스크 생성 테스트"""
    
    # 테스트 DICOM 파일 경로
    test_dicom = "/workspace/Contrast_CT/hyunsu/Dataset_DucosyGAN/Kangwon_National_Univ_Masked/00000102-1542-11-06/POST STD/1.2.410.200010.1129341.20090626110639.1.302.dcm"
    
    if not os.path.exists(test_dicom):
        print(f"Error: Test DICOM file not found: {test_dicom}")
        return
    
    print("="*80)
    print("Testing Automatic Mask Generation")
    print("="*80)
    
    # DICOM 파일 로드
    print(f"\n1. Loading DICOM file: {os.path.basename(test_dicom)}")
    dcm = pydicom.dcmread(test_dicom)
    
    # HU 값으로 변환
    print("2. Converting to HU values...")
    hu_image = dcm.pixel_array.astype(np.float32)
    hu_image = hu_image * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)
    print(f"   HU range: [{hu_image.min():.1f}, {hu_image.max():.1f}]")
    print(f"   Image shape: {hu_image.shape}")
    
    # 마스크 생성
    print("\n3. Generating anatomical masks...")
    mask_types = ['lung', 'mediastinum', 'bone', 'lung_vessel']
    masks = generate_anatomical_masks(hu_image, mask_types)
    
    # 결과 출력
    print("\n4. Mask generation results:")
    for mask_name, mask_array in masks.items():
        num_pixels = mask_array.sum()
        percentage = (num_pixels / mask_array.size) * 100
        print(f"   - {mask_name:15s}: {int(num_pixels):7d} pixels ({percentage:5.2f}%)")
    
    # 시각화
    print("\n5. Saving visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # HU 이미지 (windowing 적용)
    wc, ww = 40, 400
    hu_windowed = np.clip((hu_image - (wc - ww/2)) / ww, 0, 1)
    axes[0, 0].imshow(hu_windowed, cmap='gray')
    axes[0, 0].set_title('Original HU Image\n(Window: 40/400)')
    axes[0, 0].axis('off')
    
    # 마스크들
    mask_titles = {
        'lung': 'Lung Mask',
        'mediastinum': 'Mediastinum Mask',
        'bone': 'Bone Mask',
        'lung_vessel': 'Lung Vessel Mask'
    }
    
    for idx, (mask_name, mask_array) in enumerate(masks.items()):
        row = (idx + 1) // 3
        col = (idx + 1) % 3
        axes[row, col].imshow(mask_array, cmap='hot', vmin=0, vmax=1)
        axes[row, col].set_title(mask_titles.get(mask_name, mask_name))
        axes[row, col].axis('off')
    
    # 오버레이 이미지
    overlay = np.stack([hu_windowed]*3, axis=-1)
    alpha = 0.5
    if 'bone' in masks:
        overlay[masks['bone'] > 0] = overlay[masks['bone'] > 0] * (1-alpha) + np.array([0, 0, 1]) * alpha
    if 'lung' in masks:
        overlay[masks['lung'] > 0] = overlay[masks['lung'] > 0] * (1-alpha) + np.array([1, 0, 0]) * alpha
    if 'mediastinum' in masks:
        overlay[masks['mediastinum'] > 0] = overlay[masks['mediastinum'] > 0] * (1-alpha) + np.array([1, 1, 0]) * alpha
    if 'lung_vessel' in masks:
        overlay[masks['lung_vessel'] > 0] = overlay[masks['lung_vessel'] > 0] * (1-alpha) + np.array([0, 1, 0]) * alpha
    
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title('Combined Overlay\n(Blue=Bone, Red=Lung, Yellow=Mediastinum, Green=Vessel)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    output_path = './test_mask_generation.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Visualization saved to: {output_path}")
    
    print("\n" + "="*80)
    print("Test completed successfully!")
    print("="*80)


if __name__ == "__main__":
    test_mask_generation()

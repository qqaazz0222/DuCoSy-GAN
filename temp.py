import nibabel as nib
import numpy as np

test_file = '/workspace/ct-dual-energy/DuCoSy-GAN/data/output/masked/Kangwon_National_Univ/00000102-1542-11-06.nii'

# NIfTI 파일 로드
print(f"Loading file: {test_file}")
nifti_img = nib.load(test_file)

# 기본 정보 출력
print("\n" + "="*80)
print("NIfTI File Information")
print("="*80)

# 데이터 배열
data = nifti_img.get_fdata()
print(f"\nData shape: {data.shape}")
print(f"Data type: {data.dtype}")
print(f"Data min: {np.min(data)}")
print(f"Data max: {np.max(data)}")
print(f"Data mean: {np.mean(data):.2f}")
print(f"Data std: {np.std(data):.2f}")

# 고유값 확인 (세그멘테이션 레이블인 경우)
unique_values = np.unique(data)
print(f"\nNumber of unique values: {len(unique_values)}")
if len(unique_values) < 200:  # 세그멘테이션 마스크인 경우
    print(f"Unique values (labels): {unique_values}")
    print("\nLabel distribution:")
    for val in unique_values:
        count = np.sum(data == val)
        percentage = (count / data.size) * 100
        print(f"  Label {int(val):3d}: {count:10d} voxels ({percentage:6.2f}%)")
else:
    print(f"First 20 unique values: {unique_values[:20]}")
    print(f"Last 20 unique values: {unique_values[-20:]}")

# Affine 정보
print("\n" + "="*80)
print("Affine Matrix:")
print("="*80)
print(nifti_img.affine)

# Voxel spacing 정보
pixdim = nifti_img.header.get_zooms()
print(f"\nVoxel dimensions (spacing): {pixdim}")

# Header 정보
print("\n" + "="*80)
print("Header Information:")
print("="*80)
print(f"Data type: {nifti_img.header.get_data_dtype()}")
print(f"Dimensions: {nifti_img.header.get_data_shape()}")
print(f"qform code: {nifti_img.header['qform_code']}")
print(f"sform code: {nifti_img.header['sform_code']}")

# 슬라이스별 통계 (중간 슬라이스)
if len(data.shape) == 3:
    mid_slice_idx = data.shape[2] // 2
    mid_slice = data[:, :, mid_slice_idx]
    print(f"\n" + "="*80)
    print(f"Middle slice (z={mid_slice_idx}) statistics:")
    print("="*80)
    print(f"Min: {np.min(mid_slice):.2f}")
    print(f"Max: {np.max(mid_slice):.2f}")
    print(f"Mean: {np.mean(mid_slice):.2f}")
    print(f"Non-zero voxels: {np.count_nonzero(mid_slice)}")
    
print("\n" + "="*80)



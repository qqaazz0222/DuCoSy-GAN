# 심장에서 폐로 나가는 혈관에 대해 불필요한 마스크를 제거하는 스크립트
# PLASS LAB. Hyunsu Kim, M.S. Student

import os
import cv2
import shutil
import pydicom
import numpy as np
import nibabel as nib
from scipy import ndimage
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial

from tqdm import tqdm
from glob import glob

def clear_temp_files():
    """macOS에서 생성되는 임시 파일(._* 형식)을 제거하는 함수"""
    if os.name != 'posix':
        return # macOS가 아닌 경우 종료
    target_dir = '.'
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if file.startswith('._'):
                file_path = os.path.join(root, file)
                os.remove(file_path)

def raw_to_hu(slices):
    """DICOM 슬라이스를 Hounsfield Unit(HU) 값으로 변환"""
    image = np.stack([s.pixel_array for s in slices]).astype(np.int16)
    intercept = slices[0].RescaleIntercept
    slope = slices[0].RescaleSlope
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
    image += np.int16(intercept)
    return np.array(image, dtype=np.int16)

def load_dicom_volume(dicom_path):
    """DICOM 파일들을 로드하여 3D 볼륨으로 변환"""
    dicom_files = sorted(glob(os.path.join(dicom_path, '*.dcm')))
    if not dicom_files:
        return None
    slices = [pydicom.dcmread(f) for f in dicom_files]
    slices.sort(key=lambda x: float(x.InstanceNumber))
    volume = raw_to_hu(slices)
    volume = np.transpose(volume, (2, 1, 0))
    return volume

def process_file(file, dicom_dir_root, mask_dir, modified_mask_dir):
    """개별 파일을 처리하는 함수 (병렬 처리용)"""
    name = os.path.splitext(file)[0]
    dicom_path = os.path.join(dicom_dir_root, name, 'POST VUE')
    dicom_volume = load_dicom_volume(dicom_path)
    if dicom_volume is None:
        return None
    mask_path = os.path.join(mask_dir, file)
    modified_mask_path = os.path.join(modified_mask_dir, file)
    return (name, {
        'dicom_volume': dicom_volume,
        'mask_path': mask_path, 
        'modified_mask_path': modified_mask_path
    })

def load_data(dicom_dir_root, mask_dir, modified_mask_dir):
    """DICOM 이미지와 마스크 데이터를 로드하는 함수"""
    # 마스크 디렉토리에서 모든 .nii 파일 로드
    files = [f for f in os.listdir(mask_dir) if f.endswith('.nii')]
    data = {}
    
    # 병렬 처리로 데이터 로드
    num_processes = max(1, cpu_count() - 1)
    with Pool(num_processes) as pool:
        process_fn = partial(process_file, dicom_dir_root=dicom_dir_root, 
                           mask_dir=mask_dir, modified_mask_dir=modified_mask_dir)
        results = list(tqdm(pool.imap_unordered(process_fn, files), 
                           total=len(files), desc="Loading data"))
    
    for result in results:
        if result is not None:
            name, file_data = result
            data[name] = file_data
    
    return data

def modify_heart_mask(dicom_volume, mask_path, modified_mask_path, sample_dir):
    """심장 마스크에서 불필요한 부분을 제거하는 함수"""

    mask_data = nib.load(mask_path)
    mask_volume = mask_data.get_fdata().astype(np.uint8)  # Load as uint8 to save memory
    heart_mask = (mask_volume == 51).astype(np.uint8)  # 심장 마스크 추출 (값이 51인 부분)
    heart_mask_original = heart_mask.copy()  # 원본 마스크 백업
    
    # 심장 마스크의 z 방향 범위 계산
    z_min = np.min(np.nonzero(heart_mask)[2])
    z_max = np.max(np.nonzero(heart_mask)[2])

    # 연결된 성분 라벨링
    labeled_array, num_features = ndimage.label(heart_mask)

    # 각 클러스터의 무게중심 찾기
    centers = ndimage.center_of_mass(heart_mask, labeled_array, range(1, num_features + 1))
    centers.sort(key=lambda x: x[2])  # z 좌표 기준으로 정렬
    
    gap_threshold = 2  # 마스크 간격 임계값
    if len(centers) > 0:
        start_z = int(centers[0][2])
        cut_z = -1
        
        # x, y 좌표마다 z 방향에서 마스크 간격 검사
        for x_coord in range(heart_mask.shape[0]):
            for y_coord in range(heart_mask.shape[1]):
                gap_count = 0
                for z in range(start_z, heart_mask.shape[2]):
                    if heart_mask[x_coord, y_coord, z] == 0:
                        gap_count += 1
                    else:
                        gap_count = 0
                    
                    if gap_count >= gap_threshold:
                        cut_z = z - gap_count + 1
                        heart_mask[x_coord, y_coord, cut_z:] = 0
                        break

    # 심장의 중심을 기준으로 불필요한 혈관 제거
    if len(centers) > 0:
        x, y, z = centers[0]
        cur_slice = heart_mask[:, :, int(z)].copy()
        max_distnace = 0
        offset = 1.15  # 거리 오프셋
        offset_y_base = 1.4  # y축 거리 기본 오프셋
        offset_z = 2.65  # z축 거리 오프셋
        
        # 현재 슬라이스에서 최대 거리 계산
        nonzero_i, nonzero_j = np.nonzero(cur_slice)
        if len(nonzero_i) > 0:
            distances_slice = np.sqrt((nonzero_i - x)**2 + (nonzero_j - y)**2)
            max_distnace = np.max(distances_slice) * offset
            
            # 최대 거리 이상의 복셀 제거 (벡터화 연산)
            i_indices, j_indices, k_indices = np.nonzero(heart_mask)
            x_distances = i_indices - x
            y_distances = j_indices - y
            z_distances = k_indices - z
            
            # x축 거리에 따라 가변적인 offset_y 적용
            # x축 거리가 멀수록 offset_y가 강하게 적용됨
            offset_y_variable = 1 + (offset_y_base - 1) * np.abs(x_distances) / (np.max(np.abs(x_distances)) + 1e-5)
            
            # 3D 거리 계산 (가변적 오프셋 적용)
            distances = np.sqrt((i_indices - x)**2 + 
                            np.where((y_distances > 0) & (z_distances > 0), (y_distances * offset_y_variable)**2, y_distances**2) +
                            np.where(z_distances > 0, (z_distances * offset_z)**2, z_distances**2))
            remove_mask = distances >= max_distnace
            heart_mask[i_indices[remove_mask], j_indices[remove_mask], k_indices[remove_mask]] = 0
            
    # 연결된 영역 찾기 및 사이즈 출력
    labeled_array, num_features = ndimage.label(heart_mask)
    region_sizes = ndimage.sum(heart_mask, labeled_array, range(1, num_features + 1))
    region_size_threshold = 1024  # 최소 크기 임계값

    for region_id in range(1, num_features + 1):
        size = region_sizes[region_id - 1]
        region_coords = np.nonzero(labeled_array == region_id)
        if size < region_size_threshold:
            heart_mask[region_coords] = 0  # 작은 영역 제거
            
        
    # DICOM 볼륨을 사용하여 샘플 이미지 저장
    # DICOM 볼륨을 시각화를 위해 0-255 범위로 정규화
    dicom_normalized = dicom_volume.copy().astype(np.float32)
    dicom_min = np.percentile(dicom_normalized, 2)
    dicom_max = np.percentile(dicom_normalized, 98)
    dicom_normalized = np.clip((dicom_normalized - dicom_min) / (dicom_max - dicom_min + 1e-5) * 255, 0, 255).astype(np.uint8)
    
    # DICOM 볼륨에서 RGB 이미지 생성
    volume_rgb = np.stack([dicom_normalized, dicom_normalized, dicom_normalized], axis=-1).astype(np.uint8)
    volume_rgb[heart_mask_original == 1] = [255, 0, 0]  # 원본 심장 마스크는 빨간색
    volume_rgb[heart_mask == 1] = [0, 255, 0]  # 수정된 심장 마스크는 녹색
    
    # # 샘플 이미지 저장 (20개 슬라이스)
    # num_samples = 20
    # z_mid = (z_min + z_max) // 2
    # step = max(1, (z_max - z_mid) // num_samples)
    # for z in range(z_mid, z_max + 1, step):
    #     cur_sample_slice = volume_rgb[:, :, z]
    #     w, h = cur_sample_slice.shape[1], cur_sample_slice.shape[0]
    #     cur_sample_slice = cur_sample_slice.transpose(1, 0, 2)
    #     plt.figure(figsize=(w/100, h/100), dpi=100)
    #     plt.imshow(cur_sample_slice)
    #     plt.axis('off')
    #     plt.savefig(os.path.join(sample_dir, f'slice_{z:03d}.png'), bbox_inches='tight', pad_inches=0, dpi=100)
    #     plt.close()
                            
    # 수정된 마스크를 원본 마스크 볼륨에 저장
    mask_volume[mask_volume == 51] = 0
    mask_volume[heart_mask == 1] = 51

    # 압축된 NIfTI 형식으로 저장
    img = nib.Nifti1Image(mask_volume, mask_data.affine)
    nib.save(img, modified_mask_path)
    
def process_mask(args):
    """개별 마스크를 처리하는 함수 (병렬 처리용)"""
    file_name, paths, sample_dir = args
    cur_sample_dir = os.path.join(sample_dir, file_name)
    os.makedirs(cur_sample_dir, exist_ok=True)
    modify_heart_mask(paths['dicom_volume'], paths['mask_path'], paths['modified_mask_path'], cur_sample_dir)
    return file_name

def main():
    """메인 함수: 모든 심장 마스크를 수정하고 저장"""
    clear_temp_files() # macOS 사용자를 위한 임시 파일 정리
    
    # 데이터 디렉토리 설정
    dataset_name = 'Kyunghee_Univ'  # 데이터셋 이름 설정
    dicom_dir = f'./data/input/{dataset_name}'
    mask_dir = f'./data/output/mask/{dataset_name}'
    if os.path.exists(dicom_dir) is False or os.path.exists(mask_dir) is False:
        print("DICOM 또는 마스크 디렉토리가 존재하지 않습니다. 경로를 확인해주세요.")
        return
    modified_mask_dir = './data/output/modified_mask'
    os.makedirs(modified_mask_dir, exist_ok=True)
    modified_mask_dir = os.path.join(modified_mask_dir, dataset_name)
    os.makedirs(modified_mask_dir, exist_ok=True)
    sample_dir = './sample_images'
    if os.path.exists(sample_dir):
        shutil.rmtree(sample_dir)
    os.makedirs(sample_dir, exist_ok=True)
    
    # 데이터 로드
    data = load_data(dicom_dir, mask_dir, modified_mask_dir)
    
    # 각 환자의 심장 마스크를 병렬로 수정
    num_processes = max(1, min(cpu_count() - 1, 8))  # 최대 8개 프로세스 사용
    with Pool(num_processes) as pool:
        process_args = [(file_name, paths, sample_dir) for file_name, paths in data.items()]
        list(tqdm(pool.imap_unordered(process_mask, process_args), 
                 total=len(data), desc="Modifying heart masks"))
        
    clear_temp_files() # macOS 사용자를 위한 임시 파일 정리

if __name__ == "__main__":
    main()
import os
import glob
import pydicom
import numpy as np
import torch
from torch.utils.data import Dataset
from modules.preprocess import apply_hu_transform
from modules.mask_generator import generate_anatomical_masks
from tqdm import tqdm
import warnings

# pydicom 라이브러리에서 발생하는 사용자 경고를 무시
warnings.filterwarnings("ignore", category=UserWarning)


def load_mask_from_dicom(mask_path):
    """마스크 DICOM 파일을 로드하여 [0, 1] 범위의 numpy 배열로 반환"""
    if not os.path.exists(mask_path):
        return None
    try:
        dcm = pydicom.dcmread(mask_path)
        mask = dcm.pixel_array.astype(np.float32)
        # 이진 마스크로 정규화 (0 또는 1)
        mask = (mask > 0).astype(np.float32)
        return mask
    except Exception as e:
        print(f"Warning: Failed to load mask from {mask_path}: {e}")
        return None


def generate_masks_from_hu(hu_image, mask_types):
    """
    HU 이미지로부터 해부학적 마스크 자동 생성
    
    Args:
        hu_image: HU 값의 numpy 배열
        mask_types: 생성할 마스크 종류 리스트 (e.g., ['lung', 'mediastinum', 'bone', 'lung_vessel'])
    
    Returns:
        dict: {mask_name: mask_tensor} 형태의 딕셔너리
    """
    if not mask_types:
        return {}
    
    try:
        # HU 이미지로부터 마스크 생성
        masks_dict = generate_anatomical_masks(hu_image, mask_types)
        
        # numpy 배열을 torch 텐서로 변환
        masks_tensors = {}
        for mask_name, mask_array in masks_dict.items():
            masks_tensors[mask_name] = torch.from_numpy(mask_array.astype(np.float32))
        
        return masks_tensors
    except Exception as e:
        print(f"Warning: Failed to generate masks from HU image: {e}")
        return {}
      
        
# ---- 데이터셋 및 유틸리티 함수 -----
class DicomDataset(Dataset):
    """ DICOM 파일 쌍을 로드하는 커스텀 데이터셋 (자동 마스크 생성 지원) """
    def __init__(self, patient_dirs, args, transform=None):
        self.transform = transform
        self.args = args
        self.paired_files = []
        self.use_masks = getattr(args, 'use_masks', False)
        self.auto_generate_masks = getattr(args, 'auto_generate_masks', False)  # 자동 마스크 생성 옵션
        self.mask_types = getattr(args, 'mask_types', ['lung', 'mediastinum', 'bone', 'lung_vessel'])  # 생성할 마스크 타입
        self.mask_folders = getattr(args, 'mask_folders', [])
        
        for patient_dir in tqdm(patient_dirs, desc="데이터 처리 중"):
            ncct_path = os.path.join(patient_dir, args.ncct_folder)
            cect_path = os.path.join(patient_dir, args.cect_folder)
            
            ncct_files = sorted(glob.glob(os.path.join(ncct_path, "*.dcm")))
            cect_files = sorted(glob.glob(os.path.join(cect_path, "*.dcm")))

            if not ncct_files or not cect_files: continue

            try:
                ncct_files.sort(key=lambda x: int(pydicom.dcmread(x, stop_before_pixels=True).InstanceNumber))
                cect_files.sort(key=lambda x: int(pydicom.dcmread(x, stop_before_pixels=True).InstanceNumber))
            except (AttributeError, KeyError, ValueError):
                try:
                    ncct_files.sort(key=lambda x: float(pydicom.dcmread(x, stop_before_pixels=True).SliceLocation))
                    cect_files.sort(key=lambda x: float(pydicom.dcmread(x, stop_before_pixels=True).SliceLocation))
                except (AttributeError, KeyError):
                    print(f"Warning: InstanceNumber/SliceLocation not found in {patient_dir}. Falling back to filename sort.")
                    pass

            for ncct_file, cect_file in zip(ncct_files, cect_files):
                # 마스크 파일 경로 찾기 (선택적, auto_generate_masks가 False일 때만)
                mask_paths = {}
                if self.use_masks and not self.auto_generate_masks:
                    for mask_name in self.mask_folders:
                        mask_folder_path = os.path.join(patient_dir, mask_name)
                        if os.path.exists(mask_folder_path):
                            # NCCT 파일명과 동일한 마스크 파일 찾기
                            mask_file = os.path.join(mask_folder_path, os.path.basename(ncct_file))
                            if os.path.exists(mask_file):
                                mask_paths[mask_name] = mask_file
                
                self.paired_files.append((ncct_file, cect_file, mask_paths))
                
    def __len__(self): 
        return len(self.paired_files)
    
    def __getitem__(self, index):
        ncct_path, cect_path, mask_paths = self.paired_files[index]
        ncct_dcm, cect_dcm = pydicom.dcmread(ncct_path), pydicom.dcmread(cect_path)
        
        # HU 이미지 생성 (마스크 생성을 위해 원본 HU 값 필요)
        ncct_hu_image = ncct_dcm.pixel_array.astype(np.float32)
        ncct_hu_image = ncct_hu_image * float(ncct_dcm.RescaleSlope) + float(ncct_dcm.RescaleIntercept)
        
        # HU transform with soft squeezing (모델 입력용)
        use_soft_squeezing = getattr(self.args, 'use_soft_squeezing', True)
        ncct_img = apply_hu_transform(ncct_dcm, self.args.hu_min, self.args.hu_max, use_soft_squeezing)
        cect_img = apply_hu_transform(cect_dcm, self.args.hu_min, self.args.hu_max, use_soft_squeezing)
        
        if self.transform:
            ncct_img = self.transform(ncct_img)
            cect_img = self.transform(cect_img)
        
        result = {"A": ncct_img, "B": cect_img}
        
        # 마스크 처리
        if self.use_masks:
            if self.auto_generate_masks:
                # 자동으로 마스크 생성
                masks_dict = generate_masks_from_hu(ncct_hu_image, self.mask_types)
                
                if masks_dict:
                    # 마스크 순서대로 결합 (mask_types 순서 유지)
                    masks = []
                    for mask_type in self.mask_types:
                        if mask_type in masks_dict:
                            mask = masks_dict[mask_type]
                            # 자동 생성된 마스크는 이미 torch.Tensor이므로
                            # 크기만 조정 (ToTensor 변환 제외)
                            if mask.dim() == 2:  # [H, W]
                                mask = mask.unsqueeze(0)  # [1, H, W]로 변환
                            # 크기 조정이 필요한 경우 (ncct_img와 크기가 다른 경우)
                            if mask.shape[-2:] != ncct_img.shape[-2:]:
                                mask = torch.nn.functional.interpolate(
                                    mask.unsqueeze(0), 
                                    size=ncct_img.shape[-2:], 
                                    mode='nearest'
                                ).squeeze(0)
                            masks.append(mask)
                        else:
                            # 생성 실패 시 0으로 채운 마스크
                            masks.append(torch.zeros_like(ncct_img))
                    
                    if masks:
                        # 마스크들을 채널로 결합
                        result["masks"] = torch.cat(masks, dim=0)
            else:
                # 기존 방식: 파일에서 마스크 로드
                if mask_paths:
                    masks = []
                    for mask_name in self.mask_folders:
                        if mask_name in mask_paths:
                            mask = load_mask_from_dicom(mask_paths[mask_name])
                            if mask is not None:
                                if self.transform:
                                    mask = self.transform(mask)
                                masks.append(mask)
                            else:
                                # 마스크 로드 실패 시 0으로 채운 마스크 사용
                                masks.append(torch.zeros_like(ncct_img))
                        else:
                            # 마스크 파일이 없으면 0으로 채운 마스크 사용
                            masks.append(torch.zeros_like(ncct_img))
                    
                    if masks:
                        # 마스크들을 채널로 결합
                        result["masks"] = torch.cat(masks, dim=0)
        
        return result



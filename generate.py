import os
import glob
import shutil
import pydicom
import traceback
import numpy as np
import torch
import torchvision.transforms as transforms
from copy import deepcopy
from tqdm import tqdm

from modules.argmanager import get_common_infer_args, get_soft_tissue_infer_args, get_lung_infer_args
from modules.model import Generator
from modules.preprocess import preprocess_dicom, postprocess_tensor
from modules.postprocess import postprocess_ct_volume

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")

def generate(args, soft_tissue_args, lung_args):
    """Soft-tissue 및 Lung CycleGAN 모델을 사용하여 DICOM 파일을 추론하고 통합"""
    
    # GPU 장치 설정
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    
    # 모델 초기화
    soft_tissue_model = Generator(input_channels=1, num_residual_blocks=9).to(device)
    lung_model = Generator(input_channels=1, num_residual_blocks=9).to(device)
    print(f"Using Checkpoint: Soft Tissue: {soft_tissue_args.model_path}, Lung: {lung_args.model_path}")
    
    # 가중치 로드
    soft_tissue_state_dict = torch.load(soft_tissue_args.model_path, map_location=device)
    lung_state_dict = torch.load(lung_args.model_path, map_location=device)
    
    # DataParallel로 학습된 모델의 경우 'module.' 접두사 제거
    if all(key.startswith('module.') for key in soft_tissue_state_dict.keys()):
        print("Soft Tissue Model was trained with DataParallel. Removing 'module.' prefix.")
        soft_tissue_state_dict = {k.replace('module.', ''): v for k, v in soft_tissue_state_dict.items()}
    if all(key.startswith('module.') for key in lung_state_dict.keys()):
        print("Lung Model was trained with DataParallel. Removing 'module.' prefix.")
        lung_state_dict = {k.replace('module.', ''): v for k, v in lung_state_dict.items()}
        
    # 모델에 가중치 로드 및 평가 모드 설정
    soft_tissue_model.load_state_dict(soft_tissue_state_dict)
    lung_model.load_state_dict(lung_state_dict)
    soft_tissue_model.eval()
    lung_model.eval()
    
    # 이미지 변환 (리사이즈)
    transform = transforms.Resize((args.img_size, args.img_size), antialias=True)
    
    # 추론 시작
    with torch.no_grad():
        # 각 데이터셋 폴더 순회
        for dataset_name in args.dataset_names:
            # 입력 및 출력 디렉토리 설정
            input_dir = os.path.join(args.input_dir_root, dataset_name)
            working_dir = os.path.join(args.working_dir_root, dataset_name)
            
            print(f"\nProcessing dataset: {dataset_name}")

            # 환자 디렉토리 검색
            patient_dirs = sorted([d for d in glob.glob(os.path.join(input_dir, '*')) if os.path.isdir(d)])
            
            # 환자 디렉토리 순회
            for patient_dir in tqdm(patient_dirs, desc=f"Patients in {dataset_name}"):
                # 환자 ID 추출
                patient_id = os.path.basename(patient_dir)
                ncct_folder_path = os.path.join(patient_dir, args.ncct_folder)
                
                if not os.path.isdir(ncct_folder_path):
                    continue

                # 출력 폴더 생성 (soft_tissue 및 lung 서브폴더 포함)
                working_patient_folder = os.path.join(working_dir, patient_id)
                working_raw_folder = os.path.join(working_patient_folder, "raw")
                working_soft_tissue_folder = os.path.join(working_patient_folder, "soft_tissue")
                working_lung_folder = os.path.join(working_patient_folder, "lung")
                os.makedirs(working_patient_folder, exist_ok=True)
                os.makedirs(working_raw_folder, exist_ok=True)
                os.makedirs(working_soft_tissue_folder, exist_ok=True)
                os.makedirs(working_lung_folder, exist_ok=True)

                # DICOM 파일 검색
                dcm_files = sorted(glob.glob(os.path.join(ncct_folder_path, "*.dcm")))
                
                for dcm_path in dcm_files:
                    try:
                        image_soft_tissue_tensor, image_lung_tensor, original_dcm = preprocess_dicom(dcm_path, soft_tissue_args.hu_min, soft_tissue_args.hu_max, lung_args.hu_min, lung_args.hu_max)
                        original_size = (original_dcm.Rows, original_dcm.Columns)

                        input_soft_tissue_tensor = transform(image_soft_tissue_tensor.unsqueeze(0)).to(device)
                        input_lung_tensor = transform(image_lung_tensor.unsqueeze(0)).to(device)
                        output_soft_tissue_tensor = soft_tissue_model(input_soft_tissue_tensor)
                        output_lung_tensor = lung_model(input_lung_tensor)

                        output_soft_tissue_tensor_resized = transforms.functional.resize(output_soft_tissue_tensor, original_size, antialias=True)
                        output_lung_tensor_resized = transforms.functional.resize(output_lung_tensor, original_size, antialias=True)
                        soft_tissue_new_pixel_data = postprocess_tensor(output_soft_tissue_tensor_resized, original_dcm, soft_tissue_args.hu_min, soft_tissue_args.hu_max)
                        lung_new_pixel_data = postprocess_tensor(output_lung_tensor_resized, original_dcm, lung_args.hu_min, lung_args.hu_max)

                        soft_tissue_raw_range = (np.min(soft_tissue_new_pixel_data), np.max(soft_tissue_new_pixel_data))
                        lung_raw_range = (np.min(lung_new_pixel_data), np.max(lung_new_pixel_data))

                        output_dcm = pydicom.dcmread(dcm_path)
                        output_dcm.SeriesDescription = f"Synthetic CECT (from {original_dcm.SeriesDescription})"
                        output_dcm.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
                        
                        output_soft_tissue_dcm = deepcopy(output_dcm)
                        output_lung_dcm = deepcopy(output_dcm)
                    
                        output_soft_tissue_dcm.SmallestImagePixelValue = int(soft_tissue_raw_range[0])
                        output_soft_tissue_dcm.LargestImagePixelValue = int(soft_tissue_raw_range[1])
                        
                        output_lung_dcm.SmallestImagePixelValue = int(lung_raw_range[0])
                        output_lung_dcm.LargestImagePixelValue = int(lung_raw_range[1])

                        output_soft_tissue_dcm.PixelData = soft_tissue_new_pixel_data.tobytes()
                        output_lung_dcm.PixelData = lung_new_pixel_data.tobytes()

                        output_filename = os.path.basename(dcm_path)
                        shutil.copy(dcm_path, os.path.join(working_raw_folder, output_filename))
                        output_soft_tissue_dcm.save_as(os.path.join(working_soft_tissue_folder, output_filename))
                        output_lung_dcm.save_as(os.path.join(working_lung_folder, output_filename))

                    except Exception as e:
                        # 오류 메시지를 더 자세하게 출력합니다.
                        print(f"Could not process file {dcm_path}. Error: {e}")
                        import traceback
                        traceback.print_exc() # 전체 traceback을 확인하기 위해 추가
    
    print("\nGeneration complete.")
    

def integrate(args, soft_tissue_args, lung_args):
    """Lung 및 Soft-tissue CycleGAN 모델의 결과물을 통합"""

    def get_hu_array(dcm):
        """DICOM 객체에서 HU 배열을 추출하는 헬퍼 함수"""
        intercept = dcm.RescaleIntercept if 'RescaleIntercept' in dcm else 0
        slope = dcm.RescaleSlope if 'RescaleSlope' in dcm else 1
        hu_array = dcm.pixel_array.astype(np.float32) * slope + intercept
        return hu_array
    
    # 범위 설정
    mask_threshold_hu = lung_args.hu_max - 10
    raw_range = (min(soft_tissue_args.hu_min, lung_args.hu_min), max(soft_tissue_args.hu_max, lung_args.hu_max))
    
    # 각 데이터셋 폴더 순회
    for dataset_name in args.dataset_names:
        working_dir = os.path.join(args.working_dir_root, dataset_name)
        output_dir = os.path.join(args.output_dir_root, dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nIntegrating dataset: {dataset_name}")

        # 환자 디렉토리 검색
        patient_dirs = sorted([d for d in glob.glob(os.path.join(working_dir, '*')) if os.path.isdir(d)])
        
        # 환자 디렉토리 순회
        for patient_dir in tqdm(patient_dirs, desc=f"Patients in {dataset_name}"):
            patient_id = os.path.basename(patient_dir)
            raw_base_path = os.path.join(patient_dir, "raw")
            soft_tissue_base_path = os.path.join(patient_dir, "soft_tissue")
            lung_base_path = os.path.join(patient_dir, "lung")
            output_base_path = os.path.join(output_dir, patient_id)
            os.makedirs(output_base_path, exist_ok=True)

            raw_dcm_list = sorted(glob.glob(os.path.join(raw_base_path, "*.dcm")))
            if not raw_dcm_list:
                print(f"No raw DICOM files found for patient {patient_id}. Skipping integration.")
                continue
            soft_tissue_dcm_list = sorted(glob.glob(os.path.join(soft_tissue_base_path, "*.dcm")))
            if not soft_tissue_dcm_list:
                print(f"No soft tissue DICOM files found for patient {patient_id}. Skipping integration.")
                continue
            lung_dcm_list = sorted(glob.glob(os.path.join(lung_base_path, "*.dcm")))
            if not lung_dcm_list:
                print(f"No lung DICOM files found for patient {patient_id}. Skipping integration.")
                continue
            if len(raw_dcm_list) != len(soft_tissue_dcm_list) != len(lung_dcm_list):
                print(f"Warning: Mismatch in number of DICOM files for patient {patient_id}. Raw: {len(raw_dcm_list)}, Soft tissue: {len(soft_tissue_dcm_list)}, Lung: {len(lung_dcm_list)}. Proceeding with integration.")
                continue
            
            merged_volume = []

            for idx, (raw_dcm_path, soft_tissue_dcm_path, lung_dcm_path) in enumerate(zip(raw_dcm_list, soft_tissue_dcm_list, lung_dcm_list)):
                try:
                    if os.path.basename(raw_dcm_path) != os.path.basename(soft_tissue_dcm_path) != os.path.basename(lung_dcm_path):
                        print(f"Warning: Filename mismatch between soft tissue and lung DICOM files: {soft_tissue_dcm_path} vs {lung_dcm_path}. Skipping this pair.")
                        continue

                    # DICOM 파일 로드
                    raw_dcm = pydicom.dcmread(raw_dcm_path)
                    soft_tissue_dcm = pydicom.dcmread(soft_tissue_dcm_path)
                    lung_dcm = pydicom.dcmread(lung_dcm_path)
                    
                    # 이미지 로드
                    raw_pixel_array = raw_dcm.pixel_array
                    soft_tissue_pixel_array = soft_tissue_dcm.pixel_array
                    lung_pixel_array = lung_dcm.pixel_array
                    merged_pixel_array = lung_pixel_array.copy()
                    
                    # HU 배열 계산 및 마스크 생성
                    raw_hu_array = get_hu_array(raw_dcm)
                    raw_mask = np.logical_or(raw_hu_array > raw_range[1], raw_hu_array < raw_range[0])
                    lung_hu_array = get_hu_array(lung_dcm)
                    mask = lung_hu_array > mask_threshold_hu 
                    
                    # 마스크를 사용하여 픽셀 교체
                    merged_pixel_array[mask] = soft_tissue_pixel_array[mask]
                    merged_pixel_array[raw_mask] = raw_pixel_array[raw_mask]
                    
                    merged_volume.append(merged_pixel_array)
                    
                except Exception as e:
                    print(f"Error Occurred: error on processing file {soft_tissue_dcm_path}. Message: {e}")
                    traceback.print_exc()
                    exit()

            # 후처리 적용
            merged_volume = np.array(merged_volume)
            merged_volume = postprocess_ct_volume(merged_volume, method='gaussian3d',
                                                    sigma_z=0.75, sigma_xy=0.5,
                                                    enhance_sharpness=True, sharpen_amount=0.7, sharpen_radius=1.0)
            
            # 후처리된 슬라이스 저장
            for idx, soft_tissue_dcm_path in enumerate(soft_tissue_dcm_list):
                try:
                    soft_tissue_dcm = pydicom.dcmread(soft_tissue_dcm_path)
                    merged_pixel_array = merged_volume[idx]

                    # 최종 DICOM 객체 생성 및 저장
                    output_dcm = soft_tissue_dcm.copy()
                    output_dcm.PixelData = merged_pixel_array.tobytes()
                    
                    # VR 에러 해결 및 메타데이터 업데이트
                    if output_dcm.PixelRepresentation == 0:
                        vr = 'US'
                    else:
                        vr = 'SS'
                    output_dcm.add_new((0x0028, 0x0106), vr, int(merged_pixel_array.min()))
                    output_dcm.add_new((0x0028, 0x0107), vr, int(merged_pixel_array.max()))
                    
                    full_range_width = 250 - (-1000)
                    full_range_center = -1000 + (full_range_width / 2)
                    output_dcm.WindowWidth = full_range_width
                    output_dcm.WindowCenter = full_range_center
                    output_dcm.SeriesDescription = "sCECT (Full Range Integrated) v4"
                    
                    output_dcm_path = os.path.join(output_base_path, f"{idx:04d}.dcm")
                    output_dcm.save_as(output_dcm_path)
                    
                except Exception as e:
                    print(f"Error Occurred: error on processing file {soft_tissue_dcm_path}. Message: {e}")
                    traceback.print_exc()
                    exit()
                    
    print("\nIntegration complete.")


if __name__ == "__main__":
    print("Starting DUCOSY-GAN Inference and Integration Process")
    # 인자 설정
    args = get_common_infer_args() # 공통 추론 인자
    soft_tissue_args = get_soft_tissue_infer_args() # Soft-tissue CycleGAN 추론 인자
    lung_args = get_lung_infer_args() # Lung CycleGAN 추론 인자
    
    # 생성 실행
    generate(args, soft_tissue_args, lung_args)
    
    # 통합 실행
    integrate(args, soft_tissue_args, lung_args)

    print("\nAll processing complete!")
    print(f" - Final integrated DICOM files are saved in: {args.output_dir_root}")
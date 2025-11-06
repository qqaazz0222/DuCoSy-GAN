import os
import glob
import pydicom
import numpy as np
from tqdm import tqdm
import nibabel as nib
from multiprocessing import Pool
import torch
import signal
import sys
import atexit

from modules.argmanager import get_common_infer_args

import warnings
import subprocess
warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")


def cleanup_gpu_memory():
    """GPU 메모리 정리 함수"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("\n✓ GPU memory cleaned up")
    except Exception as e:
        print(f"\n⚠ Failed to clean up GPU memory: {e}")


def signal_handler(signum, frame):
    """시그널 핸들러 - 프로세스 강제 종료 시 GPU 메모리 정리"""
    signal_name = signal.Signals(signum).name
    print(f"\n\n{'='*80}")
    print(f"⚠ Received signal: {signal_name} ({signum})")
    print(f"Cleaning up resources and exiting...")
    print(f"{'='*80}")
    
    # GPU 메모리 정리
    cleanup_gpu_memory()
    
    # 프로세스 종료
    sys.exit(0)


def register_signal_handlers():
    """시그널 핸들러 등록"""
    # SIGINT (Ctrl+C), SIGTERM (kill 명령) 처리
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 정상 종료 시에도 GPU 메모리 정리
    atexit.register(cleanup_gpu_memory)
    
    print("✓ Signal handlers registered (SIGINT, SIGTERM)")


def raw_to_hu(raw_pixel_array, rescale_slope, rescale_intercept):
    """Raw Pixel Data를 HU 단위로 변환"""
    hu_array = raw_pixel_array.astype(np.int16) * rescale_slope + rescale_intercept
    return hu_array


def dicom_to_nifti(dicom_dir, output_nifti_path):
    """
    DICOM 디렉토리를 NIfTI 파일로 변환
    
    Args:
        dicom_dir: DICOM 파일들이 있는 디렉토리
        output_nifti_path: 출력 NIfTI 파일 경로
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # DICOM 파일 로드
        dcm_files = sorted(glob.glob(os.path.join(dicom_dir, "*.dcm")))
        
        if not dcm_files:
            print(f"  ✗ No DICOM files found in {dicom_dir}")
            return False
        
        # 첫 번째 DICOM 파일을 읽어서 메타데이터 확인
        first_dcm = pydicom.dcmread(dcm_files[0])
        
        # DICOM 슬라이스들을 ImagePositionPatient의 Z 좌표로 정렬
        dicom_slices = []
        for dcm_file in dcm_files:
            dcm = pydicom.dcmread(dcm_file)
            # ImagePositionPatient가 없는 경우 InstanceNumber 사용
            if hasattr(dcm, 'ImagePositionPatient'):
                z_pos = float(dcm.ImagePositionPatient[2])
            elif hasattr(dcm, 'SliceLocation'):
                z_pos = float(dcm.SliceLocation)
            elif hasattr(dcm, 'InstanceNumber'):
                z_pos = float(dcm.InstanceNumber)
            else:
                z_pos = 0.0
            
            dicom_slices.append((z_pos, dcm))
        
        # Z 좌표 기준으로 정렬
        dicom_slices.sort(key=lambda x: x[0])
        
        # 3D 볼륨 생성
        slices = []
        for _, dcm in dicom_slices:
            # Pixel Array 가져오기
            pixel_array = dcm.pixel_array
            
            # HU 값으로 변환
            slope = float(dcm.RescaleSlope) if hasattr(dcm, 'RescaleSlope') else 1.0
            intercept = float(dcm.RescaleIntercept) if hasattr(dcm, 'RescaleIntercept') else 0.0
            
            hu_array = pixel_array.astype(np.float32) * slope + intercept
            slices.append(hu_array)
        
        # NumPy 배열로 스택 (Z, Y, X)
        volume = np.stack(slices, axis=0).astype(np.float32)
        
        # 축 순서 변경: (Z, Y, X) -> (X, Y, Z)
        # 이렇게 해야 NIfTI에서 axial view로 올바르게 표시됨
        volume = np.transpose(volume, (2, 1, 0))
        
        # Spacing 정보 추출
        pixel_spacing = [1.0, 1.0]
        if hasattr(first_dcm, 'PixelSpacing'):
            pixel_spacing = [float(x) for x in first_dcm.PixelSpacing]
        
        slice_thickness = 1.0
        if hasattr(first_dcm, 'SliceThickness'):
            slice_thickness = float(first_dcm.SliceThickness)
        elif len(dicom_slices) > 1:
            # SliceThickness가 없으면 슬라이스 간격으로 계산
            slice_thickness = abs(dicom_slices[1][0] - dicom_slices[0][0])
        
        # 간단한 Affine 행렬 생성
        affine = np.eye(4)
        
        # 축이 변경되었으므로 spacing도 재배치
        # 원래: Z, Y, X -> 변경 후: X, Y, Z
        affine[0, 0] = -pixel_spacing[1]   # X spacing (row spacing)
        affine[1, 1] = -pixel_spacing[0]   # Y spacing (col spacing)
        affine[2, 2] = slice_thickness      # Z spacing (slice thickness)
        
        # Origin 설정
        if hasattr(first_dcm, 'ImagePositionPatient'):
            position = np.array([float(x) for x in first_dcm.ImagePositionPatient])
            affine[0, 3] = -position[0]
            affine[1, 3] = -position[1]
            affine[2, 3] = position[2]
        
        # NIfTI 이미지 생성 및 저장
        nifti_img = nib.Nifti1Image(volume, affine)
        
        # Header 정보 설정
        nifti_img.header.set_xyzt_units('mm', 'sec')
        
        # 파일 저장
        nib.save(nifti_img, output_nifti_path)
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error converting DICOM to NIfTI: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_single_patient(patient_info):
    """
    단일 환자 처리 함수 (병렬 처리용)
    
    Args:
        patient_info: (patient_dir, masked_patient_dir, working_patient_dir) 튜플
    
    Returns:
        (patient_id, success, error_message) 튜플
    """
    patient_dir, masked_patient_dir, working_patient_dir = patient_info
    patient_id = os.path.basename(os.path.dirname(patient_dir))  # ncct_folder의 부모 디렉토리 이름
    
    try:
        # DICOM 파일 확인
        dcm_files = glob.glob(os.path.join(patient_dir, "*.dcm"))
        
        if not dcm_files:
            return (patient_id, False, "No DICOM files found")
        
        # working 디렉토리 생성
        os.makedirs(working_patient_dir, exist_ok=True)
        
        # 1. DICOM을 NIfTI로 변환 (working 디렉토리에 저장)
        nifti_path = os.path.join(working_patient_dir, "input.nii.gz")
        
        if not dicom_to_nifti(patient_dir, nifti_path):
            return (patient_id, False, "Failed to convert DICOM to NIfTI")
        
        # 2. TotalSegmentator 실행
        try:
            cmd = [
                "TotalSegmentator", 
                "-i", nifti_path,
                "-o", masked_patient_dir,
                "--device", "gpu",
                "--ml",  # multi-label output
            ]
            
            result = subprocess.run(
                cmd, 
                check=True, 
                capture_output=True, 
                text=True,
                timeout=600  # 10분 타임아웃
            )
            
            return (patient_id, True, None)
            
        except subprocess.CalledProcessError as e:
            error_msg = f"TotalSegmentator error (code {e.returncode})"
            if e.stderr:
                error_msg += f": {e.stderr[-200:]}"
            return (patient_id, False, error_msg)
        except subprocess.TimeoutExpired:
            return (patient_id, False, "TotalSegmentator timeout")
        except FileNotFoundError:
            return (patient_id, False, "TotalSegmentator command not found")
        except Exception as e:
            return (patient_id, False, f"Unexpected error: {str(e)}")
    
    except Exception as e:
        import traceback

        error_msg = f"Exception: {str(e)}\n{traceback.format_exc()}"
        return (patient_id, False, error_msg)

def generate(args):
    processed_dir = os.path.join(args.input_dir_root)
    mask_dir = os.path.join(args.output_dir_root, "mask")
    working_dir = os.path.join(args.working_dir_root)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(working_dir, exist_ok=True)
    
    # 배치 크기 결정
    batch_size = args.batch_size if hasattr(args, 'batch_size') and args.batch_size > 0 else 1
    
    print(f"\nMasking Configuration:")
    print(f"  - Batch size (parallel processes): {batch_size}")
    print(f"  - Working directory: {working_dir}")
    print(f"  - Output directory: {mask_dir}")
    
    dataset_names = args.dataset_names
    for dataset in dataset_names:
        masked_dataset_dir = os.path.join(mask_dir, dataset)
        working_dataset_dir = os.path.join(working_dir, dataset)
        os.makedirs(masked_dataset_dir, exist_ok=True)
        os.makedirs(working_dataset_dir, exist_ok=True)
        
        dataset_dir = os.path.join(processed_dir, dataset)
        patient_dirs = sorted([d for d in glob.glob(os.path.join(dataset_dir, "*")) if os.path.isdir(d)])
        
        if not patient_dirs:
            print(f"\n  ✗ No patient directories found in {dataset_dir}")
            continue
        
        print(f"\nProcessing dataset: {dataset}")
        print(f"  - Total patients: {len(patient_dirs)}")
        print(f"  - Parallel processes: {batch_size}")
        
        # 환자 정보 준비 (patient_dir, masked_patient_dir, working_patient_dir)
        patient_info_list = [
            (
                os.path.join(patient_dir, args.ncct_folder), 
                os.path.join(masked_dataset_dir, os.path.basename(patient_dir)),
                os.path.join(working_dataset_dir, os.path.basename(patient_dir))
            )
            for patient_dir in patient_dirs
        ]
        
        # 병렬 처리
        success_count = 0
        error_count = 0
        
        if batch_size == 1:
            # 단일 프로세스로 처리 (진행 상황 표시)
            for patient_info in tqdm(patient_info_list, desc=f"Masking {dataset}", unit="patient"):
                patient_id, success, error_msg = process_single_patient(patient_info)
                if success:
                    success_count += 1
                else:
                    error_count += 1
                    print(f"\n  ✗ Failed: {patient_id} - {error_msg}")
        else:
            # 멀티프로세싱으로 처리
            with Pool(processes=batch_size) as pool:
                # imap_unordered를 사용하여 진행 상황 표시
                results = list(tqdm(
                    pool.imap_unordered(process_single_patient, patient_info_list),
                    total=len(patient_info_list),
                    desc=f"Masking {dataset}",
                    unit="patient"
                ))
                
                for patient_id, success, error_msg in results:
                    if success:
                        success_count += 1
                    else:
                        error_count += 1
                        print(f"\n  ✗ Failed: {patient_id} - {error_msg}")
        
        print(f"\nDataset {dataset} completed:")
        print(f"  - Successful: {success_count}")
        print(f"  - Failed: {error_count}")
    
    print("Generate mask process completed!")
    print(f"Masks saved in: {mask_dir}")
    
    
def masking(args):
    processed_dir = os.path.join(args.output_dir_root)
    mask_dir = os.path.join(args.output_dir_root, "mask")
    masked_dir = os.path.join(args.output_dir_root, "masked")
    os.makedirs(masked_dir, exist_ok=True)
    
    mask_target_label = [51, 68] 
    # https://github.com/wasserth/TotalSegmentator?tab=readme-ov-file#class-details
    
    dataset_names = args.dataset_names
    for dataset in dataset_names:
        dataset_dir = os.path.join(processed_dir, dataset)
        mask_dataset_dir = os.path.join(mask_dir, dataset)
        masked_dataset_dir = os.path.join(masked_dir, dataset)
        os.makedirs(masked_dataset_dir, exist_ok=True)
        
        patient_dirs = sorted([d for d in glob.glob(os.path.join(dataset_dir, "*")) if os.path.isdir(d)])
        if not patient_dirs:
            print(f"\n  ✗ No patient directories found in {dataset_dir}")
            continue
        
        print(f"\nApplying masks for dataset: {dataset}")
        for patient_dir in tqdm(patient_dirs, desc=f"Applying masks to {dataset}", unit="patient"):
            patient_id = os.path.basename(patient_dir)
            mask_file_path = os.path.join(mask_dataset_dir, patient_id + '.nii')
            masked_patient_dir = os.path.join(masked_dataset_dir, patient_id)
            os.makedirs(masked_patient_dir, exist_ok=True)
            
            slice_files = sorted(glob.glob(os.path.join(patient_dir, "*.dcm")))
            if not slice_files:
                print(f"  ✗ No DICOM files found for patient {patient_id}, skipping.")
                continue
            slice_files.sort(key=lambda x: int(pydicom.dcmread(x).InstanceNumber))
            mask_volume = nib.load(mask_file_path).get_fdata()
            mask_volume = np.transpose(mask_volume, (1, 0, 2))
            
            if len(slice_files) != mask_volume.shape[2]:
                print(f"  ✗ Slice count mismatch for patient {patient_id}, skipping.")
                continue
            
            for idx, slice_file in enumerate(slice_files):
                dcm = pydicom.dcmread(slice_file)
                pixel_array = dcm.pixel_array
                
                # 마스크 적용 (심장 영역을 0으로 설정)
                heart_mask = np.where((mask_volume[:, :, idx] >= mask_target_label[0]) & (mask_volume[:, :, idx] <= mask_target_label[1]), 1, 0)
                pixel_array[heart_mask == 1] = 0
                
                # Pixel Data 업데이트
                dcm.PixelData = pixel_array.astype(np.int16).tobytes()

                # 익명화된 파일로 저장
                output_path = os.path.join(masked_patient_dir, os.path.basename(slice_file))
                dcm.save_as(output_path)
                
    print("Masking process completed!")
    print(f"Masked DICOMs saved in: {masked_dir}")

            
if __name__ == "__main__":
    print("Starting DUCOSY-GAN Anonymization Process")
    
    # 시그널 핸들러 등록
    register_signal_handlers()
    
    # 공통 추론 인자
    args = get_common_infer_args()
    
    # CUDA 캐시 정리
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"✓ CUDA cache cleared (Available GPUs: {torch.cuda.device_count()})")
    except ImportError:
        print("⚠ PyTorch not available, skipping CUDA cache clearing")
    except Exception as e:
        print(f"⚠ Failed to clear CUDA cache: {e}")
    
    try:
        # 마스크 생성
        generate(args)
        
        # 마스크 적용
        masking(args)
        
        print("\n" + "="*80)
        print("✓ All processes completed successfully!")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("⚠ Process interrupted by user (Ctrl+C)")
        print("="*80)
        cleanup_gpu_memory()
        sys.exit(1)
        
    except Exception as e:
        print("\n\n" + "="*80)
        print(f"✗ Error occurred: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        cleanup_gpu_memory()
        sys.exit(1)
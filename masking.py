import os
import csv
import glob
import uuid
import shutil
import pydicom
import numpy as np
from tqdm import tqdm
import nibabel as nib
import tempfile

from modules.argmanager import get_common_infer_args

import warnings
import subprocess
warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")


def raw_to_hu(raw_pixel_array, rescale_slope, rescale_intercept):
    """Raw Pixel Data를 HU 단위로 변환"""
    hu_array = raw_pixel_array.astype(np.float32) * rescale_slope + rescale_intercept
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
        # DICOM 파일 로드 및 정렬
        dcm_files = sorted(glob.glob(os.path.join(dicom_dir, "*.dcm")))
        
        if not dcm_files:
            print(f"  ✗ No DICOM files found in {dicom_dir}")
            return False
        
        # DICOM 데이터를 Z 위치 기준으로 정렬
        dicom_data = []
        for dcm_path in dcm_files:
            dcm = pydicom.dcmread(dcm_path)
            z_position = dcm.get('ImagePositionPatient', [0.0, 0.0, 0.0])[2]
            dicom_data.append((dcm, z_position))
        
        dicom_data.sort(key=lambda x: x[1])
        
        # HU 값으로 변환하여 3D 볼륨 생성
        volume = []
        for dcm, _ in dicom_data:
            hu_array = raw_to_hu(dcm.pixel_array, dcm.RescaleSlope, dcm.RescaleIntercept)
            volume.append(hu_array)
        
        volume = np.stack(volume, axis=0)
        
        # DICOM 메타데이터에서 spacing 정보 추출
        first_dcm = dicom_data[0][0]
        pixel_spacing = first_dcm.get('PixelSpacing', [1.0, 1.0])
        slice_thickness = first_dcm.get('SliceThickness', 1.0)
        
        # Affine 행렬 생성 (spacing 정보 포함)
        affine = np.eye(4)
        affine[0, 0] = pixel_spacing[1]  # x spacing
        affine[1, 1] = pixel_spacing[0]  # y spacing
        affine[2, 2] = slice_thickness    # z spacing
        
        # NIfTI 이미지 생성 및 저장
        nifti_img = nib.Nifti1Image(volume, affine)
        nib.save(nifti_img, output_nifti_path)
        return True
        
    except Exception as e:
        print(f"  ✗ Error converting DICOM to NIfTI: {e}")
        import traceback
        traceback.print_exc()
        return False

def masking(args):
    processed_dir = os.path.join(args.output_dir_root)
    masked_dir = os.path.join(args.output_dir_root, "masked")
    os.makedirs(masked_dir, exist_ok=True)
    
    dataset_names = args.dataset_names
    for dataset in dataset_names:
        masked_dataset_dir = os.path.join(masked_dir, dataset)
        os.makedirs(masked_dataset_dir, exist_ok=True)
        
        dataset_dir = os.path.join(processed_dir, dataset)
        patient_dirs = sorted([d for d in glob.glob(os.path.join(dataset_dir, "*")) if os.path.isdir(d)])
        
        for patient_dir in tqdm(patient_dirs, desc=f"Masking dataset: {dataset}", unit="patient"):
            patient_id = os.path.basename(patient_dir)
            masked_patient_dir = os.path.join(masked_dataset_dir, patient_id)
            
            # DICOM 파일 확인
            dcm_files = glob.glob(os.path.join(patient_dir, "*.dcm"))
            
            if not dcm_files:
                print(f"\n  ✗ No DICOM files found in {patient_dir}, skipping...")
                continue
            
            # 임시 디렉토리 생성
            with tempfile.TemporaryDirectory() as temp_dir:
                # 1. DICOM을 NIfTI로 변환
                temp_nifti = os.path.join(temp_dir, "input.nii.gz")
                
                if not dicom_to_nifti(patient_dir, temp_nifti):
                    print(f"  ✗ Failed to convert DICOM to NIfTI for {patient_id}")
                    continue
                
                # 2. TotalSegmentator 실행
                try:
                    cmd = [
                        "TotalSegmentator", 
                        "-i", temp_nifti,
                        "-o", masked_patient_dir,
                        "--device", "gpu",
                        "--task", "total",
                        "--ml",  # multi-label output
                    ]
                    
                    result = subprocess.run(
                        cmd, 
                        check=True, 
                        capture_output=True, 
                        text=True,
                        timeout=600  # 10분 타임아웃
                    )
                    
                except subprocess.CalledProcessError as e:
                    print(f"  ✗ Error during TotalSegmentator execution for patient {patient_id}")
                    print(f"    Return code: {e.returncode}")
                    if e.stdout:
                        print(f"    STDOUT: {e.stdout[-500:]}")  # 마지막 500자만 출력
                    if e.stderr:
                        print(f"    STDERR: {e.stderr[-500:]}")
                    continue
                except subprocess.TimeoutExpired:
                    print(f"  ✗ TotalSegmentator timed out for patient {patient_id}")
                    continue
                except FileNotFoundError:
                    print(f"  ✗ TotalSegmentator command not found. Please ensure it's installed and in PATH.")
                    break
                except Exception as e:
                    print(f"  ✗ Unexpected error for patient {patient_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    print("\n" + "="*80)
    print("Masking process completed!")
    print(f"Results saved in: {masked_dir}")
    print("="*80)

            
if __name__ == "__main__":
    print("Starting DUCOSY-GAN Anonymization Process")
    # 공통 추론 인자
    args = get_common_infer_args()
    
    # 마스킹 실행
    masking(args)
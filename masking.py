import os
import glob
import pydicom
import numpy as np
from tqdm import tqdm
import nibabel as nib
from multiprocessing import Pool
import cv2
import torch
import signal
import sys
import atexit
import psutil
import time
from totalsegmentator.python_api import totalsegmentator
from scipy.spatial import ConvexHull, QhullError

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
            print("\nGPU memory cleaned up")
    except Exception as e:
        print(f"\nFailed to clean up GPU memory: {e}")


def kill_process_tree(pid):
    """프로세스와 모든 자식 프로세스를 종료"""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        
        # 자식 프로세스들 먼저 종료
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        
        # 부모 프로세스 종료
        try:
            parent.terminate()
        except psutil.NoSuchProcess:
            pass
        
        # 종료 대기 (최대 3초)
        gone, alive = psutil.wait_procs(children + [parent], timeout=3)
        
        # 강제 종료 (SIGKILL)
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
                
    except psutil.NoSuchProcess:
        pass  # 이미 종료된 프로세스
    except Exception as e:
        print(f"  Error killing process {pid}: {e}")


def signal_handler(signum, frame):
    """시그널 핸들러 - 프로세스 강제 종료 시 GPU 메모리 정리"""
    signal_name = signal.Signals(signum).name
    print(f"\n\n{'='*80}")
    print(f"Received signal: {signal_name} ({signum})")
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
    
    print(" - Signal handlers registered (SIGINT, SIGTERM)")


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
            print(f"No DICOM files found in {dicom_dir}")
            return False
        
        # 첫 번째 DICOM 파일을 읽어서 메타데이터 확인
        first_dcm = pydicom.dcmread(dcm_files[0])
        
        # DICOM 슬라이스들을 ImagePositionPatient의 Z 좌표로 정렬
        temp = []
        dicom_slices = []
        for dcm_file in dcm_files:
            dcm = pydicom.dcmread(dcm_file)
            if hasattr(dcm, 'InstanceNumber'):
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
        print(f"Error converting DICOM to NIfTI: {e}")
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
        process = None
        if os.path.exists(f"{masked_patient_dir}.nii"):
            return (patient_id, True, None)
        try:
            cmd = [
                "TotalSegmentator", 
                "-i", nifti_path,
                "-o", masked_patient_dir,
                "--device", "gpu",
                "--ml",  # multi-label output
            ]
            
            # 프로세스 시작 및 PID 저장
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            pid = process.pid
            
            # 프로세스 완료 대기 (타임아웃 포함)
            try:
                stdout, stderr = process.communicate(timeout=1200)
                return_code = process.returncode
                
                # 프로세스와 자식 프로세스 명시적 종료
                kill_process_tree(pid)
                
                # GPU 메모리 정리
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 성공 여부 확인
                if return_code != 0:
                    error_msg = f"TotalSegmentator error (code {return_code})"
                    if stderr:
                        error_msg += f": {stderr[-200:]}"
                    return (patient_id, False, error_msg)
                
                return (patient_id, True, None)
                
            except subprocess.TimeoutExpired:
                # 타임아웃 시 프로세스 강제 종료
                kill_process_tree(pid)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return (patient_id, False, "TotalSegmentator timeout")
                
        except FileNotFoundError:
            return (patient_id, False, "TotalSegmentator command not found")
        except Exception as e:
            # 예외 발생 시에도 프로세스 정리
            if process and process.pid:
                kill_process_tree(process.pid)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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
    
    print(f"Masking Configuration:")
    print(f" - Batch size (parallel processes): {batch_size}")
    print(f" - Working directory: {working_dir}")
    print(f" - Output directory: {mask_dir}")
    
    dataset_names = args.dataset_names
    for dataset in dataset_names:
        masked_dataset_dir = os.path.join(mask_dir, dataset)
        working_dataset_dir = os.path.join(working_dir, dataset)
        os.makedirs(masked_dataset_dir, exist_ok=True)
        os.makedirs(working_dataset_dir, exist_ok=True)
        
        dataset_dir = os.path.join(processed_dir, dataset)
        patient_dirs = sorted([d for d in glob.glob(os.path.join(dataset_dir, "*")) if os.path.isdir(d)])
        
        if not patient_dirs:
            print(f"\nNo patient directories found in {dataset_dir}")
            continue
        
        print(f"\nProcessing dataset: {dataset}")
        print(f" - Total patients: {len(patient_dirs)}")
        print(f" - Parallel processes: {batch_size}")
        
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
                    print(f"✗ Failed: {patient_id} - {error_msg}")
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
                        print(f"\nFailed: {patient_id} - {error_msg}")
        
        print(f"Dataset {dataset} completed:")
        print(f" - Successful: {success_count}")
        print(f" - Failed: {error_count}")
    
    print("Generate mask process completed!")
    print(f"Masks saved in: {mask_dir}")
    
    
def masking(args):
    cect_dir = os.path.join(args.input_dir_root)
    scect_dir = os.path.join(args.output_dir_root)
    mask_dir = os.path.join(args.output_dir_root, "modified_mask")
    masked_dir = os.path.join(args.output_dir_root, "masked")
    os.makedirs(masked_dir, exist_ok=True)
    
    mask_target_label = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 18, 19, 20, 21, 22, 23, 24, 51, 52, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68]
    # mask_target_group = [51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64]
    # https://github.com/wasserth/TotalSegmentator?tab=readme-ov-file#class-details
    
    dataset_names = args.dataset_names
    for dataset in dataset_names:
        original_dataset_dir = os.path.join(cect_dir, dataset)
        generated_dataset_dir = os.path.join(scect_dir, dataset)
        mask_dataset_dir = os.path.join(mask_dir, dataset)
        masked_dataset_dir = os.path.join(masked_dir, dataset)
        os.makedirs(masked_dataset_dir, exist_ok=True)
        
        ncct_patient_dirs = sorted([os.path.join(d, args.ncct_folder) for d in glob.glob(os.path.join(original_dataset_dir, "*")) if os.path.isdir(d)])
        cect_patient_dirs = sorted([os.path.join(d, args.cect_folder) for d in glob.glob(os.path.join(original_dataset_dir, "*")) if os.path.isdir(d)])
        scect_patient_dirs = sorted([d for d in glob.glob(os.path.join(generated_dataset_dir, "*")) if os.path.isdir(d)])
        if not len(ncct_patient_dirs) == len(cect_patient_dirs) == len(scect_patient_dirs):
            print(f"Mismatch in number of patient directories among NCCT, CECT, and SCECT for dataset {dataset}. Skipping.")
            continue
        
        patient_dirs = zip(ncct_patient_dirs, cect_patient_dirs, scect_patient_dirs)
        if not patient_dirs:
            print(f"No patient directories found in {ncct_patient_dirs} or {cect_patient_dirs} or {scect_patient_dirs}")
            continue
        
        print(f"\nApplying masks for dataset: {dataset}")
        for ncct_patient_dir, cect_patient_dir, scect_patient_dir in tqdm(patient_dirs, desc=f"Applying masks", unit="patient", total=len(cect_patient_dirs)):
            patient_id = os.path.basename(scect_patient_dir)
            if patient_id not in cect_patient_dir:
                print(f"Patient ID mismatch between CECT and SCECT directories: {cect_patient_dir}, {scect_patient_dir}. Skipping.")
                continue
            mask_file_path = os.path.join(mask_dataset_dir, patient_id + '.nii')
            masked_patient_dir = os.path.join(masked_dataset_dir, patient_id)
            masked_dataset_ncct_dir = os.path.join(masked_patient_dir, args.ncct_folder)
            masked_patient_cect_dir = os.path.join(masked_patient_dir, args.cect_folder)
            masked_patient_scect_dir = os.path.join(masked_patient_dir, 'generated')
            os.makedirs(masked_patient_dir, exist_ok=True)
            os.makedirs(masked_dataset_ncct_dir, exist_ok=True)
            os.makedirs(masked_patient_cect_dir, exist_ok=True)
            os.makedirs(masked_patient_scect_dir, exist_ok=True)

            ncct_slice_files = sorted(glob.glob(os.path.join(ncct_patient_dir, "*.dcm")))
            cect_slice_files = sorted(glob.glob(os.path.join(cect_patient_dir, "*.dcm")))
            scect_slice_files = glob.glob(os.path.join(scect_patient_dir, "*.dcm"))
            if not ncct_slice_files:
                print(f"No NCCT DICOM files found for patient {patient_id}, skipping.")
                continue
            if not cect_slice_files:
                print(f"No CECT DICOM files found for patient {patient_id}, skipping.")
                continue
            if not scect_slice_files:
                print(f"No SCECT DICOM files found for patient {patient_id}, skipping.")
                continue
            ncct_slice_files.sort(key=lambda x: int(pydicom.dcmread(x).InstanceNumber))
            cect_slice_files.sort(key=lambda x: int(pydicom.dcmread(x).InstanceNumber))
            scect_slice_files.sort(key=lambda x: int(pydicom.dcmread(x).InstanceNumber))
            
            if not os.path.exists(mask_file_path):
                print(f"Mask file not found for patient {patient_id}, skipping masking.")
                continue
            mask_volume = nib.load(mask_file_path).get_fdata()
            mask_volume = np.transpose(mask_volume, (2, 1, 0))
            
            heart_mask_volume = np.zeros_like(mask_volume, dtype=np.uint8)
            
            
            for z in range(mask_volume.shape[0]):
                slice_mask = mask_volume[z, :, :]
                filtered_slice_mask = np.zeros_like(slice_mask)
                for label in mask_target_label:
                    label_mask = (slice_mask == label).astype(np.uint8)
                    
                    contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(filtered_slice_mask, contours, -1, (label), thickness=-1)
                    cv2.drawContours(filtered_slice_mask, contours, -1, (label), thickness=2)
                    filtered_slice_mask[label_mask == 1] = label
                #     # ConvexHull을 위한 포인트 추출
                #     points = np.column_stack(np.where(label_mask == 1))
                #     if len(points) >= 3:  # ConvexHull은 최소 3개의 점이 필요
                #         try:
                #             hull = ConvexHull(points)
                #             # Hull 포인트를 contour 형식으로 변환 (y, x -> x, y)
                #             label_mask_convex = points[hull.vertices][:, ::-1].astype(np.int32)
                #         except QhullError:
                #             label_mask_convex = points[:, ::-1].astype(np.int32)
                #     else:
                #         # 포인트가 부족하면 원본 포인트 사용
                #         label_mask_convex = points[:, ::-1].astype(np.int32) if len(points) > 0 else np.array([])
                    
                #     # fillPoly는 유효한 포인트가 있을 때만 호출
                #     if len(label_mask_convex) > 0:
                #         cv2.fillPoly(filtered_slice_mask, [label_mask_convex], label)
                        
                # group_mask = np.isin(filtered_slice_mask, mask_target_group).astype(np.uint8)
                # group_points = np.column_stack(np.where(group_mask == 1))
                # if len(group_points) >= 3:
                #     try:
                #         hull = ConvexHull(group_points)
                #         group_mask_convex = group_points[hull.vertices][:, ::-1].astype(np.int32)
                #     except QhullError:
                #         group_mask_convex = group_points[:, ::-1].astype(np.int32)
                # else:
                #     group_mask_convex = group_points[:, ::-1].astype(np.int32) if len(group_points) > 0 else np.array([])
                
                # if len(group_mask_convex) > 0:
                #     cv2.fillPoly(filtered_slice_mask, [group_mask_convex], 1)
                    
                filtered_slice_mask = (filtered_slice_mask > 0).astype(np.uint8)
                                    
                # from PIL import Image
                # # Debug: 마스크된 슬라이스 저장
                # debug_img = Image.fromarray((filtered_slice_mask * 255).astype(np.uint8))
                # debug_img.save(f"debug/slice_{z}.png")    
                
                heart_mask_volume[z, :, :] = filtered_slice_mask
            
            heart_mask_volume = heart_mask_volume.astype(np.uint8)
            
            for z in range(heart_mask_volume.shape[0]):
                slice_mask = heart_mask_volume[z, :, :]
                cur_slice_mask = slice_mask.copy().astype(np.uint8)
                contours, _ = cv2.findContours(cur_slice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(cur_slice_mask, contours, -1, (1), thickness=4)
                heart_mask_volume[z, :, :] = cur_slice_mask

            if len(cect_slice_files) != mask_volume.shape[0] and len(scect_slice_files) != mask_volume.shape[0]:
                print(f"Slice count mismatch for patient {patient_id}, skipping.")
                continue

            for idx, (ncct_slice_file, cect_slice_file, scect_slice_file) in enumerate(zip(ncct_slice_files, cect_slice_files, scect_slice_files)):
                ncct_dcm = pydicom.dcmread(ncct_slice_file)
                ncct_pixel_array = ncct_dcm.pixel_array.copy()
                cect_dcm = pydicom.dcmread(cect_slice_file)
                cect_pixel_array = cect_dcm.pixel_array.copy()
                scect_dcm = pydicom.dcmread(scect_slice_file)
                scect_pixel_array = scect_dcm.pixel_array.copy()

                # 마스크 적용
                heart_mask = heart_mask_volume[idx, :, :]
                ncct_pixel_array[heart_mask != 0] = 9999
                cect_pixel_array[heart_mask != 0] = 9999
                scect_pixel_array[heart_mask != 0] = 9999
                
                # 압축된 Transfer Syntax를 비압축으로 변경
                # Explicit VR Little Endian (비압축)
                ncct_dcm.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
                cect_dcm.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
                scect_dcm.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
                
                # Pixel Data 업데이트
                ncct_dcm.PixelData = ncct_pixel_array.astype(np.int16).tobytes()
                cect_dcm.PixelData = cect_pixel_array.astype(np.int16).tobytes()
                scect_dcm.PixelData = scect_pixel_array.astype(np.int16).tobytes()
                
                # 압축 관련 속성 제거 (있을 경우)
                if hasattr(ncct_dcm, 'PhotometricInterpretation'):
                    if ncct_dcm.PhotometricInterpretation in ['YBR_FULL_422', 'YBR_FULL']:
                        ncct_dcm.PhotometricInterpretation = 'MONOCHROME2'
                if hasattr(cect_dcm, 'PhotometricInterpretation'):
                    if cect_dcm.PhotometricInterpretation in ['YBR_FULL_422', 'YBR_FULL']:
                        cect_dcm.PhotometricInterpretation = 'MONOCHROME2'
                if hasattr(scect_dcm, 'PhotometricInterpretation'):
                    if scect_dcm.PhotometricInterpretation in ['YBR_FULL_422', 'YBR_FULL']:
                        scect_dcm.PhotometricInterpretation = 'MONOCHROME2'

                # 익명화된 파일로 저장
                ncct_output_path = os.path.join(masked_dataset_ncct_dir, os.path.basename(ncct_slice_file))
                cect_output_path = os.path.join(masked_patient_cect_dir, os.path.basename(cect_slice_file))
                scect_output_path = os.path.join(masked_patient_scect_dir, os.path.basename(scect_slice_file))
                ncct_dcm.save_as(ncct_output_path, write_like_original=False)
                cect_dcm.save_as(cect_output_path, write_like_original=False)
                scect_dcm.save_as(scect_output_path, write_like_original=False)


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
            print(f" - CUDA cache cleared (Available GPUs: {torch.cuda.device_count()})")
    except ImportError:
        print("PyTorch not available, skipping CUDA cache clearing")
    except Exception as e:
        print(f"Failed to clear CUDA cache: {e}")
    
    try:
        # 마스크 생성
        # generate(args)
        
        # 마스크 적용
        masking(args)
        
        print("All processes completed successfully!")
        
    except KeyboardInterrupt:
        print("Process interrupted by user (Ctrl+C)")
        cleanup_gpu_memory()
        sys.exit(1)
        
    except Exception as e:
        print(f"✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        cleanup_gpu_memory()
        sys.exit(1)
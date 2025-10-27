import os
import csv
import glob
import uuid
import shutil
import pydicom
import numpy as np
from tqdm import tqdm

from modules.argmanager import get_common_infer_args

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")


def init_mapping(mapping_path):
    """익명화 매핑 파일 관리"""
    if os.path.exists(mapping_path):
        print(f"Anonymization mapping file already exists. Removing it to avoid overwriting.")
        os.remove(mapping_path)
    else:
        with open(mapping_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Category", "Site", "OriginalPatientID", "AnonymizedPatientID"])
    
    
def update_mapping(mapping_path, category, site, original_id, anonymized_id):
    """익명화 매핑 파일 업데이트"""
    with open(mapping_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([category, site, original_id, anonymized_id])
        

def anonymize(args, mapping_path):
    """DICOM 파일 익명화"""
    
    # 출력 디렉토리 생성
    original_dir = os.path.join(args.input_dir_root)
    generated_dir = os.path.join(args.output_dir_root)
    output_dir = os.path.join(args.output_dir_root, "anonymized")
    output_pixel_dir = os.path.join(args.output_dir_root, "anonymized_pixel")
    
    # 기존에 존재하는 경우 삭제 후 재생성
    if os.path.exists(output_dir):
        print(f"Anonymized output directory already exists. Removing it to avoid overwriting.")
        shutil.rmtree(output_dir)
    if os.path.exists(output_pixel_dir):
        print(f"Anonymized pixel output directory already exists. Removing it to avoid overwriting.")
        shutil.rmtree(output_pixel_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_pixel_dir, exist_ok=True)

    for category, category_dir in [("original", original_dir), ("generated", generated_dir)]:
        # 각 데이터셋 폴더 순회
        for dataset_name in args.dataset_names:
            data_dir = os.path.join(category_dir, dataset_name)
            
            # 환자 디렉토리 검색
            patient_dirs = sorted([d for d in glob.glob(os.path.join(data_dir, '*')) if os.path.isdir(d)])
            
            # 환자 디렉토리 순회
            for patient_dir in tqdm(patient_dirs, desc=f"Patients in {dataset_name}"):
                patient_id = os.path.basename(patient_dir)
                if category == "original":
                    patient_dir = os.path.join(patient_dir, args.ncct_folder)
                anonymized_id = str(uuid.uuid4().hex)  # UUID로 익명화된 ID 생성
                dcm_list = sorted(glob.glob(os.path.join(patient_dir, '*.dcm')))
                
                # 익명화 매핑 파일 업데이트
                update_mapping(mapping_path, category, dataset_name, patient_id, anonymized_id)
                
                # Pixel Data 초기화
                pixel_data_list = []
                
                for idx, dcm_path in enumerate(dcm_list):
                    try:
                        dcm = pydicom.dcmread(dcm_path)
                        
                        # 익명화: 환자 ID, 이름, 생년월일 등 민감 정보 제거
                        dcm.PatientID = anonymized_id
                        dcm.PatientName = "Anonymized"
                        dcm.PatientSex = "N"
                        dcm.PatientAge = ""
                        dcm.PatientBirthDate = ""
                        dcm.InstitutionName = "Anonymized"
                        dcm.InstitutionAddress = ""
                        dcm.ReferringPhysicianName = "Anonymized"
                        dcm.ImageType = ["PRIMARY", "AXIAL"]
                        dcm.StudyID = "1"
                        dcm.StudyDate = "19000101"
                        dcm.StudyTime = "000000"
                        dcm.StudyDescription = "-"
                        dcm.SeriesNumber = "1"
                        dcm.SeriesDescription = "-"
                        z_position = dcm.get('ImagePositionPatient', [0.0, 0.0, 0.0])[2]
                        pixel_data_list.append([dcm.pixel_array, z_position])
                        
                        # 익명화된 DICOM 파일 저장
                        output_patient_dir = os.path.join(output_dir, anonymized_id)
                        os.makedirs(output_patient_dir, exist_ok=True)
                        output_dcm_path = os.path.join(output_patient_dir, f"{idx:04d}.dcm")
                        # dcm.save_as(output_dcm_path)
                        
                    except Exception as e:
                        print(f"Could not process file {dcm_path}. Error: {e}")
                        import traceback
                        traceback.print_exc() # 전체 traceback을 확인하기 위해 추가
                        
                # Pixel Data 저장
                if pixel_data_list:
                    # Z 위치 기준으로 정렬
                    pixel_data_list.sort(key=lambda x: x[1])
                    # 정렬된 픽셀 데이터만 추출
                    pixel_data_list = [item[0] for item in pixel_data_list]
                    # 3D numpy 배열로 변환 후 저장
                    pixel_data_array = np.stack(pixel_data_list, axis=0)
                    npy_output_path = os.path.join(output_pixel_dir, f"{anonymized_id}.npy")
                    np.save(npy_output_path, pixel_data_array)
            
            
if __name__ == "__main__":
    print("Starting DUCOSY-GAN Anonymization Process")
    # 공통 추론 인자
    args = get_common_infer_args()
    
    # 익명화 매핑 파일 초기화
    mapping_path = os.path.join(args.output_dir_root, "anonymization_mapping.csv")
    init_mapping(mapping_path)

    # 익명화 실행
    anonymize(args, mapping_path)
    
    print("\nAnonymization complete.")
    print(f" - Anonymized DICOM files are saved in: {os.path.join(args.output_dir_root, 'anonymized')}")
    print(f" - Anonymized Pixel Data files are saved in: {os.path.join(args.output_dir_root, 'anonymized_pixel')}")
    print(f" - Anonymization mapping file is saved at: {mapping_path}")
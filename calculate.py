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

def convert(args):
    """DICOM 파일 압축"""
    # 출력 디렉토리 생성
    original_dir = os.path.join(args.input_dir_root)
    generated_dir = os.path.join(args.output_dir_root)
    output_dir = os.path.join(args.output_dir_root, "calculated")
    calculated_data_dir = os.path.join(output_dir, "data")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(calculated_data_dir, exist_ok=True)
    
    task_list = []

    for category, category_dir in [("vue", original_dir), ("std", original_dir), ("generated", generated_dir)]:
        print(f"Processing category: {category.upper()}")
        # 각 데이터셋 폴더 순회
        for dataset_name in args.dataset_names:
            data_dir = os.path.join(category_dir, dataset_name)
            
            # 환자 디렉토리 검색
            patient_dirs = sorted([d for d in glob.glob(os.path.join(data_dir, '*')) if os.path.isdir(d)])
            
            # 환자 디렉토리 순회
            for patient_dir in tqdm(patient_dirs, desc=f"Patients in {dataset_name}"):
                patient_id = os.path.basename(patient_dir)
                
                if (dataset_name, patient_id) not in task_list:
                    task_list.append((dataset_name, patient_id))
                
                npy_output_path = os.path.join(calculated_data_dir, f"{dataset_name}_{patient_id}_{category}.npy")
                if os.path.exists(npy_output_path):
                    continue
                
                if category == "vue":
                    patient_dir = os.path.join(patient_dir, args.ncct_folder)
                elif category == "std":
                    patient_dir = os.path.join(patient_dir, args.cect_folder)
                    
                dcm_list = sorted(glob.glob(os.path.join(patient_dir, '*.dcm')))
                
                # Pixel Data 초기화
                pixel_data_list = []

                for dcm_path in dcm_list:
                    try:
                        dcm = pydicom.dcmread(dcm_path)                        
                        z_position = dcm.get('ImagePositionPatient', [0.0, 0.0, 0.0])[2]
                        pixel_data_list.append([dcm.pixel_array, z_position])
                        
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
                    
                    np.save(npy_output_path, pixel_data_array)
                    
    return output_dir, calculated_data_dir, task_list


def calculate_mae(slice_list_1, slice_list_2):
    """Mean Absolute Error 계산"""
    mae_list = []
    for img1, img2 in zip(slice_list_1, slice_list_2):
        mae = np.mean(np.abs(img1 - img2))
        mae_list.append(mae)
    return mae_list, np.mean(mae_list)


def calculate(output_dir, calculated_data_dir, task_list):
    """MAE 계산 및 저장"""

    for task in task_list:
        dataset_name, patient_id = task
        result_csv_path = os.path.join(output_dir, f"{dataset_name}_{patient_id}_mae_results.csv")
        
        if os.path.exists(result_csv_path):
            continue
        
        # NPZ 파일 경로 설정
        vue_npy_path = os.path.join(calculated_data_dir, f"{dataset_name}_{patient_id}_vue.npy")
        std_npy_path = os.path.join(calculated_data_dir, f"{dataset_name}_{patient_id}_std.npy")
        generated_npy_path = os.path.join(calculated_data_dir, f"{dataset_name}_{patient_id}_generated.npy")
        
        if not (os.path.exists(vue_npy_path) and os.path.exists(std_npy_path) and os.path.exists(generated_npy_path)):
            continue
        
        # NPZ 파일 로드
        vue_slices = np.load(vue_npy_path)
        std_slices = np.load(std_npy_path)
        generated_slices = np.load(generated_npy_path)
        print(f"Loaded slices for {dataset_name} - {patient_id}: VUE({vue_slices.shape}), STD({std_slices.shape}), GENERATED({generated_slices.shape})")
        
        # # MAE 계산
        # mae_generated_std_slices, mae_generated_std_mean = calculate_mae(generated_slices, std_slices)
        
        # # 결과 저장
        # with open(result_csv_path, mode='w', newline='') as csv_file:
        #     csv_writer = csv.writer(csv_file)
        #     csv_writer.writerow(['Slice Index', 'MAE Generated-STD'])
        #     for idx in range(len(mae_generated_std_slices)):
        #         csv_writer.writerow([idx, mae_generated_std_slices[idx]])
        #     csv_writer.writerow(['Mean', mae_generated_std_mean])
        
        # print(f"MAE results saved to {result_csv_path}")
        print(f"MAE calculation completed for {dataset_name} - {patient_id}, {np.mean(np.abs(generated_slices - std_slices)):.2f}")


if __name__ == "__main__":
    print("Starting DUCOSY-GAN Anonymization Process")
    # 공통 추론 인자
    args = get_common_infer_args()
    
    # 변환 실행
    output_dir, calculated_data_dir, task_list = convert(args)
    
    # 계산 실행
    calculate(output_dir, calculated_data_dir, task_list)
    
    
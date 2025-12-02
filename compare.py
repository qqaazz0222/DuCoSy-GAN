import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 백그라운드에서 실행
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

from modules.argmanager import get_common_infer_args


def process_single_patient(patient_info):
    """
    단일 환자의 모든 슬라이스를 처리하는 함수 (병렬 처리용)
    
    Args:
        patient_info: (patient_id, data_dir, output_dir) 튜플
    
    Returns:
        (patient_id, success, slice_count, error_message) 튜플
    """
    patient_id, data_dir, output_dir = patient_info
    
    try:
        cect_path = os.path.join(data_dir, f'{patient_id}_std.npy')
        scect_path = os.path.join(data_dir, f'{patient_id}_generated.npy')

        if not os.path.exists(cect_path) or not os.path.exists(scect_path):
            return (patient_id, False, 0, "Missing files")
        
        cect_data = np.load(cect_path)
        scect_data = np.load(scect_path)
        
        diff = cect_data - scect_data
        threshold = 320
        diff = np.clip(diff, -threshold, threshold)
        
        patient_dir = os.path.join(output_dir, patient_id)
        os.makedirs(patient_dir, exist_ok=True)
        
        # 각 슬라이스를 순차적으로 처리 (환자별 병렬화이므로 슬라이스는 순차 처리)
        success_count = 0
        error_count = 0
        
        for i in range(diff.shape[0]):
            try:
                plt.figure(figsize=(12, 6))
                
                plt.subplot(1, 3, 1)
                plt.title('CECT')
                plt.imshow(cect_data[i], cmap='gray')
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.title('Synthesized CECT')
                plt.imshow(scect_data[i], cmap='gray')
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.title('Difference')
                plt.imshow(diff[i], cmap='jet', vmin=-threshold, vmax=threshold)
                plt.colorbar()
                plt.axis('off')
                
                plt.tight_layout()
                output_path = os.path.join(patient_dir, f'slice_{i:03d}.png')
                plt.savefig(output_path, dpi=100, bbox_inches='tight')
                plt.close()
                
                success_count += 1
                
            except Exception as e:
                plt.close('all')  # 메모리 누수 방지
                error_count += 1
        
        return (patient_id, True, success_count, None if error_count == 0 else f"{error_count} slices failed")
        
    except Exception as e:
        import traceback
        error_msg = f"Exception: {str(e)}\n{traceback.format_exc()}"
        return (patient_id, False, 0, error_msg)


def compare(args, fast_mode_flag = False, mask_flag = False):
    
    output_dir = os.path.join(args.output_dir_root, 'compared')
    os.makedirs(output_dir, exist_ok=True)
    if not mask_flag:
        data_dir = os.path.join(args.output_dir_root, 'calculated', 'data')
        output_dir = os.path.join(output_dir, 'original')
    else:
        data_dir = os.path.join(args.output_dir_root, 'calculated_mask', 'data')
        output_dir = os.path.join(output_dir, 'masked')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Data directory: {data_dir}")

    file_list = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    
    patient_ids = list(set("_".join(f.split('_')[:-1]) for f in file_list))
    
    # 병렬 처리 설정 (환자별 병렬 처리)
    if fast_mode_flag:
        num_workers = cpu_count()  # 모든 CPU 코어 사용
    else:
        num_workers = getattr(args, 'num_workers', cpu_count() // 2)  # CPU 코어의 절반 사용
    num_workers = max(1, min(num_workers, cpu_count()))  # 최소 1, 최대 CPU 코어 수
    
    print(f"\nProcessing Configuration:")
    print(f" - Total patients: {len(patient_ids)}")
    print(f" - Patient workers (parallel processes): {num_workers}")
    print(f" - Output directory: {output_dir}")

    # 환자별 정보 준비
    patient_info_list = [
        (patient_id, data_dir, output_dir)
        for patient_id in patient_ids
    ]
    
    # 환자별 병렬 처리
    success_count = 0
    total_slices = 0
    
    print("\nProcessing patients in parallel...")
    
    if num_workers > 1:
        # 멀티프로세싱으로 환자별 병렬 처리
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_patient, patient_info_list),
                total=len(patient_info_list),
                desc='Comparing CECT and Synthesized CECT',
                unit='patient'
            ))
            
            for patient_id, success, slice_count, error_msg in results:
                if success:
                    success_count += 1
                    total_slices += slice_count
                    if error_msg:
                        print(f"\n ⚠ Warning: {patient_id} - {error_msg}")
                else:
                    print(f"\n ✗ Failed: {patient_id} - {error_msg}")
    else:
        # 단일 프로세스로 순차 처리
        for patient_info in tqdm(patient_info_list, desc='Comparing CECT and Synthesized CECT', unit='patient'):
            patient_id, success, slice_count, error_msg = process_single_patient(patient_info)
            
            if success:
                success_count += 1
                total_slices += slice_count
                if error_msg:
                    print(f"\n ⚠ Warning: {patient_id} - {error_msg}")
            else:
                print(f"\n ✗ Failed: {patient_id} - {error_msg}")
    
    print(f"\nComparison completed!")
    print(f" - Patients processed: {success_count}/{len(patient_ids)}")
    print(f" - Total slices: {total_slices}")
    print(f" - Output directory: {output_dir}")

if __name__ == "__main__":
    print("Starting comparison of CECT and Synthesized CECT images...")
    # 공통 인자
    args = get_common_infer_args()
    fast_mode_flag = args.fast
    mask_flag = args.mask
    
    if fast_mode_flag:
        print("Fast mode enabled: All CPU core will be used.")
    
    # 비교 실행
    compare(args, fast_mode_flag, mask_flag)
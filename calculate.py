import os
import glob
import pickle
import shutil
import pydicom
import numpy as np
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
import csv

from modules.argmanager import get_common_infer_args

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")
import traceback

def convert(args, reset_flag, mask_flag = False, skip_convert_flag=False):
    """DICOM 파일 압축"""
    
    def get_hu_array(dcm):
        """DICOM 객체에서 HU 배열을 추출하는 헬퍼 함수"""
        intercept = dcm.RescaleIntercept if 'RescaleIntercept' in dcm else 0
        slope = dcm.RescaleSlope if 'RescaleSlope' in dcm else 1
        hu_array = dcm.pixel_array.astype(np.float32) * slope + intercept
        return hu_array
    
    print("Starting DICOM to NPZ conversion...")
    
    # 출력 디렉토리 생성
    original_dir = os.path.join(args.input_dir_root)
    masked_dir = os.path.join(args.output_dir_root, "masked")
    generated_dir = os.path.join(args.output_dir_root)
    if not mask_flag:
        output_dir = os.path.join(args.output_dir_root, "calculated")
    else:
        output_dir = os.path.join(args.output_dir_root, "calculated_mask")
    calculated_data_dir = os.path.join(output_dir, "data")
    
    if reset_flag and os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Resetting output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(calculated_data_dir, exist_ok=True)
    
    task_list = []
    
    if not mask_flag:
        category_dirs = [("vue", original_dir), ("std", original_dir), ("generated", generated_dir)]
    else:
        category_dirs = [("vue", masked_dir), ("std", masked_dir), ("generated", masked_dir)]

    for category, category_dir in category_dirs:
        print(f"\nProcessing Category: {category.upper()}, Directory: {category_dir}")
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
                    
                if skip_conversion_flag:
                    continue
                
                npy_output_path = os.path.join(calculated_data_dir, f"{dataset_name}_{patient_id}_{category}.npy")
                if os.path.exists(npy_output_path):
                    continue
                
                if category == "std":
                    patient_dir = os.path.join(patient_dir, args.cect_folder)
                elif category == "vue":
                    patient_dir =  os.path.join(patient_dir, args.ncct_folder)
                elif category == "generated" and mask_flag:
                    patient_dir = os.path.join(patient_dir, "generated")
                    
                dcm_list = sorted(glob.glob(os.path.join(patient_dir, '*.dcm')))
                
                # Pixel Data 초기화
                pixel_data_list = []
                slice_thickness = None

                for dcm_path in dcm_list:
                    try:
                        dcm = pydicom.dcmread(dcm_path)                        
                        z_position = dcm.get('ImagePositionPatient', [0.0, 0.0, 0.0])[2]
                        
                        # SliceThickness 추출 (첫 번째 슬라이스에서만)
                        if slice_thickness is None:
                            slice_thickness = float(dcm.get('SliceThickness', 1.0))
                        
                        # HU 변환
                        pixel_array = get_hu_array(dcm)
                        
                        pixel_data_list.append([pixel_array, z_position])
                        
                    except Exception as e:
                        print(f"Could not process file {dcm_path}. Error: {e}")
                        
                # Pixel Data 저장
                if pixel_data_list:
                    # Z 위치 기준으로 정렬
                    pixel_data_list.sort(key=lambda x: x[1])
                    
                    # Z 위치 정보 추출
                    z_positions = [item[1] for item in pixel_data_list]
                    pixel_arrays = [item[0] for item in pixel_data_list]
                    
                    # Z 위치 간격 확인 및 보간이 필요한지 체크
                    z_min = min(z_positions)
                    z_max = max(z_positions)
                    expected_num_slices = int(round((z_max - z_min) / slice_thickness)) + 1
                    
                    if expected_num_slices == len(pixel_arrays):
                        # 간격이 일정한 경우: 단순 스택
                        pixel_data_array = np.stack(pixel_arrays, axis=0)
                    else:
                        # 간격이 불균일한 경우: 메타데이터에 Z 위치 정보 포함
                        print(f"Warning: Irregular slice spacing detected for {patient_id}")
                        print(f"  Expected slices: {expected_num_slices}, Actual slices: {len(pixel_arrays)}")
                        print(f"  Z range: {z_min:.2f} to {z_max:.2f}, Thickness: {slice_thickness}")
                        
                        # 불균일하더라도 일단 스택 (추후 보간 로직 추가 가능)
                        pixel_data_array = np.stack(pixel_arrays, axis=0)
                    
                    # Z 위치 메타데이터와 함께 저장
                    np.save(npy_output_path, pixel_data_array)
                    
                    # Z 위치 정보를 별도 파일로 저장
                    z_positions_path = npy_output_path.replace('.npy', '_z_positions.npy')
                    np.save(z_positions_path, np.array(z_positions))
                    
                    # 메타데이터 저장 (slice thickness 등)
                    metadata_path = npy_output_path.replace('.npy', '_metadata.pkl')
                    metadata = {
                        'z_positions': z_positions,
                        'slice_thickness': slice_thickness,
                        'z_min': z_min,
                        'z_max': z_max,
                        'num_slices': len(pixel_arrays)
                    }
                    with open(metadata_path, 'wb') as f:
                        pickle.dump(metadata, f)
                    
    return output_dir, calculated_data_dir, task_list


def calculate_mae(original_slice_list, generated_slice_list):
    """Mean Absolute Error 계산"""
    mae_list_all = np.abs(original_slice_list - generated_slice_list)
    mae = np.mean(mae_list_all)
    # 각 슬라이스별 MAE 계산
    mae_list = [np.mean(mae_slice) for mae_slice in mae_list_all]
    return mae, mae_list

def calculate_psnr(original_slice_list, generated_slice_list):
    """Peak Signal-to-Noise Ratio 계산"""
    mse_list_all = (np.abs(original_slice_list - generated_slice_list)) ** 2
    mse = np.mean(mse_list_all)
    if mse == 0:
        return float('inf'), [float('inf')] * len(original_slice_list)
    max_pixel = np.max(original_slice_list)
    # 각 슬라이스별 PSNR 계산
    psnr_list = []
    for mse_slice in mse_list_all:
        mse_slice_mean = np.mean(mse_slice)
        if mse_slice_mean == 0:
            psnr_list.append(float('inf'))
        else:
            psnr_list.append(20 * np.log10(max_pixel / np.sqrt(mse_slice_mean)))
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr, psnr_list

def calculate_ms_ssim(original_slice_list, generated_slice_list):
    """Multi-Scale Structural Similarity Index 계산"""
    ms_ssim_list = []
    for orig_slice, gen_slice in zip(original_slice_list, generated_slice_list):
        ssim_value = ssim(orig_slice, gen_slice, data_range=gen_slice.max() - gen_slice.min())
        ms_ssim_list.append(ssim_value)
    ms_ssim = np.mean(ms_ssim_list)
    return ms_ssim, ms_ssim_list

def visualize_mae(mae_list, output_path):
    """ MAE 시각화 """
    try:
        mae_vue_std, mae_std_gen, mae_vue_gen = zip(*mae_list)
        
        x = np.array(mae_vue_std)
        y = np.array(mae_vue_gen)
        sizes = np.array(mae_std_gen)

        if x.size and y.size and sizes.size and (len(x) == len(y) == len(sizes)):
            _, ax = plt.subplots(figsize=(8,6))
            
            # Scatter plot with points
            scatter = ax.scatter(x, y, c=y, s=50, cmap='viridis', alpha=0.7, edgecolors='black', linewidth=0.5)
            
            # colorbar for VUE vs Generated MAE
            cb = plt.colorbar(scatter, ax=ax, pad=0.02)
            cb.set_label('MAE (VUE vs Generated)')

            # 1:1 reference line
            vmin = min(x.min(), y.min())
            vmax = max(x.max(), y.max())
            pad = (vmax - vmin) * 0.05 if vmax > vmin else (abs(vmin) * 0.1 if vmin != 0 else 1.0)
            ax.plot([vmin-pad, vmax+pad], [vmin-pad, vmax+pad], 'r--', linewidth=1)
            ax.set_xlabel('MAE (NCCT vs CECT)')
            ax.set_ylabel('MAE (NCCT vs sCECT)')
            ax.set_title('MAE Comparison')
            ax.grid(alpha=0.3)
            ax.set_xlim(vmin-pad, vmax+pad)
            ax.set_ylim(vmin-pad, vmax+pad)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved MAE plot: {output_path}")
        else:
            print("No MAE data available for standard MAE plot or mismatched lengths.")
    except Exception as e:
        print(f"Error generating MAE plot: {e}")
        
def visualize_psnr(psnr_list, output_path):
    """ PSNR 시각화 - 슬라이스별 평균 ± 표준편차 """
    if not psnr_list:
        print("시각화할 PSNR 데이터가 없습니다.")
        return

    try:
        psnr_ncct_vs_cect_patients, psnr_cect_vs_scect_patients, psnr_ncct_vs_scect_patients = zip(*psnr_list)

        data_to_plot = [
            list(psnr_ncct_vs_cect_patients),
            list(psnr_cect_vs_scect_patients),
            list(psnr_ncct_vs_scect_patients)
        ]

        fig, ax = plt.subplots(figsize=(8, 6)) # 박스 플롯에 적합한 크기

        box_plot = ax.boxplot(data_to_plot,
                              patch_artist=True, # 박스 내부 색상 채우기
                              showfliers=True, # 이상치 표시 (True or False)
                              labels=['NCCT vs CECT\n(Baseline)',
                                      'CECT vs sCECT\n(Generated)',
                                      'NCCT vs sCECT'],
                              widths=0.6) # 박스 너비 조정

        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)

        for median in box_plot['medians']:
            median.set_color('black')
            median.set_linewidth(1.5)

        ax.set_ylabel('PSNR (dB)', fontsize=12)
        ax.set_title(f'Distribution of PSNR', fontsize=14)
        ax.yaxis.grid(True, linestyle='--', alpha=0.6) # Y축 그리드만 표시
        ax.tick_params(axis='both', which='major', labelsize=10)
        plt.xticks(rotation=0, ha='center') # X축 라벨 회전 없음
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved PSNR box plot: {output_path}")

    except Exception as e:
        print(f"Error generating PSNR plot: {e}")
        
        
def visualize_ms_ssim_box_plot(ms_ssim_list, output_path):
    """ MS-SSIM 시각화 """
    if not ms_ssim_list:
        print("시각화할 MS-SSIM 데이터가 없습니다.")
        return

    try:
        ms_ssim_ncct_vs_cect_patients, ms_ssim_cect_vs_scect_patients, ms_ssim_ncct_vs_scect_patients = zip(*ms_ssim_list)
        data_to_plot = [
            list(ms_ssim_ncct_vs_cect_patients),
            list(ms_ssim_cect_vs_scect_patients),
            list(ms_ssim_ncct_vs_scect_patients)
        ]
        # 그래프 생성
        fig, ax = plt.subplots(figsize=(8, 6))
        box_plot = ax.boxplot(data_to_plot,
                            patch_artist=True, # 박스 내부 색상 채우기
                            showfliers=True, # 이상치 표시 (True or False)
                            labels=['NCCT vs CECT\n(Baseline)',
                                    'CECT vs sCECT\n(Generated)',
                                    'NCCT vs sCECT'],
                            widths=0.6)

        # 박스 색상 설정 (옵션)
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)

        # 중앙값(median) 선 색상 변경 (옵션)
        for median in box_plot['medians']:
            median.set_color('black')
            median.set_linewidth(1.5)

        # 그래프 설정
        ax.set_ylabel('Average MS-SSIM per Patient (0-1)', fontsize=12)
        ax.set_ylim(bottom=min(0, np.min(data_to_plot)-0.05), top=max(1, np.max(data_to_plot)+0.05)) # Y축 범위 설정 (0~1 포함)
        ax.set_title(f'Distribution of Average Patient MS-SSIM', fontsize=14)
        ax.yaxis.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=10)
        plt.xticks(rotation=0, ha='center')
        plt.tight_layout()

        # 그래프 저장
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"환자별 평균 MS-SSIM 박스 플롯 저장 완료: {output_path}")

    except Exception as e:
        print(f"환자별 평균 MS-SSIM 박스 플롯 생성 중 오류 발생: {e}")
        
        
def visualize_ms_ssim_violin_plot(ms_ssim_list, output_path):
    """ MS-SSIM 시각화 """
    if not ms_ssim_list:
        print("시각화할 MS-SSIM 데이터가 없습니다.")
        return

    try:
        # Seaborn에서 사용하기 좋도록 DataFrame 형태로 변환
        data = []
        labels = ['NCCT vs CECT\n(Baseline)', 'CECT vs sCECT\n(Generated)', 'NCCT vs sCECT']
        for i, label in enumerate(labels):
            for patient_data in ms_ssim_list:
                data.append({'Comparison': label, 'Average MS-SSIM': patient_data[i]})

        df = pd.DataFrame(data)

        # 그래프 생성
        plt.figure(figsize=(8, 6)) # 바이올린 플롯에 적합한 크기 조정
        ax = sns.violinplot(x='Comparison', y='Average MS-SSIM', data=df,
                            palette=['lightblue', 'lightgreen', 'lightcoral'], # 색상 지정
                            inner='quartile', # 바이올린 내부에 사분위수 표시 ('box', 'quartile', 'point', 'stick', None)
                            cut=0) # 데이터 범위를 넘어서는 꼬리 부분 자르기

        # 그래프 설정
        ax.set_xlabel('') # X축 라벨은 범례 역할로 충분
        ax.set_ylabel('Average MS-SSIM per Patient (0-1)', fontsize=12)
        ax.set_ylim(bottom=min(0, df['Average MS-SSIM'].min()-0.05), top=max(1, df['Average MS-SSIM'].max()+0.05)) # Y축 범위 설정 (0~1 포함)
        ax.set_title(f'Distribution of Average Patient MS-SSIM', fontsize=14)
        ax.yaxis.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=10)
        plt.xticks(rotation=0, ha='center')
        plt.tight_layout()

        # 그래프 저장
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close() # plt.figure()로 생성했으므로 plt.close() 사용
        print(f"환자별 평균 MS-SSIM 바이올린 플롯 저장 완료: {output_path}")

    except ImportError:
        print("오류: 이 기능을 사용하려면 'seaborn'과 'pandas' 라이브러리가 필요합니다.")
        print("pip install seaborn pandas")
    except Exception as e:
        print(f"환자별 평균 MS-SSIM 바이올린 플롯 생성 중 오류 발생: {e}")


def calculate(output_dir, calculated_data_dir, task_list):
    """MAE 계산 및 저장"""
    print("\nCalculating MAE for generated images...")
    
    result_path = os.path.join(output_dir, "result.pkl")
    detail_result_dir = os.path.join(output_dir, "detail")
    os.makedirs(detail_result_dir, exist_ok=True)
    if os.path.exists(result_path):
        with open(result_path, 'rb') as f:
            mae_list, psnr_list, ms_ssim_list = pickle.load(f)
        print("Loaded existing results from pickle.")
    else:
        mae_list = []
        psnr_list = []
        ms_ssim_list = []

        for task in tqdm(task_list, desc="Calculating MAE for tasks"):
            try:
                dataset_name, patient_id = task
                mae_result_csv_path = os.path.join(detail_result_dir, f"{dataset_name}_{patient_id}_mae_results.csv")
                psnr_result_csv_path = os.path.join(detail_result_dir, f"{dataset_name}_{patient_id}_psnr_results.csv")
                ms_ssim_result_csv_path = os.path.join(detail_result_dir, f"{dataset_name}_{patient_id}_ms_ssim_results.csv")
                header = ['Slice_Index', 'MAE_VUE_vs_STD', 'MAE_STD_vs_Generated', 'MAE_VUE_vs_Generated']

                # NPZ 파일 경로 설정
                vue_npy_path = os.path.join(calculated_data_dir, f"{dataset_name}_{patient_id}_vue.npy")
                std_npy_path = os.path.join(calculated_data_dir, f"{dataset_name}_{patient_id}_std.npy")
                generated_npy_path = os.path.join(calculated_data_dir, f"{dataset_name}_{patient_id}_generated.npy")

                if not os.path.exists(std_npy_path) and os.path.exists(generated_npy_path):
                    continue
                
                # NPZ 파일 로드
                vue_slices = np.load(vue_npy_path)
                std_slices = np.load(std_npy_path)
                generated_slices = np.load(generated_npy_path)
                
                if vue_slices.shape != std_slices.shape != generated_slices.shape:
                    min_length = min(vue_slices.shape[0], std_slices.shape[0], generated_slices.shape[0])
                    vue_slices = vue_slices[:min_length]
                    std_slices = std_slices[:min_length]
                    generated_slices = generated_slices[:min_length]
                
                print("--------------------------------------------------------------------------------------")
                
                # MAE 계산
                mae_vue_std, mae_vue_std_list = calculate_mae(std_slices, vue_slices)
                mae_std_gen, mae_std_gen_list = calculate_mae(std_slices, generated_slices)
                mae_vue_gen, mae_vue_gen_list = calculate_mae(vue_slices, generated_slices)
                mae_list.append([mae_vue_std, mae_std_gen, mae_vue_gen])
                # MAE 결과를 CSV에 저장
                with open(mae_result_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(header)
                    for slice_idx, (mae_vs, mae_sg, mae_vg) in enumerate(zip(mae_vue_std_list, mae_std_gen_list, mae_vue_gen_list)):
                        # numpy 타입을 Python float로 변환
                        writer.writerow([slice_idx, f"{float(mae_vs):.6f}", f"{float(mae_sg):.6f}", f"{float(mae_vg):.6f}"])
                
                print(f"MAE (VUE vs STD): {mae_vue_std:.4f}, MAE (STD vs Generated): {mae_std_gen:.4f}, MAE (VUE vs Generated): {mae_vue_gen:.4f}")
                
                # PSNR 계산
                psnr_vue_std, psnr_vue_std_list = calculate_psnr(std_slices, vue_slices)
                psnr_std_gen, psnr_std_gen_list = calculate_psnr(std_slices, generated_slices)
                psnr_vue_gen, psnr_vue_gen_list = calculate_psnr(vue_slices, generated_slices)
                psnr_list.append([psnr_vue_std, psnr_std_gen, psnr_vue_gen])
                # PSNR 결과를 CSV에 저장
                with open(psnr_result_csv_path, 'w', newline='', encoding='utf-8') as csvfile:   
                    writer = csv.writer(csvfile)
                    writer.writerow(['Slice_Index', 'PSNR_VUE_vs_STD', 'PSNR_STD_vs_Generated', 'PSNR_VUE_vs_Generated'])
                    for slice_idx, (psnr_vs, psnr_sg, psnr_vg) in enumerate(zip(psnr_vue_std_list, psnr_std_gen_list, psnr_vue_gen_list)):
                        # numpy 타입을 Python float로 변환
                        writer.writerow([slice_idx, f"{float(psnr_vs):.6f}", f"{float(psnr_sg):.6f}", f"{float(psnr_vg):.6f}"])
                        
                print(f"PSNR (VUE vs STD): {psnr_vue_std:.4f}, PSNR (STD vs Generated): {psnr_std_gen:.4f}, PSNR (VUE vs Generated): {psnr_vue_gen:.4f}")
                
                # MS-SSIM 계산
                ms_ssim_vue_std, ms_ssim_vue_std_list = calculate_ms_ssim(std_slices, vue_slices)
                ms_ssim_std_gen, ms_ssim_std_gen_list = calculate_ms_ssim(std_slices, generated_slices)
                ms_ssim_vue_gen, ms_ssim_vue_gen_list = calculate_ms_ssim(vue_slices, generated_slices)
                ms_ssim_list.append([ms_ssim_vue_std, ms_ssim_std_gen, ms_ssim_vue_gen])
                # MS-SSIM 결과를 CSV에 저장
                with open(ms_ssim_result_csv_path, 'w', newline='', encoding='utf-8') as csvfile:   
                    writer = csv.writer(csvfile)
                    writer.writerow(['Slice_Index', 'MS_SSIM_VUE_vs_STD', 'MS_SSIM_STD_vs_Generated', 'MS_SSIM_VUE_vs_Generated'])
                    for slice_idx, (ms_ssim_vs, ms_ssim_sg, ms_ssim_vg) in enumerate(zip(ms_ssim_vue_std_list, ms_ssim_std_gen_list, ms_ssim_vue_gen_list)):
                        # numpy 타입을 Python float로 변환
                        writer.writerow([slice_idx, f"{float(ms_ssim_vs):.6f}", f"{float(ms_ssim_sg):.6f}", f"{float(ms_ssim_vg):.6f}"])
                print(f"MS-SSIM (VUE vs STD): {ms_ssim_vue_std:.4f}, MS-SSIM (STD vs Generated): {ms_ssim_std_gen:.4f}, MS-SSIM (VUE vs Generated): {ms_ssim_vue_gen:.4f}")
                
                print("--------------------------------------------------------------------------------------")

            except Exception as e:
                print(f"Error processing task {task}: {e}")
                traceback.print_exc()  # 전체 traceback을 확인하기 위해 추가
                
        # 결과 저장
        with open(result_path, 'wb') as f:
            pickle.dump((mae_list, psnr_list, ms_ssim_list), f)
        print(f"Saved results to pickle: {result_path}")
        
    mae_p_list = [[task_list[idx], mae_list[idx]] for idx in range(len(mae_list))]
    psnr_p_list = [[task_list[idx], psnr_list[idx]] for idx in range(len(psnr_list))]
    ms_ssim_p_list = [[task_list[idx], ms_ssim_list[idx]] for idx in range(len(ms_ssim_list))]
    
    mae_p_list.sort(key=lambda x: x[1][1], reverse=True)  # std vs generated 기준 정렬
    psnr_p_list.sort(key=lambda x: x[1][1])  # std vs generated 기준 정렬
    ms_ssim_p_list.sort(key=lambda x: x[1][1])  # std vs generated 기준 정렬
    
    print('--------------------------------------------------------------------------------------')
    print("Worst 5 MAE (STD vs Generated):")
    for item in mae_p_list[:10]:
        print(f"Patient: {item[0]}, MAE: {item[1]}")
    print('--------------------------------------------------------------------------------------')
    print("Worst 5 PSNR (STD vs Generated):")
    for item in psnr_p_list[:10]:
        print(f"Patient: {item[0]}, PSNR: {item[1]}")
    print('--------------------------------------------------------------------------------------')
    print("Worst 5 MS-SSIM (STD vs Generated):")
    for item in ms_ssim_p_list[:10]:
        print(f"Patient: {item[0]}, MS-SSIM: {item[1]}")
    print('--------------------------------------------------------------------------------------')

    # CSV 파일로 결과 저장
    try:
        print("\nSaving results to CSV files...")
        
        # 전체 결과를 하나의 CSV로 저장
        all_results_csv_path = os.path.join(output_dir, "all_results.csv")
        with open(all_results_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # 헤더 작성
            writer.writerow([
                'Dataset', 'Patient_ID',
                'MAE_NCCT_vs_CECT', 'MAE_CECT_vs_sCECT', 'MAE_NCCT_vs_sCECT',
                'PSNR_NCCT_vs_CECT', 'PSNR_CECT_vs_sCECT', 'PSNR_NCCT_vs_sCECT',
                'MS_SSIM_NCCT_vs_CECT', 'MS_SSIM_CECT_vs_sCECT', 'MS_SSIM_NCCT_vs_sCECT'
            ])
            
            # 데이터 작성
            for idx, task in enumerate(task_list):
                dataset_name, patient_id = task
                mae_values = mae_list[idx]
                psnr_values = psnr_list[idx]
                ms_ssim_values = ms_ssim_list[idx]
                
                writer.writerow([
                    dataset_name, patient_id,
                    f"{mae_values[0]:.6f}", f"{mae_values[1]:.6f}", f"{mae_values[2]:.6f}",
                    f"{psnr_values[0]:.6f}", f"{psnr_values[1]:.6f}", f"{psnr_values[2]:.6f}",
                    f"{ms_ssim_values[0]:.6f}", f"{ms_ssim_values[1]:.6f}", f"{ms_ssim_values[2]:.6f}"
                ])
        
        print(f"Saved all results to: {all_results_csv_path}")
        
        # 통계 요약 CSV 저장
        summary_csv_path = os.path.join(output_dir, "summary_statistics.csv")
        with open(summary_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Metric', 'Comparison', 'Mean', 'Std', 'Min', 'Max', 'Median'])
            
            # MAE 통계
            mae_array = np.array(mae_list)
            for idx, comparison in enumerate(['NCCT_vs_CECT', 'CECT_vs_sCECT', 'NCCT_vs_sCECT']):
                data = mae_array[:, idx]
                writer.writerow([
                    'MAE', comparison,
                    f"{np.mean(data):.6f}",
                    f"{np.std(data):.6f}",
                    f"{np.min(data):.6f}",
                    f"{np.max(data):.6f}",
                    f"{np.median(data):.6f}"
                ])
            
            # PSNR 통계
            psnr_array = np.array(psnr_list)
            for idx, comparison in enumerate(['NCCT_vs_CECT', 'CECT_vs_sCECT', 'NCCT_vs_sCECT']):
                data = psnr_array[:, idx]
                writer.writerow([
                    'PSNR', comparison,
                    f"{np.mean(data):.6f}",
                    f"{np.std(data):.6f}",
                    f"{np.min(data):.6f}",
                    f"{np.max(data):.6f}",
                    f"{np.median(data):.6f}"
                ])
            
            # MS-SSIM 통계
            ms_ssim_array = np.array(ms_ssim_list)
            for idx, comparison in enumerate(['NCCT_vs_CECT', 'CECT_vs_sCECT', 'NCCT_vs_sCECT']):
                data = ms_ssim_array[:, idx]
                writer.writerow([
                    'MS-SSIM', comparison,
                    f"{np.mean(data):.6f}",
                    f"{np.std(data):.6f}",
                    f"{np.min(data):.6f}",
                    f"{np.max(data):.6f}",
                    f"{np.median(data):.6f}"
                ])
        
        print(f"Saved summary statistics to: {summary_csv_path}")
        
        # Worst cases CSV 저장
        worst_cases_csv_path = os.path.join(output_dir, "worst_cases.csv")
        with open(worst_cases_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # MAE worst cases
            writer.writerow(['Metric: MAE (CECT vs sCECT) - Worst 10 Cases'])
            writer.writerow(['Rank', 'Dataset', 'Patient_ID', 'MAE_NCCT_vs_CECT', 'MAE_CECT_vs_sCECT', 'MAE_NCCT_vs_sCECT'])
            for rank, item in enumerate(mae_p_list[:10], 1):
                dataset_name, patient_id = item[0]
                mae_values = item[1]
                writer.writerow([
                    rank, dataset_name, patient_id,
                    f"{mae_values[0]:.6f}", f"{mae_values[1]:.6f}", f"{mae_values[2]:.6f}"
                ])
            
            writer.writerow([])  # 빈 줄
            
            # PSNR worst cases
            writer.writerow(['Metric: PSNR (CECT vs sCECT) - Worst 10 Cases'])
            writer.writerow(['Rank', 'Dataset', 'Patient_ID', 'PSNR_NCCT_vs_CECT', 'PSNR_CECT_vs_sCECT', 'PSNR_NCCT_vs_sCECT'])
            for rank, item in enumerate(psnr_p_list[:10], 1):
                dataset_name, patient_id = item[0]
                psnr_values = item[1]
                writer.writerow([
                    rank, dataset_name, patient_id,
                    f"{psnr_values[0]:.6f}", f"{psnr_values[1]:.6f}", f"{psnr_values[2]:.6f}"
                ])
            
            writer.writerow([])  # 빈 줄
            
            # MS-SSIM worst cases
            writer.writerow(['Metric: MS-SSIM (CECT vs sCECT) - Worst 10 Cases'])
            writer.writerow(['Rank', 'Dataset', 'Patient_ID', 'MS_SSIM_NCCT_vs_CECT', 'MS_SSIM_CECT_vs_sCECT', 'MS_SSIM_NCCT_vs_sCECT'])
            for rank, item in enumerate(ms_ssim_p_list[:10], 1):
                dataset_name, patient_id = item[0]
                ms_ssim_values = item[1]
                writer.writerow([
                    rank, dataset_name, patient_id,
                    f"{ms_ssim_values[0]:.6f}", f"{ms_ssim_values[1]:.6f}", f"{ms_ssim_values[2]:.6f}"
                ])
        
        print(f"Saved worst cases to: {worst_cases_csv_path}")
        
    except Exception as e:
        print(f"Error while saving CSV results: {e}")
        traceback.print_exc()

    # 시각화 및 결과 저장
    try:
        print("\nVisualizing and saving results...")
        visualize_mae(mae_list, os.path.join(output_dir, "mae_results.png"))
        visualize_psnr(psnr_list, os.path.join(output_dir, "psnr_results.png"))
        visualize_ms_ssim_box_plot(ms_ssim_list, os.path.join(output_dir, "ms_ssim_boxplot.png"))

    except Exception as e:
        print(f"Error while visualizing/saving results: {e}")
        traceback.print_exc()

def visualize():
    pass
        

if __name__ == "__main__":
    print("Starting DUCOSY-GAN Calculation Process")

    # 공통 추론 인자
    args = get_common_infer_args()
    reset_flag = args.reset
    mask_flag = args.mask
    skip_conversion_flag = args.skip_convert
    if mask_flag:
        print("Mask mode enabled: Calculations will be performed on masked data.")
    
    # 변환
    output_dir, calculated_data_dir, task_list = convert(args, reset_flag, mask_flag, skip_conversion_flag)

    # 계산
    calculate(output_dir, calculated_data_dir, task_list)
    
    # 시각화
    visualize()
    
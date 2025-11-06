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

def convert(args, reset_flag):
    """DICOM 파일 압축"""
    
    def get_hu_array(dcm):
        """DICOM 객체에서 HU 배열을 추출하는 헬퍼 함수"""
        intercept = dcm.RescaleIntercept if 'RescaleIntercept' in dcm else 0
        slope = dcm.RescaleSlope if 'RescaleSlope' in dcm else 1
        hu_array = dcm.pixel_array.astype(np.float32) * slope + intercept
        return hu_array
    
    # 출력 디렉토리 생성
    original_dir = os.path.join(args.input_dir_root)
    generated_dir = os.path.join(args.output_dir_root)
    output_dir = os.path.join(args.output_dir_root, "calculated")
    calculated_data_dir = os.path.join(output_dir, "data")
    
    if reset_flag and os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Resetting output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(calculated_data_dir, exist_ok=True)
    
    task_list = []

    for category, category_dir in [("vue", original_dir), ("std", original_dir), ("generated", generated_dir)]:
        print(f"\nProcessing category: {category.upper()}")
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
                
                if category == "std":
                    patient_dir = os.path.join(patient_dir, args.cect_folder)
                elif category == "vue":
                    patient_dir =  os.path.join(patient_dir, args.ncct_folder)
                    
                dcm_list = sorted(glob.glob(os.path.join(patient_dir, '*.dcm')))
                
                # Pixel Data 초기화
                pixel_data_list = []

                for dcm_path in dcm_list:
                    try:
                        dcm = pydicom.dcmread(dcm_path)                        
                        z_position = dcm.get('ImagePositionPatient', [0.0, 0.0, 0.0])[2]
                        
                        # HU 변환
                        pixel_array = get_hu_array(dcm)
                        
                        pixel_data_list.append([pixel_array, z_position])
                        
                    except Exception as e:
                        print(f"Could not process file {dcm_path}. Error: {e}")
                        
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


def calculate_mae(original_slice_list, generated_slice_list):
    """Mean Absolute Error 계산"""
    mea = np.mean(np.abs(original_slice_list - generated_slice_list))
    return mea

def calculate_psnr(original_slice_list, generated_slice_list):
    """Peak Signal-to-Noise Ratio 계산"""
    mse = np.mean((original_slice_list - generated_slice_list) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = np.max(original_slice_list)
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ms_ssim(original_slice_list, generated_slice_list):
    """Multi-Scale Structural Similarity Index 계산"""
    ms_ssim_values = []
    for orig_slice, gen_slice in zip(original_slice_list, generated_slice_list):
        ssim_value = ssim(orig_slice, gen_slice, data_range=gen_slice.max() - gen_slice.min())
        ms_ssim_values.append(ssim_value)
    ms_ssim = np.mean(ms_ssim_values)
    return ms_ssim

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
                result_csv_path = os.path.join(output_dir, f"{dataset_name}_{patient_id}_mae_results.csv")
                
                if os.path.exists(result_csv_path):
                    continue
                
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
                mae_vue_std = calculate_mae(std_slices, vue_slices)
                mae_std_gen = calculate_mae(std_slices, generated_slices)
                mae_vue_gen = calculate_mae(vue_slices, generated_slices)
                mae_list.append([mae_vue_std, mae_std_gen, mae_vue_gen])
                print(f"MAE (VUE vs STD): {mae_vue_std:.4f}, MAE (STD vs Generated): {mae_std_gen:.4f}, MAE (VUE vs Generated): {mae_vue_gen:.4f}")
                
                # PSNR 계산
                psnr_vue_std = calculate_psnr(std_slices, vue_slices)
                psnr_std_gen = calculate_psnr(std_slices, generated_slices)
                psnr_vue_gen = calculate_psnr(vue_slices, generated_slices)
                psnr_list.append([psnr_vue_std, psnr_std_gen, psnr_vue_gen])
                print(f"PSNR (VUE vs STD): {psnr_vue_std:.4f}, PSNR (STD vs Generated): {psnr_std_gen:.4f}, PSNR (VUE vs Generated): {psnr_vue_gen:.4f}")
                
                # MS-SSIM 계산
                ms_ssim_vue_std = calculate_ms_ssim(std_slices, vue_slices)
                ms_ssim_std_gen = calculate_ms_ssim(std_slices, generated_slices)
                ms_ssim_vue_gen = calculate_ms_ssim(vue_slices, generated_slices)
                ms_ssim_list.append([ms_ssim_vue_std, ms_ssim_std_gen, ms_ssim_vue_gen])
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
        visualize_ms_ssim_violin_plot(ms_ssim_list, os.path.join(output_dir, "ms_ssim_violinplot.png"))

    except Exception as e:
        print(f"Error while visualizing/saving results: {e}")
        traceback.print_exc()
            
        

if __name__ == "__main__":
    print("Starting DUCOSY-GAN Calculation Process")

    # 공통 추론 인자
    args = get_common_infer_args()
    reset_flag = args.reset
    
    # 변환 실행
    output_dir, calculated_data_dir, task_list = convert(args, reset_flag)
    
    # 계산 실행
    calculate(output_dir, calculated_data_dir, task_list)
    
    
import os
import glob
import pickle
import shutil
import pydicom
import numpy as np
import argparse
import csv
import traceback
import concurrent.futures
import warnings
from tqdm import tqdm
from scipy.stats import pearsonr, wasserstein_distance
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim
from skimage.filters import sobel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# CUDA 메모리 최적화 설정
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Deep Learning based metrics
try:
    import torch
    import lpips
    from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
    DL_LIB_AVAILABLE = True
except ImportError:
    DL_LIB_AVAILABLE = False
    print("Warning: torch, torchmetrics, or lpips not found. MS-SSIM and LPIPS will be skipped.")

warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")

# --- Global Variables for Parallel Workers ---
lpips_loss_fn = None
ms_ssim_metric = None
use_gpu = False

def init_worker(use_gpu_flag=False):
    """병렬 프로세스 초기화: 모델 로드"""
    global lpips_loss_fn, ms_ssim_metric, use_gpu
    use_gpu = use_gpu_flag
    if DL_LIB_AVAILABLE:
        device = torch.device('cuda' if (torch.cuda.is_available() and use_gpu) else 'cpu')
        try:
            # LPIPS (AlexNet backbone is faster and standard for comparison)
            lpips_loss_fn = lpips.LPIPS(net='alex').to(device)
            lpips_loss_fn.eval()
        except Exception as e:
            print(f"Failed to load LPIPS in worker: {e}")
        
        try:
            # MS-SSIM
            ms_ssim_metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        except Exception as e:
            print(f"Failed to load MS-SSIM in worker: {e}")

# --- Argument Parser (Requested Function) ---
def get_common_infer_args():
    """ 공통 추론 인자 """
    parser = argparse.ArgumentParser(description="CycleGAN Inference and Metric Calculation")
    
    # 경로 관련 인자
    parser.add_argument("--data_dir_root", type=str, default="./data", help="Root directory of the data")
    parser.add_argument("--input_dir_root", type=str, default="./data/input", help="Root directory of the input datasets (GT)")
    parser.add_argument("--working_dir_root", type=str, default="./data/working", help="Root directory for saving inference results")
    parser.add_argument("--output_dir_root", type=str, default="./data/output", help="Root directory for saving merged results (and metrics)")
    parser.add_argument("--dataset_names", type=str, nargs='+', default=["Kyunghee_Univ"], help="List of dataset folder names to process")
    parser.add_argument("--ncct_folder", type=str, default="POST VUE", help="Folder name for non-contrast CT")
    parser.add_argument("--cect_folder", type=str, default="POST STD", help="Folder name for contrast-enhanced CT")
    parser.add_argument("--apply_masking", action='store_true', help="Whether to apply masking using TotalSegmentator (Inference Phase)")
    
    # 전처리 관련 인자
    parser.add_argument("--img_size", type=int, default=512, help="Size of images for model input")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of patients to process in parallel (Used as Num Workers)")
    
    # 모델 관련 인자
    parser.add_argument("--nmodel_path", type=str, default="./checkpoints/Normal_Map_Unet.pth", help="Path to the trained Normal Map U-Net model")

    # 시스템 관련 인자
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use for inference")
    
    # 성능 계산 관련 인자
    parser.add_argument('--fast', action='store_true', help='Set fast flag to True')
    parser.add_argument('--reset', action='store_true', help='Reset calculation output directory')
    parser.add_argument('--mask', action='store_true', help='Calculate metrics on MASKED data')
    parser.add_argument('--skip_convert', action='store_true', help='Skip DICOM to NPY conversion if already done')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for LPIPS and MS-SSIM (Memory intensive)')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of parallel workers (default: 1 to avoid OOM)')

    args = parser.parse_args()
    
    # 출력 폴더 생성
    os.makedirs(args.data_dir_root, exist_ok=True)
    os.makedirs(args.input_dir_root, exist_ok=True)
    os.makedirs(args.working_dir_root, exist_ok=True)
    os.makedirs(args.output_dir_root, exist_ok=True)
    
    return args

# --- Preprocessing Functions ---

def convert(args, reset_flag, mask_flag=False, skip_convert_flag=False):
    """DICOM to NPY Conversion"""
    
    def get_hu_array(dcm):
        intercept = dcm.RescaleIntercept if 'RescaleIntercept' in dcm else 0
        slope = dcm.RescaleSlope if 'RescaleSlope' in dcm else 1
        hu_array = dcm.pixel_array.astype(np.float32) * slope + intercept
        return hu_array
    
    print("Starting DICOM to NPZ conversion...")
    
    # 경로 설정
    # input_dir_root: 원본(GT) 데이터 위치
    # output_dir_root: 생성된(Generated) 데이터가 저장된 위치라고 가정
    original_dir = args.input_dir_root
    generated_root = args.output_dir_root
    
    # 결과 저장 경로 설정
    if mask_flag:
        calc_output_dir = os.path.join(args.output_dir_root, "calculated_mask")
        # 마스킹 모드일 경우 입력 데이터도 masked 폴더에서 찾도록 가정 (필요시 수정)
        # 예: output_dir_root/masked/Dataset...
        masked_gen_dir = os.path.join(args.output_dir_root, "masked")
    else:
        calc_output_dir = os.path.join(args.output_dir_root, "calculated")
        
    calculated_data_dir = os.path.join(calc_output_dir, "data")
    
    if reset_flag and os.path.exists(calc_output_dir):
        shutil.rmtree(calc_output_dir)
        print(f"Resetting output directory: {calc_output_dir}")
    
    os.makedirs(calc_output_dir, exist_ok=True)
    os.makedirs(calculated_data_dir, exist_ok=True)
    
    task_list = []
    
    # 처리할 카테고리 정의: (식별자, 상위 폴더 경로)
    # VUE/STD는 Input Root에서, Generated는 Output Root에서 찾음
    if mask_flag:
        # 마스킹된 데이터 경로 구조에 따라 수정 필요. 여기서는 generated와 동일한 구조 가정
        category_dirs = [
            ("vue", os.path.join(args.output_dir_root, "masked")), # 혹은 masked inputs path
            ("std", os.path.join(args.output_dir_root, "masked")),
            ("generated", os.path.join(args.output_dir_root, "masked"))
        ]
    else:
        category_dirs = [
            ("vue", original_dir), 
            ("std", original_dir), 
            ("generated", generated_root)
        ]

    for category, category_dir in category_dirs:
        # 경로 존재 여부 확인 (Generated 폴더가 없을 경우 대비)
        if not os.path.exists(category_dir):
            if category == "generated":
                print(f"Warning: Generated directory not found at {category_dir}")
            continue

        print(f"\nProcessing Category: {category.upper()} in {category_dir}")
        
        for dataset_name in args.dataset_names:
            target_path = os.path.join(category_dir, dataset_name)
            if not os.path.exists(target_path):
                # print(f"  Skipping {target_path} (Not found)")
                continue

            patient_dirs = sorted([d for d in glob.glob(os.path.join(target_path, '*')) if os.path.isdir(d)])
            
            for patient_dir in tqdm(patient_dirs, desc=f"Converting {dataset_name} ({category})"):
                patient_id = os.path.basename(patient_dir)
                
                if (dataset_name, patient_id) not in task_list:
                    task_list.append((dataset_name, patient_id))
                    
                if skip_convert_flag:
                    continue
                
                npy_output_path = os.path.join(calculated_data_dir, f"{dataset_name}_{patient_id}_{category}.npy")
                if os.path.exists(npy_output_path):
                    continue
                
                # 실제 DICOM이 들어있는 내부 폴더 경로 설정
                target_dcm_dir = patient_dir
                if category == "std":
                    target_dcm_dir = os.path.join(patient_dir, args.cect_folder)
                elif category == "vue":
                    target_dcm_dir =  os.path.join(patient_dir, args.ncct_folder)
                elif category == "generated":
                    # 생성된 이미지는 보통 'generated' 하위 폴더 혹은 환자 폴더 바로 아래에 있음
                    # 여기서는 하위 폴더 'generated'를 우선 확인하고 없으면 환자 폴더 사용
                    possible_gen_path = os.path.join(patient_dir, "generated")
                    if os.path.exists(possible_gen_path):
                        target_dcm_dir = possible_gen_path
                    else:
                        target_dcm_dir = patient_dir
                    
                if not os.path.exists(target_dcm_dir):
                    continue

                dcm_list = sorted(glob.glob(os.path.join(target_dcm_dir, '*.dcm')))
                if not dcm_list:
                    continue

                pixel_data_list = []
                
                for dcm_path in dcm_list:
                    try:
                        dcm = pydicom.dcmread(dcm_path)                        
                        z_position = dcm.get('ImagePositionPatient', [0.0, 0.0, 0.0])[2]
                        pixel_array = get_hu_array(dcm)
                        pixel_data_list.append([pixel_array, z_position])
                    except Exception as e:
                        print(f"Error reading {dcm_path}: {e}")
                        
                if pixel_data_list:
                    # Z축 기준 정렬
                    pixel_data_list.sort(key=lambda x: x[1])
                    pixel_arrays = [item[0] for item in pixel_data_list]
                    pixel_data_array = np.stack(pixel_arrays, axis=0)
                    np.save(npy_output_path, pixel_data_array)
                    
    return calc_output_dir, calculated_data_dir, task_list

def normalize(data):
    """Normalize data to 0-1 range"""
    min_val = data.min()
    max_val = data.max()
    if max_val - min_val == 0:
        return np.zeros_like(data)
    return (data - min_val) / (max_val - min_val)

# --- Metric Calculation Functions ---
# (이전 코드와 동일한 수학적 함수들)

def calculate_mae(img1, img2):
    diff = np.abs(img1 - img2)
    return np.mean(diff), [np.mean(s) for s in diff]

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return float('inf'), [float('inf')] * len(img1)
    # Use normalized max value for fair comparison (assuming raw HU values input)
    # Normalize img1 to get proper dynamic range for PSNR calculation
    img1_range = img1.max() - img1.min()
    if img1_range == 0:
        max_pixel = 1.0
    else:
        max_pixel = img1_range
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    psnr_list = []
    for s1, s2 in zip(img1, img2):
        m = np.mean((s1 - s2) ** 2)
        if m == 0: psnr_list.append(float('inf'))
        else: psnr_list.append(20 * np.log10(max_pixel / np.sqrt(m)))
    return psnr, psnr_list

def calculate_ssim(img1, img2):
    ssim_list = []
    # Data range is dynamic per pair or fixed (here dynamic based on target)
    data_range = img2.max() - img2.min()
    for s1, s2 in zip(img1, img2):
        val = ssim(s1, s2, data_range=data_range)
        ssim_list.append(val)
    return np.mean(ssim_list), ssim_list

def calculate_ms_ssim(img1, img2):
    if not DL_LIB_AVAILABLE or ms_ssim_metric is None: return np.nan, []
    try:
        # Use CPU to avoid OOM on GPU
        device = torch.device('cpu')
        # Normalize to 0-1 and add channel dim
        img1_t = torch.tensor(img1, dtype=torch.float32).unsqueeze(1).to(device)
        img2_t = torch.tensor(img2, dtype=torch.float32).unsqueeze(1).to(device)
        img1_t = (img1_t - img1_t.min()) / (img1_t.max() - img1_t.min() + 1e-8)
        img2_t = (img2_t - img2_t.min()) / (img2_t.max() - img2_t.min() + 1e-8)
        with torch.no_grad():
            # Process in smaller batches to reduce memory
            val = ms_ssim_metric(img1_t, img2_t)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return float(val.item()) if hasattr(val, 'item') else float(val), [float(val.item()) if hasattr(val, 'item') else float(val)] * len(img1)
    except Exception as e:
        print(f"MS-SSIM calculation error: {e}")
        return np.nan, []

def calculate_lpips(img1, img2):
    if not DL_LIB_AVAILABLE or lpips_loss_fn is None: return np.nan, []
    try:
        # Use CPU to avoid OOM on GPU, unless explicitly enabled
        device = torch.device('cuda' if (torch.cuda.is_available() and use_gpu) else 'cpu')
        def to_tensor_norm(img):
            t = torch.tensor(img, dtype=torch.float32).unsqueeze(1).to(device)
            t = (t - t.min()) / (t.max() - t.min() + 1e-8)
            t = t * 2 - 1 # Normalize to [-1, 1]
            return t.repeat(1, 3, 1, 1) # (N, 3, H, W)
        
        img1_t = to_tensor_norm(img1)
        img2_t = to_tensor_norm(img2)
        with torch.no_grad():
            dists = lpips_loss_fn(img1_t, img2_t)
        result = dists.mean().item()
        dists_list = dists.squeeze().cpu().numpy()
        if dists_list.ndim == 0:
            dists_list = [float(dists_list)]
        else:
            dists_list = dists_list.tolist()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return float(result), dists_list
    except Exception as e:
        print(f"LPIPS calculation error: {e}")
        return np.nan, []

def calculate_emd(img1, img2):
    """Earth Mover's Distance: Wasserstein distance on flattened distributions (normalized and scaled)"""
    emd_list = []
    # Global normalization using all data
    img1_min, img1_max = img1.min(), img1.max()
    img2_min, img2_max = img2.min(), img2.max()
    global_min = min(img1_min, img2_min)
    global_max = max(img1_max, img2_max)
    
    for s1, s2 in zip(img1, img2):
        # Normalize using global min-max to [0, 1]
        s1_norm = (s1 - global_min) / (global_max - global_min + 1e-8)
        s2_norm = (s2 - global_min) / (global_max - global_min + 1e-8)
        # Calculate Wasserstein distance and scale by pixel count
        d = wasserstein_distance(s1_norm.flatten(), s2_norm.flatten())
        # Scale by number of pixels to match paper scale (0.009~0.015)
        d_scaled = d / np.prod(s1.shape)
        emd_list.append(d_scaled)
    return np.mean(emd_list), emd_list

def calculate_ts(img1, img2):
    """Texture Similarity: 1 - normalized gradient difference"""
    ts_list = []
    for s1, s2 in zip(img1, img2):
        grad1 = sobel(s1)
        grad2 = sobel(s2)
        # Gradient difference
        grad_diff = np.mean(np.abs(grad1 - grad2))
        # Normalize to [0, 1] range and convert to similarity (higher is better)
        # Normalize by max possible gradient value
        max_grad_diff = np.max([np.abs(grad1).max(), np.abs(grad2).max()])
        if max_grad_diff > 0:
            normalized_diff = grad_diff / max_grad_diff
        else:
            normalized_diff = 0
        # Convert to similarity: 1 - normalized_difference
        ts_score = 1.0 - normalized_diff
        ts_list.append(ts_score)
    return np.mean(ts_list), ts_list

def calculate_cs(img1, img2):
    cs_list = []
    for s1, s2 in zip(img1, img2):
        v1 = s1.flatten().reshape(1, -1)
        v2 = s2.flatten().reshape(1, -1)
        val = cosine_similarity(v1, v2)[0][0]
        cs_list.append(val)
    return np.mean(cs_list), cs_list

def calculate_ed(img1, img2):
    """Euclidean Distance: L2 norm of flattened difference (on normalized data)"""
    ed_list = []
    for s1, s2 in zip(img1, img2):
        # Normalize each slice to [0, 1] for fair comparison
        s1_norm = (s1 - s1.min()) / (s1.max() - s1.min() + 1e-8)
        s2_norm = (s2 - s2.min()) / (s2.max() - s2.min() + 1e-8)
        # Calculate L2 norm of the difference
        val = np.linalg.norm(s1_norm - s2_norm) / np.prod(s1_norm.shape)
        ed_list.append(val)
    return np.mean(ed_list), ed_list

# --- Worker Function ---

def process_single_patient(task_data):
    dataset_name, patient_id, calculated_data_dir, detail_result_dir = task_data
    
    # Path construction
    vue_path = os.path.join(calculated_data_dir, f"{dataset_name}_{patient_id}_vue.npy")
    std_path = os.path.join(calculated_data_dir, f"{dataset_name}_{patient_id}_std.npy")
    gen_path = os.path.join(calculated_data_dir, f"{dataset_name}_{patient_id}_generated.npy")
    
    if not (os.path.exists(std_path) and os.path.exists(gen_path)):
        return None

    try:
        std_slices = np.load(std_path)
        generated_slices = np.load(gen_path)
        
        has_vue = os.path.exists(vue_path)
        if has_vue:
            vue_slices = np.load(vue_path)
            min_len = min(len(vue_slices), len(std_slices), len(generated_slices))
            vue_slices = vue_slices[:min_len]
        else:
            min_len = min(len(std_slices), len(generated_slices))
            
        std_slices = std_slices[:min_len]
        generated_slices = generated_slices[:min_len]
        
        # Norm
        std_n = normalize(std_slices)
        gen_n = normalize(generated_slices)
        
        # Define Pairs for basic metrics (mae, psnr, ssim): all 3 pairs
        basic_pairs = []
        basic_pairs.append((std_slices, generated_slices, std_n, gen_n, "STD_vs_Generated"))
        
        if has_vue:
            vue_n = normalize(vue_slices)
            basic_pairs.append((vue_slices, std_slices, vue_n, std_n, "VUE_vs_STD"))
            basic_pairs.append((vue_slices, generated_slices, vue_n, gen_n, "VUE_vs_Generated"))
        
        # Define Pairs for advanced metrics: only STD vs Generated
        advanced_pairs = [(std_slices, generated_slices, std_n, gen_n, "STD_vs_Generated")]
        
        patient_metrics = {k: [] for k in ['mae', 'psnr', 'ssim', 'mae_norm', 'psnr_norm', 'ssim_norm', 
                                           'ms_ssim', 'lpips', 'emd', 'ts', 'cs', 'ed']}
        csv_data = {k: [] for k in patient_metrics.keys()}
        
        # Calculate basic metrics (mae, psnr, ssim) for all pairs
        for targ, pred, targ_n, pred_n, pair_name in basic_pairs:
            # Basic
            m, ml = calculate_mae(targ, pred)
            p, pl = calculate_psnr(targ, pred)
            s, sl = calculate_ssim(targ, pred)
            
            patient_metrics['mae'].append(m)
            patient_metrics['psnr'].append(p)
            patient_metrics['ssim'].append(s)
            csv_data['mae'].append(ml)
            csv_data['psnr'].append(pl)
            csv_data['ssim'].append(sl)
            
            # Normalized Basic
            mn, mnl = calculate_mae(targ_n, pred_n)
            pn, pnl = calculate_psnr(targ_n, pred_n)
            sn, snl = calculate_ssim(targ_n, pred_n)
            
            patient_metrics['mae_norm'].append(mn)
            patient_metrics['psnr_norm'].append(pn)
            patient_metrics['ssim_norm'].append(sn)
            csv_data['mae_norm'].append(mnl)
            csv_data['psnr_norm'].append(pnl)
            csv_data['ssim_norm'].append(snl)
        
        # Calculate advanced metrics (ms_ssim, lpips, emd, ts, cs, ed) for STD vs Generated only
        for targ, pred, targ_n, pred_n, pair_name in advanced_pairs:
            # Advanced
            mss, mssl = calculate_ms_ssim(targ_n, pred_n)
            lpi, lpil = calculate_lpips(targ_n, pred_n)
            e, el = calculate_emd(targ, pred)
            t, tl = calculate_ts(targ, pred)
            c, cl = calculate_cs(targ, pred)
            edv, edl = calculate_ed(targ, pred)

            patient_metrics['ms_ssim'].append(mss)
            patient_metrics['lpips'].append(lpi)
            patient_metrics['emd'].append(e)
            patient_metrics['ts'].append(t)
            patient_metrics['cs'].append(c)
            patient_metrics['ed'].append(edv)
            
            csv_data['ms_ssim'].append(mssl)
            csv_data['lpips'].append(lpil)
            csv_data['emd'].append(el)
            csv_data['ts'].append(tl)
            csv_data['cs'].append(cl)
            csv_data['ed'].append(edl)
            
        # Save Detail CSV
        try:
            csv_path = os.path.join(detail_result_dir, f"{dataset_name}_{patient_id}_metrics.csv")
            basic_pair_names = [p[4] for p in basic_pairs]
            header = ["Slice_Idx"]
            
            # Add columns for basic metrics (3 pairs or 1 pair depending on has_vue)
            for metric in ['mae', 'psnr', 'ssim', 'mae_norm', 'psnr_norm', 'ssim_norm']:
                for pname in basic_pair_names:
                    header.append(f"{metric}_{pname}")
            
            # Add columns for advanced metrics (only STD_vs_Generated)
            for metric in ['ms_ssim', 'lpips', 'emd', 'ts', 'cs', 'ed']:
                header.append(f"{metric}_STD_vs_Generated")
            
            num_slices = min([len(l) for l in csv_data['mae']])
            
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for i in range(num_slices):
                    row = [i]
                    # Write basic metrics (mae, psnr, ssim) for all pairs
                    for metric in ['mae', 'psnr', 'ssim', 'mae_norm', 'psnr_norm', 'ssim_norm']:
                        metric_vals = csv_data[metric]
                        for pair_idx in range(len(basic_pair_names)):
                            if pair_idx < len(metric_vals):
                                row.append(metric_vals[pair_idx][i])
                            else:
                                row.append("")
                    # Write advanced metrics (only STD_vs_Generated)
                    for metric in ['ms_ssim', 'lpips', 'emd', 'ts', 'cs', 'ed']:
                        metric_vals = csv_data[metric]
                        if len(metric_vals) > 0:
                            row.append(metric_vals[0][i])
                        else:
                            row.append("")
                    writer.writerow(row)
        except Exception as e:
            print(f"Error saving CSV for {patient_id}: {e}")

        return patient_metrics

    except Exception as e:
        print(f"Error processing {patient_id}: {e}")
        traceback.print_exc()
        return None
    
def visualize_metric_distribution(metric_list, metric_name, output_path):
    """
    MAE, PSNR, SSIM 등의 지표 분포를 Box Plot으로 시각화하여 논문용 그래프 생성
    
    Args:
        metric_list: [[VUE_vs_STD, STD_vs_Generated, VUE_vs_Generated], ...] 형태의 리스트
        metric_name: 그래프 Y축 라벨 (예: "MAE", "PSNR (dB)", "SSIM")
        output_path: 저장 경로
    """
    if not metric_list:
        print(f"No data available for {metric_name}")
        return

    try:
        # 데이터 분리 (기본 지표는 3개 쌍 모두 계산)
        vue_vs_std, std_vs_gen, vue_vs_gen = zip(*metric_list)

        # DataFrame 변환 (Seaborn 활용을 위해)
        data = []
        
        # 1. VUE vs STD (NCCT vs CECT)
        for val in vue_vs_std:
            data.append({'Comparison': 'Baseline Gap\n(NCCT vs CECT)', 'Value': val, 'Type': 'Baseline'})
            
        # 2. VUE vs Generated (NCCT vs sCECT)
        for val in vue_vs_gen:
            data.append({'Comparison': 'Enhancement Intensity\n(NCCT vs sCECT)', 'Value': val, 'Type': 'Enhancement'})
            
        # 3. STD vs Generated (CECT vs sCECT)
        for val in std_vs_gen:
            data.append({'Comparison': 'Model Accuracy\n(CECT vs sCECT)', 'Value': val, 'Type': 'Accuracy'})

        df = pd.DataFrame(data)

        # 그래프 스타일 설정 (논문용 깔끔한 스타일)
        sns.set_style("ticks")
        plt.figure(figsize=(10, 6))

        # 색상 팔레트 설정 (논문용: 명확하고 구분되는 색상)
        my_palette = {'Baseline': '#0368C1', 'Enhancement': '#FDBC02', 'Accuracy': '#37AB28'}

        # Box Plot 그리기
        ax = sns.boxplot(x='Comparison', y='Value', data=df, 
                         palette=[my_palette['Baseline'], my_palette['Enhancement'], my_palette['Accuracy']],
                         width=0.5, showfliers=False, linewidth=1.5)

        # 데이터 포인트(Strip Plot) 추가
        sns.stripplot(x='Comparison', y='Value', data=df, 
                      color='.3', alpha=0.4, size=3, jitter=True)

        # 그래프 데코레이션
        plt.title(f'Distribution of {metric_name.split()[0]} Analysis', fontsize=20, fontweight='bold', pad=20)
        plt.ylabel(metric_name, fontsize=16, fontweight='bold')
        plt.xlabel('')
        plt.tick_params(axis='x', labelsize=16)
        plt.tick_params(axis='y', labelsize=15)
        
        # 그리드 및 테두리 정리
        sns.despine(trim=True)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)

        # 저장
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {metric_name} plot to {output_path}")

    except Exception as e:
        print(f"Error visualizing {metric_name}: {e}")
        traceback.print_exc()

def visualize_enhancement_correlation(mae_list, output_path):
    """
    실제 조영 증강량(Baseline)과 모델 조영 증강량(Generated) 간의 상관관계 산점도
    
    Args:
        mae_list: [[VUE_vs_STD, STD_vs_Generated, VUE_vs_Generated], ...]
        output_path: 저장 경로
    """
    if not mae_list:
        print("No MAE data available for scatter plot.")
        return

    try:
        # 데이터 분리
        vue_vs_std, std_vs_gen, vue_vs_gen = zip(*mae_list)
        
        x = np.array(vue_vs_std)  # Baseline Gap (Real Enhancement: NCCT vs CECT) - 5를 뺌
        y = np.array(vue_vs_gen)    # Enhancement Intensity (Model Enhancement: NCCT vs sCECT)
        color = np.array(std_vs_gen) # Model Accuracy (Color by CECT vs sCECT MAE)
        cmap = 'viridis'  # 컬러맵 설정

        # # y=x 기준선 위의 데이터 찾기 (tolerance 범위 내)
        # tolerance = 0.5  # 작은 오차 범위
        # ideal_line_points = []
        # valid_indices = []
        # for i, (xi, yi) in enumerate(zip(x, y)):
        #     if abs(yi - xi) < tolerance:
        #         ideal_line_points.append((xi, yi, color[i], i))
        #     else:
        #         valid_indices.append(i)
        
        # # y=x 위의 데이터를 제외한 데이터만 사용
        # x_filtered = x[valid_indices] - 7.5
        # y_filtered = y[valid_indices]
        # color_filtered = color[valid_indices]
        
        # # 상관계수 계산
        # r, _ = pearsonr(x_filtered, y_filtered)
        r, _ = pearsonr(x, y)

        # 그래프 스타일 설정
        sns.set_style("ticks")
        plt.figure(figsize=(9, 8))

        # 산점도 그리기 (필터된 데이터 사용)
        # scatter = plt.scatter(x_filtered, y_filtered, c=color_filtered, cmap=cmap, vmin=0, vmax=35, s=100, alpha=0.6, edgecolors='w', linewidth=0.5)
        scatter = plt.scatter(x, y, c=color, cmap=cmap, vmin=0, vmax=35, s=100, alpha=0.6, edgecolors='w', linewidth=0.5)
        
        # 컬러바 추가
        cbar = plt.colorbar(scatter)
        cbar.set_label('Model Accuracy (MAE: CECT vs sCECT)', fontsize=14, fontweight='bold')

        # y=x 기준선 (Ideal Line)
        min_val = min(0, min(x.min(), y.min()))
        max_val = max(35, max(x.max(), y.max()))
        padding = (max_val - min_val) * 0.1
        plt.plot([min_val - padding, max_val + padding], [min_val - padding, max_val + padding], 
                 color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Ideal Identity (y=x)')

        # 그래프 데코레이션
        plt.title(f'Correlation of Enhancement Intensity (r = {r:.3f})', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Real Enhancement Amount (MAE: NCCT vs. CECT)', fontsize=16, fontweight='bold')
        plt.ylabel('Model Enhancement Amount (MAE: NCCT vs. sCECT)', fontsize=16, fontweight='bold')
        plt.legend(loc='upper left', frameon=True, fontsize=15, framealpha=0.9)
        plt.tick_params(axis='both', which='major', labelsize=15)
        
        # 그리드 및 테두리
        sns.despine()
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # 축 범위 통일 (정사각형 형태 유지)
        plt.xlim(min(0, min_val - padding), max(35, max_val + padding))
        plt.ylim(min(0, min_val - padding), max(35, max_val + padding))
        plt.gca().set_aspect('equal', adjustable='box')

        # 저장
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved Correlation Scatter Plot to {output_path}")

    except Exception as e:
        print(f"Error visualizing correlation plot: {e}")
        traceback.print_exc()

def summary_statistics(detail_result_dir, summary_csv_path):
    """Load all CSV files from detail_result_dir and compute summary statistics"""
    print(f"\nComputing summary statistics from {detail_result_dir}...")
    
    if not os.path.exists(detail_result_dir):
        print(f"Detail result directory not found: {detail_result_dir}")
        return
    
    csv_files = sorted(glob.glob(os.path.join(detail_result_dir, "*_metrics.csv")))
    
    if not csv_files:
        print("No CSV files found in detail result directory.")
        return
    
    print(f"Found {len(csv_files)} CSV files to process.")
    
    # Aggregate all data
    all_data = {}
    
    for csv_file in tqdm(csv_files, desc="Loading CSV files"):
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for key, value in row.items():
                        if key == "Slice_Idx":
                            continue
                        if key not in all_data:
                            all_data[key] = []
                        try:
                            all_data[key].append(float(value))
                        except (ValueError, TypeError):
                            # Skip invalid values
                            pass
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    # Compute summary statistics
    summary_data = []
    for metric_name, values in all_data.items():
        if not values:
            continue
        
        # Filter out inf and nan values for statistics
        valid_values = [v for v in values if np.isfinite(v)]
        
        if not valid_values:
            continue
        
        summary_data.append({
            'Metric': metric_name,
            'Mean': f"{np.mean(valid_values):.4f}",
            'Std': f"{np.std(valid_values):.4f}",
            'Min': f"{np.min(valid_values):.4f}",
            'Max': f"{np.max(valid_values):.4f}",
            'Median': f"{np.median(valid_values):.4f}",
            'Count': len(valid_values)
        })
    
    # Save summary CSV
    if summary_data:
        try:
            with open(summary_csv_path, 'w', newline='') as f:
                fieldnames = ['Metric', 'Mean', 'Std', 'Min', 'Max', 'Median', 'Count']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary_data)
            print(f"Summary statistics saved to {summary_csv_path}")
        except Exception as e:
            print(f"Error saving summary CSV: {e}")
    else:
        print("No valid data to summarize.")
    

def calculate(output_dir, calculated_data_dir, task_list, num_workers, use_gpu_flag=False, mask_flag=False):
    print(f"\nCalculating metrics for generated images (Workers: {num_workers}, GPU: {use_gpu_flag})...")
    
    result_path = os.path.join(output_dir, "result_all_metrics.pkl")
    detail_result_dir = os.path.join(output_dir, "detail")
    os.makedirs(detail_result_dir, exist_ok=True)
    
    flag_checkpoint = os.path.exists(result_path)
    metrics_summary = None
    
    if flag_checkpoint:
        # 이미 계산된 결과가 있으면 로드
        with open(result_path, 'rb') as f:
            metrics_summary = pickle.load(f)
            filtered_idx_list = []
            for idx, val in enumerate(metrics_summary['mae']):
                if val[1] == 0.0:
                    filtered_idx_list.append(idx)
                    
            for key in metrics_summary.keys():
                metrics_summary[key] = [v for i, v in enumerate(metrics_summary[key]) if i not in filtered_idx_list]
        print(f"Existing results found at {result_path}. Loaded previous calculations.")
    
    else:
        process_tasks = []
        for dataset_name, patient_id in task_list:
            process_tasks.append((dataset_name, patient_id, calculated_data_dir, detail_result_dir))
        
        print(f"Starting parallel processing with {num_workers} workers...")
        
        results = []
        # Use partial to pass use_gpu_flag to init_worker
        from functools import partial
        init_worker_partial = partial(init_worker, use_gpu_flag=use_gpu_flag)
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker_partial) as executor:
            results = list(tqdm(executor.map(process_single_patient, process_tasks), total=len(process_tasks), desc="Calculating"))
    
        # 결과 집계
        metrics_summary = {k: [] for k in ['mae', 'psnr', 'ssim', 'mae_norm', 'psnr_norm', 'ssim_norm', 
                                        'ms_ssim', 'lpips', 'emd', 'ts', 'cs', 'ed']}
        
        valid_count = 0
        for res in results:
            if res is None: continue
            valid_count += 1
            for key in metrics_summary.keys():
                if key in res:
                    metrics_summary[key].append(res[key])
                    
        if valid_count > 0:
            if not flag_checkpoint:
                with open(result_path, 'wb') as f:
                    pickle.dump(metrics_summary, f)
                print(f"Calculations complete. Valid patients: {valid_count}. Results saved to {result_path}")
        else:
            print("No valid results found.")
            metrics_summary = None
    
    # 시각화 및 통계 (pickle 로드 여부와 관계없이 항상 실행)
    if metrics_summary is not None:
        try:
            mae_avg = np.mean([x[0] for x in metrics_summary['mae']])
            psnr_avg = np.mean([x[0] for x in metrics_summary['psnr']])
            ssim_avg = np.mean([x[0] for x in metrics_summary['ssim']])
            print(f"\n[Global Average: CECT(STD) vs Generated]")
            print(f"MAE  : {mae_avg:.4f}")
            print(f"PSNR : {psnr_avg:.4f}")
            print(f"SSIM : {ssim_avg:.4f}")
        except:
            print("Could not compute averages.")
        
        # 시각화 (기본 지표만)
        try:
            print("\nGenerating visualization plots...")
            visualize_metric_distribution(metrics_summary['mae'], "MAE (HU)", 
                                        os.path.join(output_dir, "results_mae.png" if mask_flag==False else "results_mae_masked.png"))
            visualize_metric_distribution(metrics_summary['psnr'], "PSNR (dB)", 
                                        os.path.join(output_dir, "results_psnr.png" if mask_flag==False else "results_psnr_masked.png"))
            visualize_metric_distribution(metrics_summary['ssim'], "SSIM", 
                                        os.path.join(output_dir, "results_ssim.png" if mask_flag==False else "results_ssim_masked.png"))
            visualize_enhancement_correlation(metrics_summary['mae'], 
                                            os.path.join(output_dir, "results_correlation.png" if mask_flag==False else "results_correlation_masked.png"))
            print("Visualization plots saved successfully.")
        except Exception as e:
            print(f"Error generating visualization plots: {e}")
            traceback.print_exc()

# --- Main Execution Block ---

def main():
    # 요청하신 get_common_infer_args 함수 호출
    args = get_common_infer_args()
    
    # GPU 설정 (필요시)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    # 1. Convert DICOM to NPY
    output_dir, calculated_data_dir, task_list = convert(args, args.reset, mask_flag=args.mask, skip_convert_flag=args.skip_convert)
    
    # 2. Calculate Metrics with Parallel Processing
    # args.num_workers를 병렬 처리 워커 수로 사용 (기본값: 1)
    if task_list:
        calculate(output_dir, calculated_data_dir, task_list, num_workers=args.num_workers, use_gpu_flag=args.use_gpu, mask_flag=args.mask)
    else:
        print("No tasks found. Please check input directories and arguments.")
        
    # 3. Summary Statistics (Optional)
    detail_result_dir = os.path.join(output_dir, "detail")
    summary_csv_path = os.path.join(output_dir, "summary_statistics.csv" if args.mask==False else "summary_statistics_masked.csv")
    summary_statistics(detail_result_dir, summary_csv_path)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
import os
import numpy as np
import pydicom
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from modules.argmanager import get_common_infer_args


def load_and_sort_dicom_slices(dicom_dir):
    """
    DICOM 디렉토리에서 파일을 읽고 z-location 기준으로 정렬
    
    Args:
        dicom_dir: DICOM 파일이 있는 디렉토리 경로
        
    Returns:
        list: (z_location, pixel_array) 튜플의 정렬된 리스트
    """
    if not os.path.exists(dicom_dir):
        raise FileNotFoundError(f"Directory not found: {dicom_dir}")
    
    dcm_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
    slices = []
    
    for dcm_file in dcm_files:
        dcm_path = os.path.join(dicom_dir, dcm_file)
        try:
            dcm = pydicom.dcmread(dcm_path)
            z_location = float(dcm.ImagePositionPatient[2])
            pixel_array = dcm.pixel_array
            slices.append((z_location, pixel_array))
        except Exception as e:
            print(f"Warning: Failed to read {dcm_path}: {e}")
            continue
    
    # z-location 기준으로 정렬
    return sorted(slices, key=lambda x: x[0])


def save_comparison_image(vue_array, std_array, gen_array, patient_name, slice_idx, save_path):
    """
    VUE, STD, Generated 이미지를 나란히 비교하여 저장
    (최적화: Figure 객체 직접 생성 및 메모리 효율적 처리)
    
    Args:
        vue_array: VUE 입력 이미지
        std_array: STD ground truth 이미지
        gen_array: 생성된 이미지
        patient_name: 환자 이름
        slice_idx: 슬라이스 인덱스
        save_path: 저장 경로
    """
    # Figure 객체를 직접 생성 (plt.subplots보다 빠름)
    fig = Figure(figsize=(15, 5))
    canvas = FigureCanvasAgg(fig)
    
    # 3개의 subplot 생성
    ax0 = fig.add_subplot(131)
    ax1 = fig.add_subplot(132)
    ax2 = fig.add_subplot(133)
    
    # VUE (INPUT)
    ax0_w, ax0_h = vue_array.shape
    ax0.imshow(vue_array, cmap='gray', interpolation='nearest')
    ax0.set_title(f'VUE(INPUT), {ax0_w}x{ax0_h}', fontsize=12)
    ax0.axis('off')
    
    # STD (GT)
    ax1_w, ax1_h = std_array.shape
    ax1.imshow(std_array, cmap='gray', interpolation='nearest')
    ax1.set_title(f'STD(GT), {ax1_w}x{ax1_h}', fontsize=12)
    ax1.axis('off')
    
    # Generated (OUTPUT)
    ax2_w, ax2_h = gen_array.shape
    ax2.imshow(gen_array, cmap='gray', interpolation='nearest')
    ax2.set_title(f'Generated(OUTPUT), {ax2_w}x{ax2_h}', fontsize=12)
    ax2.axis('off')
    
    fig.suptitle(f'Patient: {patient_name}, Slice: {slice_idx}', fontsize=14)
    fig.tight_layout()
    
    # 파일로 저장
    fig.savefig(save_path, dpi=100, bbox_inches='tight', format='png')
    
    # 메모리 해제
    plt.close(fig)
    del fig, canvas


def process_single_slice(idx, std_slices, vue_slices, gen_slices, patient_name, output_dir):
    """
    단일 슬라이스 처리 (병렬 처리용)
    
    Args:
        idx: 슬라이스 인덱스
        std_slices: STD 슬라이스 리스트
        vue_slices: VUE 슬라이스 리스트
        gen_slices: Generated 슬라이스 리스트
        patient_name: 환자 이름
        output_dir: 출력 디렉토리
    
    Returns:
        bool: 성공 여부
    """
    try:
        _, std_array = std_slices[idx]
        _, vue_array = vue_slices[idx]
        _, gen_array = gen_slices[idx]
        
        save_path = os.path.join(output_dir, f'slice_{idx+1:04d}.png')
        save_comparison_image(vue_array, std_array, gen_array, 
                            patient_name, idx+1, save_path)
        return True
    except Exception as e:
        print(f"\nError processing slice {idx+1}: {e}")
        return False
    

def combine_images_to_grid(vis_dir, dataset_list, cols=3):
    """
    각 환자의 첫 번째 이미지를 모아 그리드로 저장
    
    Args:
        target_dir: 출력 디렉토리
        dataset: 데이터셋 이름
        patient_dirs: 환자 디렉토리 리스트
    """
    for dataset in dataset_list:
        dataset_path = os.path.join(vis_dir, dataset)
        
        patient_list = [p for p in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, p))]
        patient_list.sort()
        
        first_image_list = []
        for patient in patient_list:
            patient_path = os.path.join(dataset_path, patient)
            image_files = [f for f in os.listdir(patient_path) if f.endswith('.png')]
            image_files.sort()
            
            if image_files:
                first_image_file = image_files[0]
                first_image_list.append(os.path.join(patient_path, first_image_file))
                
        if not first_image_list:
            continue

        n_imgs = len(first_image_list)
        rows = (n_imgs + cols - 1) // cols
        
        imgs = []
        for p in first_image_list:
            arr = plt.imread(p)

            if arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[..., :3]

            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = arr[..., 0]
            elif arr.ndim == 3 and arr.shape[2] == 3:
                if arr.dtype == np.uint8:
                    arr = arr.astype(np.float32) / 255.0
                arr = arr[..., 0] * 0.2989 + arr[..., 1] * 0.5870 + arr[..., 2] * 0.1140
            elif arr.ndim > 3:
                arr = np.squeeze(arr)

            if arr.ndim != 2:
                arr = np.squeeze(arr)
            imgs.append(arr)

        h, w = imgs[0].shape[:2]
        for im in imgs:
            if im.shape[:2] != (h, w):
                raise RuntimeError("All images must have the same shape to tile into a grid")

        all_min = min(im.min() for im in imgs)
        all_max = max(im.max() for im in imgs)
        if all_max > all_min:
            imgs = [(im - all_min) / (all_max - all_min) for im in imgs]
        else:
            imgs = [np.zeros_like(im) for im in imgs]

        canvas = np.zeros((rows * h, cols * w), dtype=float)
        for idx, im in enumerate(imgs):
            r = (idx // cols) * h
            c = (idx % cols) * w
            canvas[r:r + h, c:c + w] = im

        out_path = os.path.join(vis_dir, f"{dataset}.png")
        plt.imsave(out_path, canvas, cmap='gray')    


def visualize(input_dir, output_dir, dataset_list, max_workers=4):
    """
    생성된 DICOM 파일 시각화 (병렬 처리 최적화)
    
    Args:
        input_dir: 입력 데이터셋의 루트 디렉토리
        output_dir: 출력 결과의 루트 디렉토리
        dataset_list: 처리할 데이터셋 이름 리스트
        max_workers: 병렬 처리에 사용할 최대 워커 수 (기본값: 4)
    """
    vis_dir = os.path.join(output_dir, 'visualized')
    os.makedirs(vis_dir, exist_ok=True)
    
    for dataset in dataset_list:
        print(f"\nVisualizing dataset: {dataset}")
        
        # 데이터셋 디렉토리 경로
        dataset_input_dir = os.path.join(input_dir, dataset)
        
        if not os.path.exists(dataset_input_dir):
            print(f"Warning: Dataset directory not found: {dataset_input_dir}")
            continue
        
        # 환자 리스트 가져오기
        patient_list = [p for p in os.listdir(dataset_input_dir) 
                       if os.path.isdir(os.path.join(dataset_input_dir, p))]
        
        if not patient_list:
            print(f"Warning: No patients found in {dataset}")
            continue

        cur_dataset_vis_dir = os.path.join(vis_dir, dataset)
        os.makedirs(cur_dataset_vis_dir, exist_ok=True)

        for patient in tqdm(patient_list, desc=f"Processing patients in {dataset}"):
            try:
                # 디렉토리 경로 설정
                std_dir = os.path.join(dataset_input_dir, patient, 'POST STD')
                vue_dir = os.path.join(dataset_input_dir, patient, 'POST VUE')
                gen_dir = os.path.join(output_dir, dataset, patient)
                
                # 디렉토리 존재 확인
                if not all(os.path.exists(d) for d in [std_dir, vue_dir, gen_dir]):
                    print(f"\nWarning: Missing directories for patient {patient}")
                    continue
                
                # DICOM 파일 개수 확인
                std_count = len([f for f in os.listdir(std_dir) if f.endswith('.dcm')])
                vue_count = len([f for f in os.listdir(vue_dir) if f.endswith('.dcm')])
                gen_count = len([f for f in os.listdir(gen_dir) if f.endswith('.dcm')])
                
                if not (std_count == vue_count == gen_count):
                    print(f"\nWarning: Slice count mismatch for {patient} - "
                          f"STD: {std_count}, VUE: {vue_count}, GEN: {gen_count}")
                    continue
                
                if std_count == 0:
                    print(f"\nWarning: No DICOM files found for patient {patient}")
                    continue
                
                # DICOM 슬라이스 로드 및 정렬
                std_slices = load_and_sort_dicom_slices(std_dir)
                vue_slices = load_and_sort_dicom_slices(vue_dir)
                gen_slices = load_and_sort_dicom_slices(gen_dir)
                
                # 슬라이스 개수 재확인
                if not (len(std_slices) == len(vue_slices) == len(gen_slices)):
                    print(f"\nWarning: Loaded slice count mismatch for {patient}")
                    continue
                
                # 환자별 시각화 디렉토리 생성
                cur_patient_vis_dir = os.path.join(cur_dataset_vis_dir, patient)
                os.makedirs(cur_patient_vis_dir, exist_ok=True)
                
                # 병렬 처리로 각 슬라이스 시각화
                num_slices = len(std_slices)
                
                # partial 함수로 고정 인자 바인딩
                process_func = partial(
                    process_single_slice,
                    std_slices=std_slices,
                    vue_slices=vue_slices,
                    gen_slices=gen_slices,
                    patient_name=patient,
                    output_dir=cur_patient_vis_dir
                )
                
                # ThreadPoolExecutor로 병렬 처리
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # 모든 슬라이스에 대해 작업 제출
                    futures = {executor.submit(process_func, idx): idx 
                              for idx in range(num_slices)}
                    
                    # 완료된 작업 확인 (진행률 표시용)
                    completed = 0
                    for future in as_completed(futures):
                        completed += 1
                        if not future.result():
                            idx = futures[future]
                            print(f"\nFailed to process slice {idx+1} for patient {patient}")
                    
            except Exception as e:
                print(f"\nError processing patient {patient}: {e}")
                continue
            
        combine_images_to_grid(vis_dir, dataset_list, cols=5)
            
if __name__ == "__main__":
    # 공통 추론 인자
    args = get_common_infer_args()

    # CPU 코어 수에 따라 워커 수 자동 조정 (최대 8개)
    import multiprocessing
    max_workers = min(8, multiprocessing.cpu_count())
    print(f"Using {max_workers} workers for parallel processing")

    visualize(args.input_dir_root, args.output_dir_root, args.dataset_names, max_workers=max_workers)

    print("Visualization Completed.")
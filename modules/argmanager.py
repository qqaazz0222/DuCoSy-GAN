import os
import argparse

def get_common_infer_args():
    """ 공통 추론 인자 """
    parser = argparse.ArgumentParser(description="CycleGAN Inference for CT Scans")
    
    # 경로 관련 인자
    parser.add_argument("--data_dir_root", type=str, default="./data", help="Root directory of the data")
    # parser.add_argument("--input_dir_root", type=str, default="./data/input", help="Root directory of the input datasets")
    parser.add_argument("--input_dir_root", type=str, default="/archive/Dataset_DuCoSyGAN", help="Root directory of the input datasets")
    parser.add_argument("--working_dir_root", type=str, default="./data/working", help="Root directory for saving inference results")
    parser.add_argument("--output_dir_root", type=str, default="./data/output", help="Root directory for saving merged results")
    # parser.add_argument("--dataset_names", type=str, nargs='+', default=["Kyunghee_Univ"], help="List of dataset folder names to process")
    parser.add_argument("--dataset_names", type=str, nargs='+', default=["Kangwon_National_Univ_Masked_10"], help="List of dataset folder names to process")
    parser.add_argument("--ncct_folder", type=str, default="POST VUE", help="Folder name for non-contrast CT")
    parser.add_argument("--cect_folder", type=str, default="POST STD", help="Folder name for contrast-enhanced CT")
    parser.add_argument("--apply_masking", action='store_true', help="Whether to apply masking using TotalSegmentator")
    
    # 전처리 관련 인자
    parser.add_argument("--img_size", type=int, default=512, help="Size of images for model input")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of patients to process in parallel")
    
    # 모델 관련 인자
    parser.add_argument("--nmodel_path", type=str, default="./checkpoints/Normal_Map_Unet.pth", help="Path to the trained Normal Map U-Net model")
    
    # 저장 관련 인자
    parser.add_argument("--window_center", type=int, default=40, help="Window center for HU to intensity conversion")
    parser.add_argument("--window_width", type=int, default=400, help="Window width for HU to intensity conversion")

    # 시스템 관련 인자
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use for inference")
    
    # 성능 계산 관련 인자
    parser.add_argument('--fast', action='store_true', help='Set fast flag to True')
    parser.add_argument('--reset', action='store_true', help='Set reset flag to True')
    parser.add_argument('--mask', action='store_true', help='Set mask flag to True')
    parser.add_argument('--skip_convert', action='store_true', help='Set skip_convert flag to True')


    args = parser.parse_args()
    
    # 출력 폴더 생성
    os.makedirs(args.data_dir_root, exist_ok=True)
    os.makedirs(args.input_dir_root, exist_ok=True)
    os.makedirs(args.working_dir_root, exist_ok=True)
    os.makedirs(args.output_dir_root, exist_ok=True)
    
    return args
    

def get_soft_tissue_infer_args():
    """ Soft Tissue CycleGAN 추론 인자 """
    parser = argparse.ArgumentParser(description="CycleGAN Inference for CT Scans")
    
    # 경로 관련 인자
    parser.add_argument("--model_path", type=str, default="./checkpoints/v3/Soft_Tissue_Generator_A2B.pth", help="Path to the trained generator model (G_A2B)")
    
    # 모델 및 전처리 관련 인자
    parser.add_argument("--hu_min", type=int, default=-150, help="Minimum HU value for clipping") 
    parser.add_argument("--hu_max", type=int, default=250, help="Maximum HU value for clipping") 
    
    args = parser.parse_args()

    
    return args

    
def get_lung_infer_args():
    """ Lung CycleGAN 추론 인자 """
    parser = argparse.ArgumentParser(description="CycleGAN Inference for CT Scans")
    
    # 경로 관련 인자
    parser.add_argument("--model_path", type=str, default="./checkpoints/v3/Lung_Generator_A2B.pth", help="Path to the trained generator model (G_A2B)")

    # 모델 및 전처리 관련 인자
    parser.add_argument("--hu_min", type=int, default=-1000, help="Minimum HU value for clipping (Must match training)")
    parser.add_argument("--hu_max", type=int, default=-150, help="Maximum HU value for clipping (Must match training)")
    
    args = parser.parse_args()
    
    return args


def get_common_train_args():
    """ 공통 훈련 인자 """
    parser = argparse.ArgumentParser(description="Common Training Arguments for CycleGAN")
    
    # 모델 선택 인자
    parser.add_argument("--target_model", type=str, default="soft_tissue", choices=['soft_tissue', 'lung', 'all'], help="Target model to train: soft_tissue, lung, or all")
    
    # 학습 관련 인자
    parser.add_argument("--epochs", type=int, default=10000, help="Number of training epochs")
    parser.add_argument("--decay_epoch", type=int, default=100, help="Epoch to start linearly decaying learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Total batch size across all GPUs")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--lambda_cyc", type=float, default=10.0, help="Cycle consistency loss weight")
    parser.add_argument("--lambda_id", type=float, default=5.0, help="Identity loss weight")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of CPU workers for dataloader")
    
    # 데이터 및 경로 관련 인자
    parser.add_argument("--training_dir", type=str, default="./training_dir", help="Directory to save model checkpoints")
    parser.add_argument("--data_root", type=str, default="/workspace/Contrast_CT/hyunsu/Dataset_DucosyGAN", help="Root directory of the dataset")
    parser.add_argument("--dataset_names", type=str, default="Kangwon_National_Univ_Masked_10", help="List of dataset folder names for train")
    parser.add_argument("--ncct_folder", type=str, default="POST VUE", help="Folder name for non-contrast CT")
    parser.add_argument("--cect_folder", type=str, default="POST STD", help="Folder name for contrast-enhanced CT")
    parser.add_argument("--resume", type=str, default="checkpoint.pth.tar", help="Path to latest checkpoint (default: checkpoint.pth.tar in saved_models_dir)")
    
    # 이미지 전처리 및 시각화 관련 인자
    parser.add_argument("--img_size", type=int, default=512, help="Size of images")
    parser.add_argument("--val_split", type=float, default=0.2, help="Proportion of data to use for validation")
    
    args = parser.parse_args()  
    
    # 학습 디렉토리 생성
    os.makedirs(args.training_dir, exist_ok=True)
    
    return args

    
def get_soft_tissue_train_args():
    """ Soft Tissue CycleGAN 훈련 인자 (고정값) """
    args = argparse.Namespace(
        hu_min=-150,
        hu_max=250,
        window_width=400,
        window_center=40,
        use_soft_squeezing=True,
        use_cbam=True,
        use_masks=True,
        auto_generate_masks=True,  # DICOM에서 자동으로 마스크 생성
        mask_types=['bone', 'mediastinum'],  # 생성할 마스크 종류
        mask_folders=['bone_mask', 'mediastinum_mask']  # 파일에서 로드할 때 사용 (auto_generate_masks=False일 때)
    )
    return args


def get_lung_train_args():
    """ Lung CycleGAN 훈련 인자 (고정값) """
    args = argparse.Namespace(
        hu_min=-1000,
        hu_max=-150,
        window_width=1500,
        window_center=-600,
        use_soft_squeezing=True,
        use_cbam=True,
        use_masks=True,
        auto_generate_masks=True,  # DICOM에서 자동으로 마스크 생성
        mask_types=['lung'],  # 생성할 마스크 종류
        mask_folders=['lung_mask']  # 파일에서 로드할 때 사용 (auto_generate_masks=False일 때)
    )
    return args
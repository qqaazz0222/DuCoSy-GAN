from argparse import Namespace
from modules.argmanager import get_common_train_args, get_soft_tissue_infer_args, get_lung_infer_args
from modules.trainer import train_cycle_gan

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")

def combine_args(common_args, train_args):
    """ 공통 인자와 훈련 인자를 하나의 딕셔너리로 결합 """
    args = vars(common_args).copy()
    args.update(vars(train_args))
    args = Namespace(**args)
    
    return args

def train(train_args, soft_tissue_args, lung_args):
    """ Soft-tissue 및 Lung CycleGAN 모델 학습 """
    # Soft-tissue CycleGAN 모델 학습
    print("Starting training for Soft-tissue CycleGAN...")
    soft_tissue_train_args = combine_args(train_args, soft_tissue_args)
    train_cycle_gan(soft_tissue_train_args, target_range='soft_tissue')
    print("Soft-tissue CycleGAN training completed.")

    # Lung CycleGAN 모델 학습
    print("Starting training for Lung CycleGAN...")
    lung_train_args = combine_args(train_args, lung_args)
    train_cycle_gan(lung_train_args, target_range='lung')
    print("Lung CycleGAN training completed.")
    
if __name__ == "__main__":
    print("Starting DUCOSY-GAN Training Process")
    
    # 인자 설정
    train_args = get_common_train_args() # 공통 훈련 인자
    soft_tissue_args = get_soft_tissue_infer_args() # Soft-tissue CycleGAN 추론 인자
    lung_args = get_lung_infer_args() # Lung CycleGAN 추론 인자
    
    # 학습 실행
    train(train_args, soft_tissue_args, lung_args)
    
    print("DUCOSY-GAN Training Process Completed")

    
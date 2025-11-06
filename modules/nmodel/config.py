import os
from datetime import datetime


class Config:
    """
    U-Net 학습 및 추론을 위한 설정 클래스
    """
    
    def __init__(self):
        # 데이터 경로
        self.data_dir = 'data'
        self.output_dir = 'output'
        
        # 모델 설정
        self.model_type = 'standard'  # 'light' 또는 'standard'
        self.in_channels = 1
        self.out_channels = 1
        self.base_channels = 16  # light: 16, standard: 32

        # 패치 기반 학습 설정 (메모리 절약)
        self.use_patches = True  # 패치 기반 학습 사용
        self.patch_size = (1, 512, 512)  # (D, H, W) - 패치 크기
        self.patches_per_volume = 128  # 전체 슬라이스 사용 (평균 depth ~126)
        
        # 학습 설정
        self.num_epochs = 100
        self.batch_size = 1  # 전체 데이터 사용으로 배치 크기 감소
        self.learning_rate = 5e-5  # 학습 안정성을 위해 낮춤 (1e-4 -> 5e-5)
        self.num_workers = 2  # 메모리 절약을 위해 줄임
        self.gradient_accumulation_steps = 8  # 배치 크기 감소로 증가 (effective batch size 유지)
        self.use_mixed_precision = True  # Mixed precision training (메모리 절약)
        self.use_gradient_checkpointing = True  # Gradient checkpointing (메모리 절약)
        self.gradient_clip_value = 1.0  # Gradient clipping 추가 (explosion 방지)
        
        # Loss 가중치
        self.l1_weight = 1.0  # L1 loss만 사용 (안정성 향상)
        self.ssim_weight = 0.0  # SSIM은 불안정할 수 있으므로 비활성화
        
        # 데이터 분할 (train/val only, test는 외부 검증 데이터 사용)
        self.val_size = 0.15  # 검증 데이터 비율
        
        # 체크포인트
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.save_interval = 10  # N 에포크마다 체크포인트 저장
        self.resume = False  # 기존 모델은 잘못된 정규화로 학습되었으므로 처음부터 시작
        self.resume_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        
        # 로깅
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = os.path.join(self.output_dir, 'logs', f'unet_{timestamp}')
        
        # 추론
        self.inference_checkpoint = os.path.join(self.checkpoint_dir, 'best.pth')
        
        # 디렉토리 생성
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def __repr__(self):
        """설정 정보 출력"""
        config_str = "=" * 50 + "\n"
        config_str += "Configuration\n"
        config_str += "=" * 50 + "\n"
        for key, value in self.__dict__.items():
            config_str += f"{key:20s}: {value}\n"
        config_str += "=" * 50
        return config_str
    
    def save(self, save_path):
        """설정을 파일로 저장"""
        import json
        with open(save_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
        print(f"Config saved to {save_path}")
    
    @classmethod
    def load(cls, load_path):
        """파일에서 설정 로드"""
        import json
        config = cls()
        with open(load_path, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                setattr(config, key, value)
        print(f"Config loaded from {load_path}")
        return config


# 사전 정의된 설정들
class LightConfig(Config):
    """경량 모델 설정 (메모리 효율적)"""
    def __init__(self):
        super().__init__()
        self.model_type = 'light'
        self.base_channels = 16
        self.batch_size = 1


class StandardConfig(Config):
    """표준 모델 설정 (더 높은 성능)"""
    def __init__(self):
        super().__init__()
        self.model_type = 'standard'
        self.base_channels = 32
        self.batch_size = 1


class FastTrainConfig(Config):
    """빠른 학습을 위한 설정 (테스트용)"""
    def __init__(self):
        super().__init__()
        self.model_type = 'light'
        self.base_channels = 8
        self.num_epochs = 10
        self.batch_size = 1


if __name__ == "__main__":
    # 설정 테스트
    config = Config()
    print(config)
    
    # 설정 저장
    config.save('config_test.json')
    
    # 설정 로드
    loaded_config = Config.load('config_test.json')
    print("\nLoaded config:")
    print(loaded_config)

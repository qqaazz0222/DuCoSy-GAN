import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class CTDiffDataset(Dataset):
    """
    CT Difference Map Dataset for U-Net training
    
    Args:
        data_dir: 데이터가 있는 디렉토리 경로
        mode: 'train', 'val', 'test' 중 하나
        transform: 데이터 증강 transform (옵션)
        use_patches: 패치 기반 학습 사용 여부
        patch_size: 패치 크기 (D, H, W)
        patches_per_volume: 볼륨당 추출할 패치 수
    """
    def __init__(self, data_dir, mode='train', transform=None, val_size=0.15, random_state=42,
                 use_patches=True, patch_size=(64, 512, 512), patches_per_volume=8):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        self.use_patches = use_patches
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        
        # diff_map 디렉토리에서 파일 로드
        diff_map_dir = os.path.join(data_dir, 'diff_map')
        
        # 모든 환자 ID 추출 (_diff.npy 파일 기준)
        all_files = [f for f in os.listdir(diff_map_dir) if f.endswith('_diff.npy')]
        patient_ids = [f.replace('_diff.npy', '') for f in all_files]
        
        # 데이터셋 분할 (train/val only, test는 외부 검증 데이터 사용)
        train_ids, val_ids = train_test_split(
            patient_ids, test_size=val_size, random_state=random_state
        )
        
        if mode == 'train':
            self.patient_ids = train_ids
        elif mode == 'val':
            self.patient_ids = val_ids
        else:
            raise ValueError(f"Unknown mode: {mode}. Only 'train' and 'val' are supported.")
        
        print(f"[{mode.upper()}] Loaded {len(self.patient_ids)} patients")
        if use_patches:
            print(f"[{mode.upper()}] Using patch-based training: patch_size={patch_size}, patches_per_volume={patches_per_volume}")
    
    def __len__(self):
        if self.use_patches:
            return len(self.patient_ids) * self.patches_per_volume
        return len(self.patient_ids)
    
    def extract_random_patch(self, volume, patch_size):
        """볼륨에서 랜덤 패치 추출"""
        d, h, w = volume.shape
        pd, ph, pw = patch_size
        
        # 패치를 추출할 수 있는 시작 위치 계산
        max_d = max(0, d - pd)
        max_h = max(0, h - ph)
        max_w = max(0, w - pw)
        
        # 랜덤 시작 위치
        start_d = np.random.randint(0, max_d + 1) if max_d > 0 else 0
        start_h = np.random.randint(0, max_h + 1) if max_h > 0 else 0
        start_w = np.random.randint(0, max_w + 1) if max_w > 0 else 0
        
        # 패치 추출
        patch = volume[start_d:start_d+pd, start_h:start_h+ph, start_w:start_w+pw]
        
        # 패치가 요구 크기보다 작으면 패딩
        if patch.shape != patch_size:
            padded = np.zeros(patch_size, dtype=volume.dtype)
            padded[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
            patch = padded
        
        return patch
    
    def extract_slice_patch(self, volume, slice_idx, patch_size):
        """특정 슬라이스를 패치로 추출 (모든 데이터 사용)"""
        d, h, w = volume.shape
        pd, ph, pw = patch_size
        
        # 슬라이스 인덱스가 범위를 벗어나면 마지막 슬라이스 사용
        if slice_idx >= d:
            slice_idx = d - 1
        
        # 단일 슬라이스 추출 (depth=1)
        if pd == 1:
            # 공간 패치 추출을 위한 랜덤 위치 (H, W)
            max_h = max(0, h - ph)
            max_w = max(0, w - pw)
            
            start_h = np.random.randint(0, max_h + 1) if max_h > 0 else 0
            start_w = np.random.randint(0, max_w + 1) if max_w > 0 else 0
            
            # 패치 추출
            patch = volume[slice_idx:slice_idx+1, start_h:start_h+ph, start_w:start_w+pw]
        else:
            # 3D 패치 추출
            max_d = max(0, d - pd)
            max_h = max(0, h - ph)
            max_w = max(0, w - pw)
            
            start_d = min(slice_idx, max_d)
            start_h = np.random.randint(0, max_h + 1) if max_h > 0 else 0
            start_w = np.random.randint(0, max_w + 1) if max_w > 0 else 0
            
            patch = volume[start_d:start_d+pd, start_h:start_h+ph, start_w:start_w+pw]
        
        # 패치가 요구 크기보다 작으면 패딩
        if patch.shape != patch_size:
            padded = np.zeros(patch_size, dtype=volume.dtype)
            padded[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
            patch = padded
        
        return patch
    
    def __getitem__(self, idx):
        if self.use_patches:
            # 패치 기반: idx를 patient_id와 patch_idx로 변환
            patient_idx = idx // self.patches_per_volume
            slice_idx = idx % self.patches_per_volume
            patient_id = self.patient_ids[patient_idx]
        else:
            patient_id = self.patient_ids[idx]
            slice_idx = 0
        
        original_dir = os.path.join(self.data_dir, 'vue_files')
        diff_map_dir = os.path.join(self.data_dir, 'diff_map')
        
        # VUE 이미지 로드 (입력) - DuCoSy-GAN의 출력 데이터에서 로드
        vue_path = os.path.join(original_dir, f'{patient_id}_vue.npy')
        vue = np.load(vue_path)
        
        # Raw difference map 로드 (타겟)
        raw_diff_path = os.path.join(diff_map_dir, f'{patient_id}_diff.npy')
        raw_diff = np.load(raw_diff_path)
        
        # 패치 기반 학습일 경우 슬라이스별 패치 추출
        if self.use_patches:
            vue = self.extract_slice_patch(vue, slice_idx, self.patch_size)
            raw_diff = self.extract_slice_patch(raw_diff, slice_idx, self.patch_size)
        
        # 정규화 (HU 값 범위 -1024 ~ 3071을 -1 ~ 1로)
        vue = self.normalize_hu(vue)
        raw_diff = self.normalize_diff(raw_diff)
        
        # NumPy array를 PyTorch tensor로 변환
        vue = torch.from_numpy(vue).float().unsqueeze(0)  # (1, D, H, W)
        raw_diff = torch.from_numpy(raw_diff).float().unsqueeze(0)  # (1, D, H, W)
        
        # Transform 적용 (있을 경우)
        if self.transform:
            vue = self.transform(vue)
            raw_diff = self.transform(raw_diff)
        
        return {
            'input': vue,
            'target': raw_diff,
            'patient_id': patient_id
        }
    
    @staticmethod
    def normalize_hu(volume, min_hu=-1024, max_hu=3071):
        """HU 값을 [-1, 1] 범위로 정규화"""
        volume = np.clip(volume, min_hu, max_hu)
        volume = (volume - min_hu) / (max_hu - min_hu)  # [0, 1]
        volume = volume * 2 - 1  # [-1, 1]
        return volume
    
    @staticmethod
    def normalize_diff(diff_map, min_diff=0, max_diff=4000):
        """
        Difference map을 [-1, 1] 범위로 정규화
        
        실제 데이터 분석 결과:
        - 실제 범위: [0, ~3682]
        - 설정 범위: [0, 4000] (여유 포함)
        """
        diff_map = np.clip(diff_map, min_diff, max_diff)
        diff_map = (diff_map - min_diff) / (max_diff - min_diff)  # [0, 1]
        diff_map = diff_map * 2 - 1  # [-1, 1]
        return diff_map
    
    @staticmethod
    def denormalize_hu(volume, min_hu=-1024, max_hu=3071):
        """정규화된 값을 HU 값으로 복원"""
        volume = (volume + 1) / 2  # [-1, 1] -> [0, 1]
        volume = volume * (max_hu - min_hu) + min_hu
        return volume
    
    @staticmethod
    def denormalize_diff(diff_map, min_diff=0, max_diff=4000):
        """
        정규화된 difference map을 원래 범위로 복원
        
        실제 데이터 분석 결과:
        - 실제 범위: [0, ~3682]
        - 설정 범위: [0, 4000] (여유 포함)
        """
        diff_map = (diff_map + 1) / 2  # [-1, 1] -> [0, 1]
        diff_map = diff_map * (max_diff - min_diff) + min_diff
        return diff_map


def get_dataloaders(data_dir, batch_size=1, num_workers=4, val_size=0.15,
                    use_patches=True, patch_size=(64, 128, 128), patches_per_volume=8):
    """
    데이터로더 생성 (train/val only)
    
    Args:
        data_dir: 데이터 디렉토리
        batch_size: 배치 크기
        num_workers: 워커 수
        val_size: 검증 데이터 비율
        use_patches: 패치 기반 학습 사용 여부
        patch_size: 패치 크기
        patches_per_volume: 볼륨당 추출할 패치 수
    
    Returns:
        train_loader, val_loader
    """
    # 데이터셋 생성
    train_dataset = CTDiffDataset(
        data_dir=data_dir,
        mode='train',
        val_size=val_size,
        use_patches=use_patches,
        patch_size=patch_size,
        patches_per_volume=patches_per_volume
    )
    
    val_dataset = CTDiffDataset(
        data_dir=data_dir,
        mode='val',
        val_size=val_size,
        use_patches=use_patches,
        patch_size=patch_size,
        patches_per_volume=patches_per_volume if use_patches else 1  # 검증은 패치 수 줄임
    )
    
    # 데이터로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # 메모리 절약을 위해 False로 설정
        prefetch_factor=2 if num_workers > 0 else None,  # prefetch 최소화
        persistent_workers=False  # 메모리 절약
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,  # 메모리 절약을 위해 False로 설정
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=False
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # 테스트 코드
    data_dir = 'data'
    train_loader, val_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=1,
        num_workers=0
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # 첫 번째 배치 확인
    for batch in train_loader:
        print(f"Input shape: {batch['input'].shape}")
        print(f"Target shape: {batch['target'].shape}")
        print(f"Patient ID: {batch['patient_id']}")
        break

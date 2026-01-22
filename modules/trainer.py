import os
import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
from pytorch_msssim import SSIM

from modules.model import Generator, Discriminator, weights_init_normal
from modules.preprocess import apply_windowing
from modules.dataset import DicomDataset

from tqdm import tqdm
import warnings

# pydicom 라이브러리에서 발생하는 사용자 경고를 무시
warnings.filterwarnings("ignore", category=UserWarning)

class GradientLoss(nn.Module):
    """
    이미지의 경계선(Gradient) 차이를 계산하여 선명도를 높이는 Loss
    """
    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, pred, target):
        # x축 방향 기울기 차이 (|pred_x - target_x|)
        dy_pred = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        dy_target = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        loss_y = torch.mean(torch.abs(dy_pred - dy_target))

        # y축 방향 기울기 차이 (|pred_y - target_y|)
        dx_pred = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        dx_target = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        loss_x = torch.mean(torch.abs(dx_pred - dx_target))

        return loss_x + loss_y


class ContrastAttentionLoss(nn.Module):
    """
    조영/비조영 이미지 간 차이가 큰 영역에 가중치를 부여하는 Attention Loss.
    
    [중요] 시간차 촬영 데이터 대응:
    - 듀얼 에너지 CT가 아닌 시간차 촬영 데이터는 해부학적 불일치(misalignment)가 있음
    - 픽셀 단위 비교 대신 패치/영역 단위 통계 기반 비교 사용
    - 가우시안 블러로 작은 위치 오차에 대한 내성 확보
    """
    def __init__(self, sigma=0.1, min_weight=1.0, max_weight=3.0, blur_kernel=5):
        super(ContrastAttentionLoss, self).__init__()
        self.sigma = sigma
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.blur_kernel = blur_kernel
        
        # 가우시안 블러를 위한 커널 생성 (작은 misalignment 허용)
        self.blur = nn.AvgPool2d(kernel_size=blur_kernel, stride=1, padding=blur_kernel//2)

    def forward(self, pred, target, source):
        """
        Args:
            pred: 생성된 이미지 (fake_B, NCCT -> CECT)
            target: 실제 조영 이미지 (real_B, CECT)
            source: 실제 비조영 이미지 (real_A, NCCT)
        """
        # 블러 적용으로 작은 위치 오차에 강건하게 만듦
        target_blurred = self.blur(target)
        source_blurred = self.blur(source)
        
        # 조영/비조영 간 차이 계산 (블러된 이미지 사용)
        contrast_diff = torch.abs(target_blurred - source_blurred)
        
        # 차이를 기반으로 가중치 맵 생성 (차이가 클수록 높은 가중치)
        # 단, max_weight를 낮춰서 misalignment 영향 완화
        attention_weight = self.min_weight + (self.max_weight - self.min_weight) * (
            1 - torch.exp(-contrast_diff / self.sigma)
        )
        
        # 가중치가 적용된 L1 손실 (블러된 예측과 타겟 비교)
        pred_blurred = self.blur(pred)
        weighted_loss = torch.mean(attention_weight * torch.abs(pred_blurred - target_blurred))
        
        return weighted_loss


class ContrastRegionLoss(nn.Module):
    """
    조영제로 인해 HU 값이 높아진 영역(혈관, 조영 증강 부위)을 
    더 정확하게 재현하도록 하는 손실 함수.
    
    [중요] 시간차 촬영 데이터 대응:
    - 픽셀 단위 마스크 대신 전체 이미지의 밝기 분포(히스토그램) 유사성 사용
    - 조영 영역의 평균 밝기와 분포를 맞추는 방식으로 변경
    """
    def __init__(self, threshold=0.3, weight=2.0):
        super(ContrastRegionLoss, self).__init__()
        self.threshold = threshold
        self.weight = weight
        self.pool = nn.AvgPool2d(kernel_size=8, stride=8)  # 패치 단위 비교

    def forward(self, pred, target, source):
        """
        Args:
            pred: 생성된 이미지 (fake_B)
            target: 실제 조영 이미지 (real_B)
            source: 실제 비조영 이미지 (real_A)
        """
        # 패치 단위로 다운샘플링하여 위치 오차에 강건하게
        pred_patches = self.pool(pred)
        target_patches = self.pool(target)
        source_patches = self.pool(source)
        
        # 조영으로 인해 밝아진 영역 마스크 생성 (패치 단위)
        contrast_enhancement = target_patches - source_patches
        
        # 조영 증강 영역 마스크 (부드러운 마스크)
        enhancement_mask = torch.sigmoid(5 * (contrast_enhancement - self.threshold))
        
        # 조영 영역에서의 손실
        region_loss = torch.mean(enhancement_mask * torch.abs(pred_patches - target_patches))
        
        # 추가: 전체 이미지의 밝기 분포 유사성 (히스토그램 매칭 효과)
        pred_mean, pred_std = pred.mean(), pred.std()
        target_mean, target_std = target.mean(), target.std()
        distribution_loss = torch.abs(pred_mean - target_mean) + torch.abs(pred_std - target_std)
        
        return self.weight * (region_loss + 0.5 * distribution_loss)


class ContrastEdgeLoss(nn.Module):
    """
    조영 경계선을 더 선명하게 학습하도록 하는 손실 함수.
    
    [중요] 시간차 촬영 데이터 대응:
    - 위치가 정확히 일치하지 않으므로, 특정 위치의 경계가 아닌
    - 전체 이미지의 경계 강도 분포를 비교하는 방식으로 변경
    - 생성 이미지가 타겟과 유사한 수준의 경계 선명도를 갖도록 유도
    """
    def __init__(self):
        super(ContrastEdgeLoss, self).__init__()
        # Sobel 필터 정의 (경계 검출용)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def get_edges(self, img):
        """이미지의 경계 추출"""
        edge_x = F.conv2d(img, self.sobel_x, padding=1)
        edge_y = F.conv2d(img, self.sobel_y, padding=1)
        edges = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
        return edges

    def forward(self, pred, target, source):
        """
        Args:
            pred: 생성된 이미지 (fake_B)
            target: 실제 조영 이미지 (real_B)
            source: 실제 비조영 이미지 (real_A) - 참고용
        """
        # 예측/타겟 이미지의 경계 추출
        pred_edges = self.get_edges(pred)
        target_edges = self.get_edges(target)
        
        # [수정] 위치 기반 비교 대신 경계 강도 통계 비교
        # 1. 경계 강도의 평균과 표준편차 비교 (전체적인 선명도 수준)
        pred_edge_mean, pred_edge_std = pred_edges.mean(), pred_edges.std()
        target_edge_mean, target_edge_std = target_edges.mean(), target_edges.std()
        
        stats_loss = torch.abs(pred_edge_mean - target_edge_mean) + \
                     torch.abs(pred_edge_std - target_edge_std)
        
        # 2. 경계 강도 히스토그램 유사성 (분포 비교)
        # 상위 k% 경계 강도의 평균 비교 (강한 경계 영역)
        k = 0.1  # 상위 10%
        pred_topk = torch.topk(pred_edges.flatten(), int(pred_edges.numel() * k)).values.mean()
        target_topk = torch.topk(target_edges.flatten(), int(target_edges.numel() * k)).values.mean()
        
        topk_loss = torch.abs(pred_topk - target_topk)
        
        return stats_loss + topk_loss


def validate_and_save_images(epoch, models, val_dataloader, criteria, args, device, fixed_val_batch):
    """ 검증 손실 계산 및 샘플 이미지 저장 """
    G_A2B, G_B2A, D_A, D_B = models
    criterion_GAN, criterion_cycle, criterion_identity = criteria
    
    G_A2B.eval()
    G_B2A.eval()
    
    total_val_loss_G = 0
    
    # 1. Validation Loss 계산
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            real_A, real_B = batch["A"].to(device), batch["B"].to(device)
            
            # 마스크가 있으면 입력에 결합
            if "masks" in batch:
                masks = batch["masks"].to(device)
                real_A_input = torch.cat([real_A, masks], dim=1)
                real_B_input = torch.cat([real_B, masks], dim=1)
            else:
                real_A_input = real_A
                real_B_input = real_B
            
            valid = torch.ones((real_A.size(0), 1, args.img_size // 16, args.img_size // 16), requires_grad=False).to(device)

            fake_B, fake_A = G_A2B(real_A_input), G_B2A(real_B_input)
            
            loss_id = (criterion_identity(G_B2A(real_A_input), real_A) + criterion_identity(G_A2B(real_B_input), real_B)) / 2
            loss_GAN = (criterion_GAN(D_B(fake_B), valid) + criterion_GAN(D_A(fake_A), valid)) / 2
            loss_cycle = (criterion_cycle(G_B2A(fake_B), real_A) + criterion_cycle(G_A2B(fake_A), real_B)) / 2
            
            loss_G = loss_GAN + args.lambda_cyc * loss_cycle + args.lambda_id * loss_id
            total_val_loss_G += loss_G.item()

    avg_val_loss = total_val_loss_G / len(val_dataloader)

    # 2. 고정된 배치로 샘플 이미지 저장
    with torch.no_grad():
        real_A = fixed_val_batch["A"].to(device)
        real_B = fixed_val_batch["B"].to(device)
        
        # 마스크가 있으면 입력에 결합
        if "masks" in fixed_val_batch:
            masks = fixed_val_batch["masks"].to(device)
            real_A_input = torch.cat([real_A, masks], dim=1)
        else:
            real_A_input = real_A
        
        fake_B = G_A2B(real_A_input)

        real_A_win = apply_windowing(real_A, args)
        real_B_win = apply_windowing(real_B, args)
        fake_B_win = apply_windowing(fake_B, args)

        image_grid = torch.cat((real_A_win, fake_B_win, real_B_win), -1)
        save_path = os.path.join(args.training_dir, "images", f"epoch_{epoch+1}.jpg")
        save_image(image_grid, save_path, nrow=min(real_A.size(0), 4), normalize=False) # nrow를 조절하여 보기 좋게 저장
        
    G_A2B.train()
    G_B2A.train()
    
    return avg_val_loss


def train_cycle_gan(args, target_range):
    """ CycleGAN 모델 학습 함수 """
    if target_range not in ['soft_tissue', 'lung']:
        raise ValueError("target_range must be either 'soft_tissue' or 'lung'")
    
    # GPU 장치 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Training on CPU.")

    gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7] if torch.cuda.is_available() and torch.cuda.device_count() > 1 else [0]

    # 출력 디렉토리 생성
    cur_training_dir = os.path.join(args.training_dir, target_range)
    os.makedirs(cur_training_dir, exist_ok=True)
    args.training_dir = cur_training_dir  # 각 타겟 범위별로 다른 훈련 디렉토리 설정
    images_dir = os.path.join(args.training_dir, "images")
    saved_models_dir = os.path.join(args.training_dir, "saved_models")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(saved_models_dir, exist_ok=True)
    print(f"Starting training with args: {args}")

    # 모델 초기화
    input_channels = 1  # 기본값
    if getattr(args, 'use_masks', False) and getattr(args, 'mask_folders', []):
        # 원본 이미지 1채널 + 마스크 채널 수
        input_channels = 1 + len(args.mask_folders)
        print(f"Using {len(args.mask_folders)} mask(s): {args.mask_folders}")
        print(f"Total input channels: {input_channels}")
    
    use_cbam = getattr(args, 'use_cbam', True)
    G_A2B = Generator(input_channels=input_channels, use_cbam=use_cbam)
    G_B2A = Generator(input_channels=input_channels, use_cbam=use_cbam)
    D_A, D_B = Discriminator(), Discriminator()
    
    # DataParallel로 모델 감싸기
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel.")
        G_A2B = nn.DataParallel(G_A2B, device_ids=gpu_ids).to(device)
        G_B2A = nn.DataParallel(G_B2A, device_ids=gpu_ids).to(device)
        D_A = nn.DataParallel(D_A, device_ids=gpu_ids).to(device)
        D_B = nn.DataParallel(D_B, device_ids=gpu_ids).to(device)

    else:
        G_A2B = G_A2B.to(device)
        G_B2A = G_B2A.to(device)
        D_A = D_A.to(device)
        D_B = D_B.to(device)
    
    # 손실 함수, 옵티마이저, 스케줄러 설정
    criterion_GAN = torch.nn.MSELoss().to(device)
    criterion_cycle = torch.nn.L1Loss().to(device)
    criterion_identity = torch.nn.L1Loss().to(device)
    criterion_gradient = GradientLoss().to(device)
    criterion_ssim = SSIM(data_range=1.0, size_average=True, channel=1).to(device)
    
    # 조영 효과 집중 학습을 위한 추가 손실 함수들
    # [참고] 시간차 촬영 데이터(듀얼 에너지 CT 아님)에 맞게 설계됨
    # - 픽셀 단위 비교 대신 패치/통계 기반 비교로 misalignment에 강건함
    criterion_contrast_attention = ContrastAttentionLoss(sigma=0.15, min_weight=1.0, max_weight=3.0, blur_kernel=7).to(device)
    criterion_contrast_region = ContrastRegionLoss(threshold=0.15, weight=1.5).to(device)
    criterion_contrast_edge = ContrastEdgeLoss().to(device)
    
    optimizer_G = torch.optim.Adam(list(G_A2B.parameters()) + list(G_B2A.parameters()), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=args.lr, betas=(0.5, 0.999))

    lr_lambda = lambda epoch: 1.0 - max(0, epoch + 1 - args.decay_epoch) / (args.epochs - args.decay_epoch)
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda)
    scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda)
    scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda)
    
    start_epoch = 0
    best_val_loss = float('inf')
    best_epoch = -1

    # --- 체크포인트 불러오기 ---
    if args.resume:
        checkpoint_path = os.path.join(saved_models_dir, args.resume)
        if os.path.isfile(checkpoint_path):
            print(f"=> Loading checkpoint '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            # DataParallel 래핑 여부에 관계없이 state_dict를 로드하기 위한 처리
            def load_state_dict_flexible(model, state_dict):
                is_parallel = isinstance(model, nn.DataParallel)
                if not is_parallel and all(key.startswith('module.') for key in state_dict.keys()):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                elif is_parallel and not all(key.startswith('module.') for key in state_dict.keys()):
                    state_dict = {'module.' + k: v for k, v in state_dict.items()}
                model.load_state_dict(state_dict)

            load_state_dict_flexible(G_A2B, checkpoint['G_A2B_state_dict'])
            load_state_dict_flexible(G_B2A, checkpoint['G_B2A_state_dict'])
            load_state_dict_flexible(D_A, checkpoint['D_A_state_dict'])
            load_state_dict_flexible(D_B, checkpoint['D_B_state_dict'])
            
            optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            optimizer_D_A.load_state_dict(checkpoint['optimizer_D_A_state_dict'])
            optimizer_D_B.load_state_dict(checkpoint['optimizer_D_B_state_dict'])
            
            scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
            scheduler_D_A.load_state_dict(checkpoint['scheduler_D_A_state_dict'])
            scheduler_D_B.load_state_dict(checkpoint['scheduler_D_B_state_dict'])

            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            best_epoch = checkpoint.get('best_epoch', -1) ### 추가 ###: 체크포인트에서 best_epoch 불러오기
            
            print(f"=> Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']+1})")
        else:
            print(f"=> No checkpoint found at '{checkpoint_path}'")
            G_A2B.apply(weights_init_normal)
            G_B2A.apply(weights_init_normal)
            D_A.apply(weights_init_normal)
            D_B.apply(weights_init_normal)
    else:
        G_A2B.apply(weights_init_normal)
        G_B2A.apply(weights_init_normal)
        D_A.apply(weights_init_normal)
        D_B.apply(weights_init_normal)

    # 데이터 로딩
    transforms_ = transforms.Compose([transforms.ToTensor(), transforms.Resize((args.img_size, args.img_size), antialias=True)])
    
    all_patient_dirs = sorted(glob.glob(os.path.join(args.data_root, args.dataset_names, "*")))
    random.seed(42)
    random.shuffle(all_patient_dirs)
    
    val_count = int(len(all_patient_dirs) * args.val_split)
    train_dirs, val_dirs = all_patient_dirs[val_count:], all_patient_dirs[:val_count]

    train_dataset = DicomDataset(train_dirs, args, transform=transforms_)
    val_dataset = DicomDataset(val_dirs, args, transform=transforms_)

    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    
    fixed_val_batch = next(iter(val_dataloader))
    print(f"Train/Val split: {len(train_dataset)} slices / {len(val_dataset)} slices")

    # --- 학습 루프 ---
    for epoch in range(start_epoch, args.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        G_A2B.train()
        G_B2A.train()
        D_A.train()
        D_B.train()

        for i, batch in enumerate(pbar):
            real_A, real_B = batch["A"].to(device), batch["B"].to(device)
            
            # 마스크가 있으면 입력에 결합
            if "masks" in batch:
                masks = batch["masks"].to(device)
                real_A_input = torch.cat([real_A, masks], dim=1)
                real_B_input = torch.cat([real_B, masks], dim=1)
            else:
                real_A_input = real_A
                real_B_input = real_B
            
            valid = torch.ones((real_A.size(0), 1, args.img_size // 16, args.img_size // 16), requires_grad=False).to(device)
            fake = torch.zeros((real_A.size(0), 1, args.img_size // 16, args.img_size // 16), requires_grad=False).to(device)

            # --- 생성자(Generator) 학습 ---
            optimizer_G.zero_grad()
            fake_B, fake_A = G_A2B(real_A_input), G_B2A(real_B_input)

            # Identity mapping
            id_A, id_B = G_B2A(real_A_input), G_A2B(real_B_input)

            loss_id = (criterion_identity(id_A, real_A) + criterion_identity(id_B, real_B)) / 2
            loss_GAN = (criterion_GAN(D_B(fake_B), valid) + criterion_GAN(D_A(fake_A), valid)) / 2
            
            # Reconstruction (Cycle consistency)
            # fake_B와 fake_A에도 마스크를 결합하여 입력
            if "masks" in batch:
                fake_B_input = torch.cat([fake_B, masks], dim=1)
                fake_A_input = torch.cat([fake_A, masks], dim=1)
            else:
                fake_B_input = fake_B
                fake_A_input = fake_A
            rec_A, rec_B = G_B2A(fake_B_input), G_A2B(fake_A_input)
            
            loss_cycle = (criterion_cycle(rec_A, real_A) + criterion_cycle(rec_B, real_B)) / 2
            loss_grad_cycle = (criterion_gradient(rec_A, real_A) + criterion_gradient(rec_B, real_B)) / 2
            loss_grad_id = (criterion_gradient(id_A, real_A) + criterion_gradient(id_B, real_B)) / 2
            loss_ssim = 1 - ((criterion_ssim(rec_A, real_A) + criterion_ssim(rec_B, real_B)) / 2)
            
            # 조영 효과 집중 학습 손실 (NCCT -> CECT 방향에 집중)
            # fake_B는 NCCT에서 생성한 가상 CECT, real_B는 실제 CECT, real_A는 NCCT
            loss_contrast_attention = criterion_contrast_attention(fake_B, real_B, real_A)
            loss_contrast_region = criterion_contrast_region(fake_B, real_B, real_A)
            loss_contrast_edge = criterion_contrast_edge(fake_B, real_B, real_A)
            
            lambda_grad = 5.0
            lambda_grad_id = 2.5
            lambda_ssim = 2.0
            
            # 조영 효과 손실 가중치 (시간차 촬영 데이터에 맞게 조정)
            # - 픽셀 정렬이 완벽하지 않으므로 가중치를 낮춤
            # - 통계 기반 비교이므로 안정적인 학습 가능
            lambda_contrast_attention = 2.0  # 조영 차이 영역 집중 (블러 적용)
            lambda_contrast_region = 1.5     # 조영 증강 영역 재현 (패치 기반)
            lambda_contrast_edge = 1.0       # 경계 선명도 (통계 기반)
            
            loss_G = loss_GAN + \
                     args.lambda_cyc * loss_cycle + \
                     args.lambda_id * loss_id + \
                     lambda_grad * loss_grad_cycle + \
                     lambda_grad_id * loss_grad_id + \
                     lambda_ssim * loss_ssim + \
                     lambda_contrast_attention * loss_contrast_attention + \
                     lambda_contrast_region * loss_contrast_region + \
                     lambda_contrast_edge * loss_contrast_edge
            loss_G.backward()
            optimizer_G.step()

            # --- 판별자(Discriminator) 학습 ---
            optimizer_D_A.zero_grad()
            loss_D_A = (criterion_GAN(D_A(real_A), valid) + criterion_GAN(D_A(fake_A.detach()), fake)) / 2
            loss_D_A.backward()
            optimizer_D_A.step()
            
            optimizer_D_B.zero_grad()
            loss_D_B = (criterion_GAN(D_B(real_B), valid) + criterion_GAN(D_B(fake_B.detach()), fake)) / 2
            loss_D_B.backward()
            optimizer_D_B.step()
            
            pbar.set_postfix({
                "G_loss": f"{loss_G.item():.4f}", 
                "D_loss": f"{(loss_D_A + loss_D_B).item():.4f}",
                "contrast": f"{(loss_contrast_attention + loss_contrast_region + loss_contrast_edge).item():.4f}"
            })

        # 스케줄러 업데이트
        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_B.step()
        
        # --- 에폭 종료 후 검증, 모델 저장, 체크포인트 저장 ---
        val_loss = validate_and_save_images(
            epoch, (G_A2B, G_B2A, D_A, D_B), val_dataloader,
            (criterion_GAN, criterion_cycle, criterion_identity), args, device, fixed_val_batch
        )
        print(f"\nEpoch {epoch+1} finished. Validation Generator Loss: {val_loss:.4f}")

        model_to_save_G_A2B = G_A2B.module if isinstance(G_A2B, nn.DataParallel) else G_A2B
        model_to_save_G_B2A = G_B2A.module if isinstance(G_B2A, nn.DataParallel) else G_B2A
        model_to_save_D_A = D_A.module if isinstance(D_A, nn.DataParallel) else D_A
        model_to_save_D_B = D_B.module if isinstance(D_B, nn.DataParallel) else D_B

        ### --- 수정된 Best 모델 저장 로직 --- ###
        if val_loss < best_val_loss:
            # 이전 best 모델 파일이 있다면 삭제
            if best_epoch != -1:
                old_best_G_A2B_path = os.path.join(saved_models_dir, f"G_A2B_best_epoch_{best_epoch}.pth")
                old_best_G_B2A_path = os.path.join(saved_models_dir, f"G_B2A_best_epoch_{best_epoch}.pth")
                if os.path.exists(old_best_G_A2B_path): os.remove(old_best_G_A2B_path)
                if os.path.exists(old_best_G_B2A_path): os.remove(old_best_G_B2A_path)

            best_val_loss = val_loss
            best_epoch = epoch + 1
            
            # 새 best 모델을 epoch 정보와 함께 저장
            torch.save(model_to_save_G_A2B.state_dict(), os.path.join(saved_models_dir, f"G_A2B_best_epoch_{best_epoch}.pth"))
            torch.save(model_to_save_G_B2A.state_dict(), os.path.join(saved_models_dir, f"G_B2A_best_epoch_{best_epoch}.pth"))
            print(f"✨ New best models saved for epoch {best_epoch} with validation loss: {best_val_loss:.4f}")

        # 에폭별 모델 (단순 가중치) 저장
        torch.save(model_to_save_G_A2B.state_dict(), os.path.join(saved_models_dir, f"G_A2B_epoch_{epoch+1}.pth"))
        torch.save(model_to_save_G_B2A.state_dict(), os.path.join(saved_models_dir, f"G_B2A_epoch_{epoch+1}.pth"))
        
        # Last 모델 (단순 가중치) 저장
        torch.save(model_to_save_G_A2B.state_dict(), os.path.join(saved_models_dir, "G_A2B_last.pth"))
        torch.save(model_to_save_G_B2A.state_dict(), os.path.join(saved_models_dir, "G_B2A_last.pth"))
        
        # 체크포인트 (전체 상태) 저장
        checkpoint_state = {
            'epoch': epoch,
            'G_A2B_state_dict': model_to_save_G_A2B.state_dict(),
            'G_B2A_state_dict': model_to_save_G_B2A.state_dict(),
            'D_A_state_dict': model_to_save_D_A.state_dict(),
            'D_B_state_dict': model_to_save_D_B.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_A_state_dict': optimizer_D_A.state_dict(),
            'optimizer_D_B_state_dict': optimizer_D_B.state_dict(),
            'scheduler_G_state_dict': scheduler_G.state_dict(),
            'scheduler_D_A_state_dict': scheduler_D_A.state_dict(),
            'scheduler_D_B_state_dict': scheduler_D_B.state_dict(),
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'args': args
        }
        torch.save(checkpoint_state, os.path.join(saved_models_dir, "checkpoint.pth.tar"))
        print(f"Checkpoint and last models saved for epoch {epoch+1}.\n")
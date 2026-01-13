import os
import glob
import random
import torch
import torch.nn as nn
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
            valid = torch.ones((real_A.size(0), 1, args.img_size // 16, args.img_size // 16), requires_grad=False).to(device)

            fake_B, fake_A = G_A2B(real_A), G_B2A(real_B)
            
            loss_id = (criterion_identity(G_B2A(real_A), real_A) + criterion_identity(G_A2B(real_B), real_B)) / 2
            loss_GAN = (criterion_GAN(D_B(fake_B), valid) + criterion_GAN(D_A(fake_A), valid)) / 2
            loss_cycle = (criterion_cycle(G_B2A(fake_B), real_A) + criterion_cycle(G_A2B(fake_A), real_B)) / 2
            
            loss_G = loss_GAN + args.lambda_cyc * loss_cycle + args.lambda_id * loss_id
            total_val_loss_G += loss_G.item()

    avg_val_loss = total_val_loss_G / len(val_dataloader)

    # 2. 고정된 배치로 샘플 이미지 저장
    with torch.no_grad():
        real_A = fixed_val_batch["A"].to(device)
        real_B = fixed_val_batch["B"].to(device)
        fake_B = G_A2B(real_A)

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
    G_A2B, G_B2A = Generator(), Generator()
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
            valid = torch.ones((real_A.size(0), 1, args.img_size // 16, args.img_size // 16), requires_grad=False).to(device)
            fake = torch.zeros((real_A.size(0), 1, args.img_size // 16, args.img_size // 16), requires_grad=False).to(device)

            # --- 생성자(Generator) 학습 ---
            optimizer_G.zero_grad()
            fake_B, fake_A = G_A2B(real_A), G_B2A(real_B)

            # Identity mapping
            id_A, id_B = G_B2A(real_A), G_A2B(real_B)

            loss_id = (criterion_identity(id_A, real_A) + criterion_identity(id_B, real_B)) / 2
            loss_GAN = (criterion_GAN(D_B(fake_B), valid) + criterion_GAN(D_A(fake_A), valid)) / 2
            
            # Reconstruction (Cycle consistency)
            rec_A, rec_B = G_B2A(fake_B), G_A2B(fake_A)
            
            loss_cycle = (criterion_cycle(rec_A, real_A) + criterion_cycle(rec_B, real_B)) / 2
            loss_grad_cycle = (criterion_gradient(rec_A, real_A) + criterion_gradient(rec_B, real_B)) / 2
            loss_grad_id = (criterion_gradient(id_A, real_A) + criterion_gradient(id_B, real_B)) / 2
            loss_ssim = 1 - ((criterion_ssim(rec_A, real_A) + criterion_ssim(rec_B, real_B)) / 2)
            
            lambda_grad = 5.0
            lambda_grad_id = 2.5
            lambda_ssim = 2.0
            
            loss_G = loss_GAN + \
                     args.lambda_cyc * loss_cycle + \
                     args.lambda_id * loss_id + \
                     lambda_grad * loss_grad_cycle + \
                     lambda_grad_id * loss_grad_id + \
                     lambda_ssim * loss_ssim
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
            
            pbar.set_postfix({"G_loss": f"{loss_G.item():.4f}", "D_loss": f"{(loss_D_A + loss_D_B).item():.4f}"})

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

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv3D -> BN -> ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # depth 차원은 pooling하지 않고 spatial 차원만 pooling (kernel_size=(1,2,2))
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()

        # depth 차원은 upsampling하지 않고 spatial 차원만 upsampling (scale_factor=(1,2,2))
        if trilinear:
            self.up = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net for CT difference map prediction
    
    Args:
        n_channels: 입력 채널 수 (기본값: 1)
        n_classes: 출력 채널 수 (기본값: 1)
        trilinear: Trilinear upsampling 사용 여부 (기본값: True)
        base_channels: 첫 번째 레이어의 채널 수 (기본값: 64)
    """
    
    def __init__(self, n_channels=1, n_classes=1, trilinear=True, base_channels=32):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.trilinear = trilinear

        self.inc = DoubleConv(n_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        factor = 2 if trilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)
        
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, trilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, trilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, trilinear)
        self.up4 = Up(base_channels * 2, base_channels, trilinear)
        self.outc = OutConv(base_channels, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNet3DLight(nn.Module):
    """
    Lightweight 3D U-Net for memory-efficient training
    
    Args:
        n_channels: 입력 채널 수 (기본값: 1)
        n_classes: 출력 채널 수 (기본값: 1)
        trilinear: Trilinear upsampling 사용 여부 (기본값: True)
        base_channels: 첫 번째 레이어의 채널 수 (기본값: 16)
    """
    
    def __init__(self, n_channels=1, n_classes=1, trilinear=True, base_channels=16):
        super(UNet3DLight, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.trilinear = trilinear

        self.inc = DoubleConv(n_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        factor = 2 if trilinear else 1
        self.down3 = Down(base_channels * 4, base_channels * 8 // factor)
        
        self.up1 = Up(base_channels * 8, base_channels * 4 // factor, trilinear)
        self.up2 = Up(base_channels * 4, base_channels * 2 // factor, trilinear)
        self.up3 = Up(base_channels * 2, base_channels, trilinear)
        self.outc = OutConv(base_channels, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits


def count_parameters(model):
    """모델의 학습 가능한 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 모델 테스트
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 표준 U-Net
    model = UNet3D(n_channels=1, n_classes=1, base_channels=32).to(device)
    print(f"Standard UNet3D parameters: {count_parameters(model):,}")
    
    # Lightweight U-Net
    model_light = UNet3DLight(n_channels=1, n_classes=1, base_channels=16).to(device)
    print(f"Lightweight UNet3D parameters: {count_parameters(model_light):,}")
    
    # 더미 입력으로 테스트 (작은 크기로 테스트)
    dummy_input = torch.randn(1, 1, 32, 64, 64).to(device)
    output = model_light(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

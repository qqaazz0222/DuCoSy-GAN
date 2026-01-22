import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- CBAM (Convolutional Block Attention Module) ----
class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# ---- 모델 아키텍처 정의 ----
class ResidualBlock(nn.Module):
    """ 2개의 CONV-BN-ReLU 층으로 구성된 잔차 블록 """
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(in_features, in_features, 3), nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True), nn.ReflectionPad2d(1), nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features))
    def forward(self, x):
        return x + self.block(x)


class ResidualBlockWithCBAM(nn.Module):
    """ CBAM Attention이 적용된 잔차 블록 """
    def __init__(self, in_features):
        super(ResidualBlockWithCBAM, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), 
            nn.Conv2d(in_features, in_features, 3), 
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True), 
            nn.ReflectionPad2d(1), 
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )
        self.cbam = CBAM(in_features)
    
    def forward(self, x):
        residual = x
        out = self.block(x)
        out = self.cbam(out)
        return residual + out
    
    
class Generator(nn.Module):
    """ ResNet 기반 생성기 (CBAM Attention 포함) """
    def __init__(self, input_channels=1, num_residual_blocks=9, use_cbam=True):
        super(Generator, self).__init__()
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_channels, 64, 7), nn.InstanceNorm2d(64), nn.ReLU(inplace=True)]
        in_features, out_features = 64, 128
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1), nn.InstanceNorm2d(out_features), nn.ReLU(inplace=True)]
            in_features, out_features = out_features, out_features * 2
        
        # ResNet blocks with optional CBAM attention
        for _ in range(num_residual_blocks):
            if use_cbam:
                model += [ResidualBlockWithCBAM(in_features)]
            else:
                model += [ResidualBlock(in_features)]
        
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.Upsample(scale_factor=2), nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                      nn.InstanceNorm2d(out_features), nn.ReLU(inplace=True)]
            in_features, out_features = out_features, out_features // 2
        model += [nn.ReflectionPad2d(3), nn.Conv2d(in_features, 1, 7), nn.Tanh()]  # output_channels=1로 고정
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)
    
    
class Discriminator(nn.Module):
    """ PatchGAN 기반 판별기 """
    def __init__(self, input_channels=1):
        super(Discriminator, self).__init__()
        def block(in_f, out_f, norm=True):
            layers = [nn.Conv2d(in_f, out_f, 4, stride=2, padding=1)]
            if norm: layers.append(nn.InstanceNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            *block(input_channels, 64, norm=False), *block(64, 128), *block(128, 256), *block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)), nn.Conv2d(512, 1, 4, padding=1))
    def forward(self, img):
        return self.model(img)
    
    
def weights_init_normal(m):
    """ 모델 가중치를 정규분포로 초기화 """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1: torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
    
class Generator(nn.Module):
    """ ResNet 기반 생성기 """
    def __init__(self, input_channels=1, num_residual_blocks=9):
        super(Generator, self).__init__()
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_channels, 64, 7), nn.InstanceNorm2d(64), nn.ReLU(inplace=True)]
        in_features, out_features = 64, 128
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1), nn.InstanceNorm2d(out_features), nn.ReLU(inplace=True)]
            in_features, out_features = out_features, out_features * 2
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(in_features)]
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.Upsample(scale_factor=2), nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                      nn.InstanceNorm2d(out_features), nn.ReLU(inplace=True)]
            in_features, out_features = out_features, out_features // 2
        model += [nn.ReflectionPad2d(3), nn.Conv2d(in_features, input_channels, 7), nn.Tanh()]
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

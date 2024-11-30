import torch
import torch.nn as nn
from torchvision import models

class ImprovedUNet(nn.Module):
    def __init__(self, pretrained=True):
        super(ImprovedUNet, self).__init__()
        
        # ResNet18 백본 초기화
        if pretrained:
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet18(weights=None)
        
        # 첫 번째 레이어 수정 (1채널 입력)
        self.firstconv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.firstconv.weight.data = torch.sum(resnet.conv1.weight.data, dim=1, keepdim=True) / 3
        
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        
        # 인코더
        self.encoder1 = resnet.layer1  # 64
        self.encoder2 = resnet.layer2  # 128
        self.encoder3 = resnet.layer3  # 256
        self.encoder4 = resnet.layer4  # 512
        
        # 디코더 (스킵 커넥션 고려한 채널 수 수정)
        self.decoder4 = self._deconv_block(512, 256)  # 512 -> 256
        self.decoder3 = self._deconv_block(512, 128)  # 256 + 256 = 512 -> 128
        self.decoder2 = self._deconv_block(256, 64)   # 128 + 128 = 256 -> 64
        self.decoder1 = self._deconv_block(128, 32)   # 64 + 64 = 128 -> 32
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 인코더
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x1 = self.firstmaxpool(x)
        
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)
        
        # 디코더 (스킵 커넥션)
        d4 = self.decoder4(x5)
        d4 = torch.cat([d4, x4], dim=1)
        
        d3 = self.decoder3(d4)
        d3 = torch.cat([d3, x3], dim=1)
        
        d2 = self.decoder2(d3)
        d2 = torch.cat([d2, x2], dim=1)
        
        d1 = self.decoder1(d2)
        
        return self.final(d1)

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.4, beta=0.4, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.bce = nn.BCELoss()
        
    def forward(self, pred, target):
        smooth = 1e-5
        
        # Dice Loss
        intersection = (pred * target).sum()
        dice_loss = 1 - ((2. * intersection + smooth) / 
                        (pred.sum() + target.sum() + smooth))
        
        # Focal Loss
        bce_loss = self.bce(pred, target)
        pt = torch.exp(-bce_loss)
        focal_loss = (1-pt)**self.gamma * bce_loss
        
        return (self.alpha * dice_loss + 
                self.beta * focal_loss + 
                (1-self.alpha-self.beta) * bce_loss)
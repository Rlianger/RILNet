import copy

import torch
import torch.nn as nn
from model.cbam import CBAMLayer

###
# 特征放大层
###
def conv3x3_block(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    conv_layer = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)

    block = nn.Sequential(
        conv_layer,
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )
    return block
class FAMLayer(nn.Module):
    def __init__(self, in_chanel):
        super(FAMLayer, self).__init__()
        self.FAM1 = nn.Sequential(
            SE_Block(in_chanel),
            nn.Conv2d(in_chanel, in_chanel, kernel_size=3, padding=1),
            nn.Conv2d(in_chanel, in_chanel // 2, kernel_size=1, padding=0),
            nn.Conv2d(in_chanel // 2, in_chanel // 2, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_chanel // 2),
            nn.ReLU(),
            nn.Conv2d(in_chanel // 2, in_chanel, kernel_size=1, padding=0),
            nn.Conv2d(in_chanel, in_chanel, kernel_size=1, padding=0),
        )
        self.FAM2 = nn.Sequential(
            SE_Block(in_chanel),
            nn.Conv2d(in_chanel, in_chanel // 2, kernel_size=3, padding=1),
            nn.Conv2d(in_chanel// 2, in_chanel // 2, kernel_size=1, padding=0),
            nn.Conv2d(in_chanel // 2, in_chanel, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_chanel),
            nn.ReLU(),
        )
        self.selfAttc = nn.Sequential(
            CBAMLayer(in_chanel),
            conv3x3_block(in_chanel, in_chanel // 2),
            # conv3x3_block(in_chanel // 2, in_chanel // 2),
            conv3x3_block(in_chanel // 2, in_chanel),
            # conv3x3_block(in_chanel, in_chanel),
            # CBAMLayer(768)
        )


    def forward(self, x):
        x = self.FAM2(x) + x
        x = self.selfAttc(x) + x
        return x



# 全局平均池化+1*1卷积核+ReLu+1*1卷积核+Sigmoid
class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Conv2d(inchannel, inchannel// ratio, 1,padding=0),
            # nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Conv2d(inchannel // ratio, inchannel, 1, padding=0),
            # nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        # 读取批数据图片数量及通道数
        b, c, h, w = x.size()
        # Fsq操作：经池化后输出b*c的矩阵
        y = self.gap(x).view(b, c, 1, 1)
        # Fex操作：经全连接层输出（b，c，1，1）矩阵
        y = self.fc(y).view(b, c, 1, 1)
        # Fscale操作：将得到的权重乘以原来的特征图x
        return x * y.expand_as(x)

if __name__ == '__main__':
    # d5 = nn.Conv2d(210, 210, kernel_size=5, padding=2)
    fma = nn.Sequential(
        SE_Block(210),
        nn.Conv2d(210, 210, kernel_size=3, padding=1),
        nn.Conv2d(210, 210 // 2 , kernel_size=1, padding=0),
        nn.Conv2d(210 // 2, 210 // 2 , kernel_size=1, padding=0),
        nn.BatchNorm2d(210//2),
        nn.ReLU(),
        nn.Conv2d(210 // 2, 210, kernel_size=1, padding=0),
        nn.Conv2d(210, 210, kernel_size=1, padding=0),
    )
    fma2 = copy.deepcopy(fma)
    input = torch.randn(8, 210, 32, 24) # b n h w
    input1 = fma(input) + input
    output = fma(input1) + input1
    print(output.size())


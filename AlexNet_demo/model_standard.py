import torch.nn as nn
import torch
import torch.nn.functional as F

class AlenNet(nn.Module):
    def __init__(self, num_class=1000):
        super(AlenNet, self).__init__()
        #nn.Sequential 是一个有序的模块容器，它将一系列的 PyTorch 模块按顺序组合在一起，以构建神经网络模型。
        #使用 nn.Sequential 时，你可以将一个列表作为输入,注意要用逗号隔开
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernal_size=11, stride=4, padding=2),
            #inplace=True 表示将计算结果直接存储在输入张量中，而不是创建一个新的张量来存储结果。
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernal_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 256, kernal_size=2, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernal_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 384, kernal_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernal_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernal_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernal_size=3, stride=2)
        )
        #使用的一个自适应平均池化层。这个层将输入张量的大小自适应地调整为 (6, 6) 的输出大小，使得无论输入张量的大小如何，都可以被转换为一个固定大小的张量。
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classfier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_class)
        )
    def forward(self, x):
        x = self.features(x)    
        x = self.avgpool(x)
        #torch.flatten(x, 1) 这一行代码的作用是将四维张量 x 沿着第二个维度（即 num_channels 维度）进行展平
        #最终得到一个形状为 (batch_size, num_features) 的二维张量，其中 num_features 表示特征向量的维度大小，即 num_channels x 6 x 6。
        x = torch.flatten(x, 1)
        x = self.classfier(x)
        return x
    
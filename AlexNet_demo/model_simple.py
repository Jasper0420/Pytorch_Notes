import torch.nn as nn
import torch.nn.functional as F

#nn.Module是所有神经网络模块的基类，继承自该类的类都必须实现前向传播方法forward()，以及初始化方法__init__()。
class AlexNet(nn.Module):
    def __init__(self):
        #super(AlexNet,self).__init__()调用了父类（即nn.Module类）的初始化方法__init__()，并将当前类AlexNet的实例和self作为参数传递给它。
        super(AlexNet, self).__init__()
        #第 1 层：卷积层，96 个大小为 11x11 的卷积核，步长为 4，padding 为 0
        self.conv1 = nn.Conv2d(3, 96, 11, 4)
        #第 2 层：池化层，2x2 的池化核，步长为 2，padding 为 0
        self.pool1 = nn.MaxPool2d(2, 2)
        #第 3 层：卷积层，256 个大小为 5x5 的卷积核，步长为 1，padding 为 2
        self.conv2 = nn.Conv2d(96, 256, 5, 1, 2)
        #第 4 层：池化层，2x2 的池化核，步长为 2，padding 为 0
        self.pool2 = nn.MaxPool2d(2, 2)
        #第 5 层：卷积层，384 个大小为 3x3 的卷积核，步长为 1，padding 为 1
        self.conv3 = nn.Conv2d(256, 384, 3, 1, 1)
        #第 6 层：卷积层，384 个大小为 3x3 的卷积核，步长为 1，padding 为 1
        self.conv4 = nn.Conv2d(384, 384, 3, 1, 1)
        #第 7 层：卷积层，256 个大小为 3x3 的卷积核，步长为 1，padding 为 1
        self.conv5 = nn.Conv2d(384, 256, 3, 1, 1)
        #第 8 层：池化层，2x2 的池化核，步长为 2，padding 为 0
        self.pool3 = nn.MaxPool2d(2, 2)
        #第 9 层：全连接层，4096 个节点
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        #第 10 层：全连接层，4096 个节点
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 4096)
        #第 11 层：全连接层，1000 个节点,因为一共只有1000个类
        self.fc3 = nn.Linear(4096, 1000)
    def forward(self, x):
        #第一层的激活函数是ReLU,下面类似
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x)) 
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool3(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 

import torch
input = torch.rand(1, 3, 227, 227)
model = AlexNet()
output = model(input)
print(output.shape)
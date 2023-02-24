import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        #这里采用了super()函数，这个函数的作用是调用父类
        #的一个方法，这里是调用nn.Module的__init__方法
        super(LeNet,self).__init__()
        #conv2d的三个参数分别是输入通道数，输出通道数，卷积核大小
        self.cov1 = nn.Conv2d(3, 16, 5)
        #MaxPool2d的两个参数分别是池化核的大小，步长
        self.pool1 = nn.MaxPool2d(2, 2)
        self.cov2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(slef, x):
        x = F.relu(slef.cov1(x))
        x = slef.pool1(x)
        x = F.relu(slef.cov2(x))
        x = slef.pool2(x)
        #view()函数的作用是将张量x变形成一维向量
        x = x.view(-1, 32 * 5 * 5) 
        x = F.relu(slef.fc1(x))
        x = F.relu(slef.fc2(x))
        #这里不需要激活函数，因为在交叉熵损失函数中已经包含了softmax
        x = slef.fc3(x)
        return x
            
import torch
input1 = torch.rand([32,3,32,32])
model = LeNet()
output = model(input1)
print(output.shape)            
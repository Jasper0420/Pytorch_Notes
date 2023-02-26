import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt 
import numpy as np 
#这部分代码详解见LeNet_demo\train.py
from model_ import AlexNet
#预处理
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#设置 训练集
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=preprocess)

trainloader = torch._utils.data.DataLoader(
    trainset, batch_size=36, shuffle=True, num_workers=0)
#设置测试集
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=False, transform=preprocess)

testloader = torch._utils.data.Dataloader(
    testset, batch_size=4, shuffle=False, num_workers=0)
#创建一个可迭代的test_data_iter,
test_data_iter = iter(testloader)
test_images, test_label = test_data_iter.next()

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print(' '.join('%5s' % classes[test_label[j]] for j in range(4)))

net = AlexNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.paramaters(),lr=0.001,momentum=0.9)
epoches = 100
for epoch in range(epoches):
    running_rate = 0.0

    for step, data in enumerate(trainloader, 0):
        inputs, label = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_fn(outputs, label)
        loss.backward()
        optimizer.step()
        running_rate += loss.item()
        if step % 500 == 499:
            outputs = net(test_images)
            predict_y = torch.max(outputs,1)[1]
            accuracy = (predict_y == test_label).sum().item() / test_label.size(0)
            print('[%d, %5d] loss: %.3f, accuracy: %.3f' % (epoch + 1, step + 1, running_rate / 500, accuracy))
            running_rate = 0.0

print('Finished Training')
save_path = './AlexNet.pth'
torch.save(net.state_dict(), save_path)

if __name__ == '__main__':
    main()
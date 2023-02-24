import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model import LeNet
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 定义了一个数据预处理的管道，它包括两个步骤：
# transforms.ToTensor()：将输入的 PIL 图像或者 ndarray 数组转换为 PyTorch 的张量，并且将像素值缩放到 [0,1] 范围内。
#transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))：
#对张量进行规范化操作，即减去均值(0.5)并除以标准差(0.5)。这里的均值和标准差是在 CIFAR-10 数据集上计算得到的。具体来说，这个操作将每个通道上的像素值缩放到 [-1,1] 的范围内。
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

'''调用torchvision.datasets.CIFAR10()函数创建了一个CIFAR10数据集的实例，
其中root参数指定数据集存储的路径，train参数指定是训练集还是测试集，
download参数指定是否需要下载数据集，transform参数指定对数据集中的图像进行的预处理。'''

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)

'''将上一步创建的数据集实例trainset作为第一个参数，batch_size参数指定每个batch的大小为36，
shuffle参数指定是否打乱数据集的顺序，num_workers参数指定用于数据加载的线程数。最终，将这些参数传入
torch.utils.data.Dataloader()函数中创建一个数据加载器trainloader，该加载器可以在训练神经网络时
提供方便的数据批次读取。'''

trainloader = torch.utils.data.Dataloader(
    trainset, batch_size = 36, shuffle = True, num_workers = 0)

#导入测试集原理同上，只是train和download改为False
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=False, transform=transform)

testloader = torch.utils.data.Dataloader(
    testset, batch_size = 4, shuffle = False, num_workers = 0)

'''将testloader作为参数传入iter()函数，创建一个可迭代的对象test_data_iter。然后，通过next()方法获取这个可迭代对象的下一个元素，
即一个batch的数据和标签。这个batch的数据和标签将被分别赋值给test_image和test_label两个变量。
注意，使用next()方法获取数据的过程中，如果数据集已经遍历完毕，那么将会引发StopIteration异常。'''

test_data_iter = iter(testloader)
test_image, test_label = test_data_iter.next()

classes = ('plane', 'car', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck')


def image(img):
    #在上面的代码中，使用了PyTorch中的ToTensor()函数将图片数据转换为了张量形式，并进行了归一化操作。因此，当需要将这个张量还原成原始的图像时，需要进行反归一化的操作。
    #这个函数将图像进行反向归一化（de-normalize），即将张量中的每个像素值除以2再加上0.5，以便将其值域从[-1,1]转换为[0,1]。
    img = img / 2 + 0.5
    npimg = img.numpy()
    #np.transpose(npimg, (1, 2, 0))将第一维和第二维进行交换，同时将第三维放到最后面，这样做是因为在Matplotlib中，图像的维度是(height, width, channel)。
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#这句代码的作用是将36张测试图片的标签用字符串表示出来，并按照一定格式打印出来。
print(' '.join('%5s' % classes[test_label[j]] for j in range(36)))

net = LeNet() 
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.001)
epoches = 100 
for epoch in range(epoches):
    running_loss = 0.0
    #for 循环会遍历 trainloader 中的所有数据，并将每个 batch 数据赋值给变量 data，每个 batch 的大小是 batch_size，在这个例子中是 36。每个 batch 包含输入数据 inputs 和对应的标签 labels。
    for step, data in enumerate(trainloader, 0):
        #获取输入数据inputs和标签labels。
        inputs ,labels = data
        #将模型参数的梯度清零，通常在每个batch的训练开始之前调用。
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        #在每个epoch中，每训练完500个batch，使用当前模型对test_image进行预测，并计算模型的准确率accuracy。
        if step % 500 ==499:
            with torch.no_grad():
                outputs = net(test_image)
                predict_y = torch.max(outputs, dim = 1)[1]
                accuracy = (predict_y == test_label).sum().item() / test_label.size(0)
                print('[%d, %5d] loss: %.3f, acc: %.3f' % (epoch + 1, step + 1, running_loss / 500, accuracy))
                running_loss = 0.0

print('Finished Training')    
save_path = './cifar10_lenet.pth'
torch.save(net.state_dict(), save_path)

'''如果当前脚本作为主程序运行，那么执行 main() 函数；如果当前脚本被作为模块导入其他脚本中，
那么不执行 main() 函数。这样做的目的是为了让模块导入时不会自动执行脚本中的代码，而只有在该脚本被作为主程序运行时才会执行其中的代码。'''

if __name__ == '__main__':
    main()
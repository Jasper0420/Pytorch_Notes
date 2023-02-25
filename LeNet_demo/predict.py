import torch
import torchvision.transforms as transforms
from PIL import Image
from model import LeNet

def main():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    net = LeNet()
    #从已经保存的模型参数文件中加载模型参数，并将其存储在LeNet实例对象中。
    net.load_state_dict(torch.load('LeNet.pth'))
    #使用PIL库的Image模块打开了一张名为"cat.jpg"的图片
    im = Image.open('cat.jpg')
    #将PIL Image对象im进行了预处理，使用了之前定义的transform函数，该函数将图片转换为一个3x32x32的张量，并将像素值归一化为-1到1之间。
    im = transform(im)
    #使用torch.unsqueeze函数增加了一维，将原本3x32x32的张量变为1x3x32x32的四维张量。这是因为LeNet模型的输入是一个四维张量，
    #第一维表示输入的样本数，这里为1，后面三维表示一张图片的通道数、高度和宽度。
    im = torch.unsqueeze(im, dim=0)
    #使用 torch.no_grad() 包装代码块，表示不需要计算梯度，可以加快计算速度并减少内存占用。
    with torch.no_grad():
        outputs = net(im)
        #用 torch.max() 函数沿着第1维度（即类别维度）找到预测结果中的最大值和对应索引，即预测的类别。
        predict = torch.max(outputs, dim=1)[1]
    print(classes[predict])

if __name__ == '__main__':
    main()

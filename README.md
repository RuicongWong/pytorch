# pytorch搭建并训练模型的套路

数据集下好解压后将pokemon和.py文件放在同一个文件夹直接运行即可。

pytorch搭建模型一般可分为以下几个步骤：

1. 数据预处理
2. 搭建模型
3. 训练模型

其中1、2无明显顺序之分。

## 1.搭建网络

pytorch为我们提供了非常方便的nn工具箱，我们搭建模型**只需要定义一个继承自nn.module的类并实现其init和forward方法就可**。**init方法中动态绑定成员变量，forword方法中决定数据流经这些成员变量的顺序**。下面是nn工具箱的结构示意图（来自网络，侵权删）。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201013151733472.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDcxODcxNg==,size_16,color_FFFFFF,t_70#pic_center)




接着看上图，nn.Module中的大多数Layer在functional中都有对应的函数，区别在于**Layer是继承nn.Module的类，会自动提取可学习的参数，而nn.functional更像是纯函数**，所以像卷积层、全连接层、Dropout层等因含有可学习参数，一般使用nn.Module，而激活函数、池化层可使用functional中对于的函数。下面以非常经典的Resnet18为例，用pytorch搭建一个简单的网络。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201013151852802.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDcxODcxNg==,size_16,color_FFFFFF,t_70#pic_center)


如上图（来自网络，侵权删）所示，左边是Resnet18的网络结构图，右边是Resnet50的网络结构图，大家可以模仿下面给出的Resnet18的代码搭建Resnet50。

### 构建最基本ResBlk块

如上图所示，整个网络结构图中有很多输入和输出连接到一起的块，这根连接输入和输出的线姑且称作跳接线，**实线表示输入维度和输出维度一样不需要升维，而虚线表示输入和输出维度不一样需要升维**。这个ResBlk的最基本的结构就是卷积之后批归一化重复两遍。为了处理输入输出维度不一样，**可以设置一个extra层，来进行升维**。至于这个基本块的形式参数的设置，观察网络结构图可以看出，第二个卷积的stride总是为1，而第一个卷积块的stride会因为虚实线的不同有所差异，故stride定为一个默认为1的形式参数，还会发生变化的就是输入和输出的特征层个数，因此也定为形式参数，至于padding，需要升维层为0，其他都为1，不需定为形参。下面给出代码：

```python
# 需要导入的包
import torch
from torch import nn
from torch.nn import functional as F
```

ResBlk块

```python
class ResBlk(nn.Module):

    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlk, self).__init__()
        
        # 卷积之后批归一化
        self.conv1 = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=ch_out)
        self.conv2 = nn.Conv2d(in_channels=ch_out, out_channels=ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=ch_out)

        # 若输入输出不一样则需要升维
        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(num_features=ch_out)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out

        return F.relu(out)
```

### 搭建Resnet18

有了ResBlk块我们就可以更方便搭建网络了，还是看上面的Resnet18的网络结构图，首先经过一个卷积和池化，再经过两个基本ResBlk块，循环3次接一个升维的ResBlk块后接一个基本ResBlk块，最后平均池化后加全连接得到输出。下面看代码：

```python
class ResNet18(nn.Module):

    def __init__(self, num_classes):
        super(ResNet18, self).__init__()

        self.conv = nn.Sequential( #经过一个卷积层和一个池化层
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 两个基本的ResBlk块
        for i in range(2):
            self.conv.add_module(name='ResBlk' + chr(i), module=ResBlk(ch_in=64, ch_out=64, stride=1))

        # 一个升维的ResBlk块后接一个基本ResBlk块
        self.conv.add_module(name='ResBlk2', module=ResBlk(ch_in=64, ch_out=128, stride=2))
        self.conv.add_module(name='ResBlk3', module=ResBlk(ch_in=128, ch_out=128, stride=1))

        # 一个升维的ResBlk块后接一个基本ResBlk块
        self.conv.add_module(name='ResBlk4', module=ResBlk(ch_in=128, ch_out=256, stride=2))
        self.conv.add_module(name='ResBlk5', module=ResBlk(ch_in=256, ch_out=256, stride=1))

        # 一个升维的ResBlk块后接一个基本ResBlk块
        self.conv.add_module(name='ResBlk6', module=ResBlk(ch_in=256, ch_out=512, stride=2))
        self.conv.add_module(name='ResBlk7', module=ResBlk(ch_in=512, ch_out=512, stride=1))


        self.outlayer = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.adaptive_avg_pool2d(x, [1, 1])
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)

        return x
```

至此一个简单的Resnet18的模型就搭建完了，下面做一个简单的测试：

```python
def main():

    x = torch.randn(2, 3, 224, 224) # 两张三通道(RGB)224 * 224大小的图片
    model = ResNet18(num_classes=5)
    out = model(x)
    print('ResNet18:', out.shape)

    p = sum(map(lambda p : p.numel(), model.parameters())) # map的第二个参数是一个可迭代对象，numel()获取tensor中包含多少个元素
    print('parameters size:', p)

if __name__ == "__main__":
    main()

######################################################
# 输出结果
# ResNet18: torch.Size([2, 5])
# parameters size: 11109637
######################################################

# 得到的是两张五分类的输出，说明我们网络结构没有结构上的错误
```

模型搭建就介绍到这里，下面开始介绍数据的预处理和模型的训练。

## 2.数据的预处理

这里以分类任务为例简单的介绍一下如何用pytorch来进行数据预处理。pytorch在内的框架都自带了一些简单的数据集，加载那些自带的数据集调用诸如load之类的方法就可以了，这里介绍的主要是对自有数据集的处理。

以我电脑上的一个数据集为例：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201013152013227.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDcxODcxNg==,size_16,color_FFFFFF,t_70#pic_center)


pokemon是数据集的根目录，有五个子目录，每个子目录代表着不同种类的pokemon,子目录中有若干张属于这个类别的图片。

首先导入需要的包：

```python
import torch
import os, glob
import random, csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
```

实现数据加载最好的方法就是创建一个继承自torch.utils.data.Dataset的类，并实现init，len和getitem方法，我们不妨叫这个用于数据加载的类为Pokemon，代码结构如下：

```python
class Pokemon(Dataset):
    def __init__(self):
        pass
    def __len__(self):
        pass
    def __getitem__(self, idx):
        pass
```

### init

在构造函数中，我们可以把**类名映射到0~4以方便训练**，**以及划分出训练集、验证集还有测试集**，我们需要三个形式参数，根路径，图片大小，模式，分别记为root、resize还有mode。我们用一个辅助函数load_csv来获取图片的路径，以便划分训练集、验证集还有测试集。load_csv代码如下：

```python
def load_csv(self, filename): # filename为将要保存的csv文件名

    if not os.path.exists(os.path.join(self.root, filename)):
        images = []
        # 将所有图片的路径放到images这个列表中
        for name in self.name2label.keys():
            images += glob.glob(os.path.join(self.root, name, '*.png')) #glob.glob返回所有匹配的文件路径列表
            images += glob.glob(os.path.join(self.root, name, '*.jpg'))
            images += glob.glob(os.path.join(self.root, name, '*.jpeg'))
            images += glob.glob(os.path.join(self.root, name, '*.gif'))

        random.shuffle(images)
        
        with open(os.path.join(self.root, filename), mode='w', newline='') as f:
            writer = csv.writer(f)
            for img in images: # eg. img == pokemon\pikachu\00000000.jpg
                name = img.split(os.sep)[-2] # name == pikachu
                label = self.name2label[name] # label == 3
                writer.writerow([img, label]) # pokemon\pikachu\00000000.jpg 3
            print('written into csv file:', filename)

    images, labels = [], []
    with open(os.path.join(self.root, filename)) as f:
        reader = csv.reader(f)
        for row in reader:
            img, label = row
            label = int(label) # 字符转为整型
            images.append(img)
            labels.append(label)

    assert len(images) == len(labels)

    return images, labels # 返回图片路径列表和对应的标签列表
```

通过load_csv我们可以得到图片路径列表和对应的标签列表，再根据mode划分数据集就可了，完整的init代码如下：

```python
def __init__(self, root, resize, mode):
    super(Pokemon, self).__init__()

    self.root = root
    self.resize = resize

    # 将不同类别映射到0~4
    self.name2label = {}
    for name in sorted(os.listdir(root)):  # 排序使得每次初始化时调用listdir返回的列表顺序相同
        if not os.path.isdir(os.path.join(root, name)):  # 不是目录则跳过
            continue
        self.name2label[name] = len(self.name2label.keys())  # 用长度来当作映射的值

    self.images, self.labels = self.load_csv('images')

    if mode == 'train':
        self.images = self.images[:int(0.6 * len(self.images))]
        self.labels = self.labels[:int(0.6 * len(self.labels))]
    elif mode == 'val':
        self.images = self.images[int(0.6 * len(self.labels)):int(0.8 * len(self.images))]
        self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.images))]
    else:
        self.images = self.images[int(0.8 * len(self.images)):]
        self.labels = self.labels[int(0.8 * len(self.labels)):]
```

### len

这个太简单了，直接看代码：

```python
def __len__(self):
    return len(self.images)
```

### getitem

**返回实实在在的图片和标签而不是路径和标签。**

代码如下：

```python
def __getitem__(self, key):
    img, label = self.images[key], self.labels[key]

    # 一些数据增强
    trans = transforms.Compose([
        lambda x: Image.open(x).convert('RGB'),
        transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
        transforms.RandomRotation(15),
        transforms.CenterCrop(self.resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img = trans(img)
    label = torch.tensor(label)

    return img, label
```

以上就是简单的数据预处理的全部步骤啦，要是你的数据集的结构和我上面提到的完全一样，pytorch还提供了一个api让你直接进行加载而不需要写这么多，但是如果你要是想定制，可以修改我上述的代码。api是：

```python
db = torchvision.datasets.ImageFolder(root='pokemon', transform=trans)

######################################################################
# 上面这一行代码就实现了我数据预处理这一块写的全部了，/(ㄒoㄒ)/~~，等同于我自己调用自己的代码如下：
######################################################################

db = Pokemon('pokemon', 224, 'train')
```

数据预处理这一块的最后的最后就是将数据分batch打包，就结束了，api如下：

```python
loader = DataLoader(db, batch_size=32, shuffle=True, num_workers=8)
```

## 3.训练模型

先做一些基础工作：

```python
batchsz = 32
lr = 1e-3
epochs = 100

device = torch.device('cuda') # 实例化"一块显卡"，如果没有显卡就注释掉这行和下面涉及cuda这个词的代码。

train_db = Pokemon('pokemon', 224, mode='train')
val_db = Pokemon('pokemon', 224, mode='val')
test_db = Pokemon('pokemon', 224, mode='test')
train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=8)
val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=4)
test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=4)
```

### 训练部分

```python
def train():

    # 创建模型并搬到显卡上
    model = ResNet18(5).to(device)
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 创建损失函数
    criterion = nn.CrossEntropyLoss()

    best_acc, best_epoch = 0, 0
    global_step = 0 # 一个epoch 共global_step次更新参数
    
    # 开始训练
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            
            x, y = x.to(device), y.to(device) # 搬到显卡上
            
            # 前向传播，并计算损失
            logits = model(x)
            loss = criterion(logits, y)

            # 梯度清零，反向传播，更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            global_step += 1
            
	return  best_acc, best_epoch
```

### 验证测试部分

```python
       if epoch % 2 == 0: # 每两个epoch进行一次验证
                val_acc = evaluate(model, val_loader)
            
                if val_acc > best_acc:
                    
                    best_epoch = epoch
                    best_acc = val_acc

                    torch.save(model.state_dict(), 'best.mdl') # 将最好的模型保存（后缀随便）
```

evaluate函数如下：

```python
def evaluate(model, loader):
    
    correct = 0
    total = len(loader.dataset)

    for x, y in loader:
        
        x, y = x.to(device), y.to(device)
        
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
            
        correct = torch.eq(pred, y).sum().float().item()

    return correct / total
```

至此我们就可以开始训练模型了：

```python
def main():
	best_acc, best_epoch = train()
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best.mdl'))
    print('loaded from ckpt')

    # 测试集上测试
    test_acc = evaluate(model, test_loader)
    print('test acc:', test_acc)
```

数据集：
链接：https://pan.baidu.com/s/1sZYuyTHYzPmTcgg1B9DvbA 提取码：duwb （数据集来自网络，侵删）



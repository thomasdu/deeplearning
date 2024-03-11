#step1 检查GPU是否可用
import torch
torch.cuda.is_available()

#step2 数据加载
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader

#定义训练设备
device = torch.device("cuda")

#准备数据集
train_data = torchvision.datasets.CIFAR10(root="../data",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.CIFAR10(root="../data",train=False,transform=torchvision.transforms.ToTensor(),download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

#打印数据集长度
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

#利用DataLoader来加载数据
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

#step3 模型编写
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self,x):
        x = self.model(x)
        return x
model = Model()
model = model.to(device)

#损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
#优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#step4 开始训练
#设置训练网络的一些参数
#记录训练的次数
total_train_step = 0
#记录测试的次数
total_test_step = 0
#训练的轮数
epoch = 30

#添加tensorboard
writer = SummaryWriter("../logs_train")

for i in range(epoch):
    print("-----第{}轮训练开始-----".format(i+1))

    #训练步骤开始
    model.train()
    for data in train_dataloader:
        #取数据，指定设备
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        #正向传播计算输出
        outputs = model(imgs)
        #计算损失值
        loss = loss_fn(outputs, targets)

        #优化器优化模型
        optimizer.zero_grad()
        #反向传播更新权重
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    #测试步骤开始
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            #加载数据计算预测值
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            #计算损失值
            loss = loss_fn(outputs,targets)
            #测试数据损失值之和
            total_test_loss = total_test_loss + loss.item()
            #预测正确的数量计算
            total_accuracy = total_accuracy + (outputs.argmax(1) == targets).sum()

    print("整体测试机上的loss：{}".format(total_test_loss))
    print("整体测试机上的正确率:{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_step)
    total_test_step = total_test_step + 1

    torch.save(model,"./model/tudui_{}.pth".format(i))
    print("模型已保存")

writer.close()

from PIL import Image

image_path = "ship.webp"
image = Image.open(image_path)
print(image)

image = image.convert("RGB")
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)), torchvision.transforms.ToTensor()])
image = transform(image)
print(image.shape)

model = torch.load("tudui_26.pth", map_location=torch.device("cpu"))
print(model)
image=torch.reshape(image,(1,3,32,32))
model.eval()

with torch.no_grad():
    output = model(image)
print(output)

print(output.argmax(1))

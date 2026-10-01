from pathlib import Path
from torchvision import datasets,models
from torch.utils.data import DataLoader
import torch 
from torch import nn
from torch.utils.data import random_split
#1.找到数据集
project_dir = Path(__file__).resolve().parent
data_dir = project_dir / "data"/"cats_and_dogs_filtered"

#2.使用预训练模型对应的图片预处理
weights = models.ResNet18_Weights.DEFAULT
preprocess = weights.transforms()

#3按文件夹读取图片和标签
train_dataset = datasets.ImageFolder(
    root = data_dir /"train",
    transform = preprocess,
)

#4.每一批取16张
train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = 16,
    shuffle = True,
    num_workers=0,
)

#5.取一批数据，检查读取结果
images , labels = next(iter(train_loader))

print("类别编号：", train_dataset.class_to_idx)
print("训练图片数量：", len(train_dataset))
print("图片形状：", images.shape)
print("标签形状：", labels.shape)
print("这一批的标签：", labels)

#6.加载预训练模型
model = models.resnet18(weights=weights)

#7.冻结已有参数
for param in model.parameters():
    param.requires_grad = False

#8.换成新二分类头.fc:全连接层
model.fc = nn.Linear(model.fc.in_features,2)

#9.模型和图片放在同一个设备上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
images = images.to(device)
labels =  labels.to(device)

#10.检查输出形状
model.eval()
with torch.no_grad():
    outputs = model(images)

print("运行设备：", device)
print("新分类头：",model.fc)
print("分头能否计算梯度：",model.fc.weight.requires_grad)
print("模型输出形状：",outputs.shape)

#11.配置损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(),lr =0.001)

#12.原有部分保持评估模式，新分类头进入训练模式
model.eval()
model.fc.train()

#保存原来分类头的权重（实验对比，观察权重变化）
weights_before = model.fc.weight.detach().clone()

#13.遍历所有训练图片，完成一个Epoch
total_loss =0
total_correct =0
total_samples =0

for images ,labels in train_loader:
    images = images.to(device)
    labels = labels.to(device)

    #训练模型主体
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs,labels)
    loss.backward()
    optimizer.step()

    #累计这一批结果
    batch_size = labels.size(0)
    total_loss +=loss.item()*batch_size#loss是平均损失

    predicted = outputs.argmax(dim=1)
    total_correct +=(predicted==labels).sum().item()
    total_samples+=batch_size

#14.汇总整轮结果
print("训练batch数量：",len(train_loader))
print("训练图片数量：",total_samples)
print("训练平均loss：",total_loss/total_samples)
print("训练准确率：",f"{total_correct/total_samples:.2%}")

#15.读取原验证集
heldout_dataset = datasets.ImageFolder(
    root = data_dir /'validation',
    transform = preprocess,
)

#固定随机划分：800张验证 ，200张测试
val_dataset , test_dataset = random_split(
    heldout_dataset,
    [800,200],
    generator = torch.Generator().manual_seed(42),
)

val_loader = DataLoader(
    val_dataset,
    batch_size= 16,
    shuffle = False,
    num_workers = 0,
)

# 16. 验证：只计算结果
model.eval()

val_loss = 0.0
val_correct = 0
val_samples = 0

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        val_loss += loss.item() * batch_size
        val_correct += (outputs.argmax(dim=1) == labels).sum().item()
        val_samples += batch_size

print("验证图片数量：", val_samples)
print("验证平均 Loss：", val_loss / val_samples)
print("验证准确率：", f"{val_correct / val_samples:.2%}")
print("留待最终测试的图片数量：", len(test_dataset))

# 17. 保存这一轮的基线模型
save_path = Path(__file__).resolve().parent / "cats_dogs_baseline.pth"
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "class_to_idx": train_dataset.class_to_idx,
        "split_seed": 42,
    },
    save_path,
)
print("模型已保存到：", save_path)
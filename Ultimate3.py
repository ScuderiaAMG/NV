import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
 
# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
# 超参数
batch_size = 64
num_epochs = 5
learning_rate = 1e-4
num_steps = 1000  # 扩散步骤
beta = torch.linspace(0.0001, 0.02, num_steps).to(device)  # 扩散系数
 
# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
 
# 模型定义
class SimpleDiffusionModel(nn.Module):
    def __init__(self):
        super(SimpleDiffusionModel, self).__init__()
        self.fc = nn.Linear(784, 784)
 
    def forward(self, x):
        return torch.sigmoid(self.fc(x))
 
# 损失函数和优化器
model = SimpleDiffusionModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
 
# 扩散过程
def q_sample(x_0, t):
    noise = torch.randn_like(x_0).to(device)
    return torch.sqrt(1 - beta[t]) * x_0 + torch.sqrt(beta[t]) * noise
 
# 反扩散过程
def p_sample(x_t, t):
    noise = torch.randn_like(x_t).to(device)
    x_0 = model(x_t)
    return x_0 + torch.sqrt(beta[t]) * noise
 
# 训练模型
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_loader):
        images = images.view(-1, 784).to(device)
        t = torch.randint(0, num_steps, (images.shape[0],)).to(device)  # 随机选择扩散步骤
 
        # 扩散过程
        x_t = q_sample(images, t)
 
        # 反扩散过程
        x_0 = p_sample(x_t, t)
 
        # 计算损失
        loss = criterion(x_0, images)
 
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
 
# 生成图像
with torch.no_grad():
    x_t = torch.randn(batch_size, 784).to(device)
    for t in range(num_steps-1, -1, -1):
        x_t = p_sample(x_t, torch.tensor([t]*batch_size).to(device))
    generated_images = x_t.view(batch_size, 1, 28, 28).cpu()
 
# 显示生成的图像
fig, axs = plt.subplots(1, 10, figsize=(10, 1))
for i in range(10):
    axs[i].imshow(generated_images[i][0], cmap='gray')
    axs[i].axis('off')
plt.show()
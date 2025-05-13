import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import random

# -------------------- 参数设置 --------------------
T = 10                     # 去噪迭代的步数
embed_dim = 10            # 嵌入向量维度（对应10个类别）
batch_size = 128          # 每批训练样本数量
lr = 1e-3                 # 学习率
epochs = 50              # 训练轮数

# alpha 是从1线性递减到0.1的序列，用于控制加噪与还原的强度
alpha = torch.linspace(1.0, 0.1, T)

# -------------------- 定义去噪 MLP --------------------
class DenoisingMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(128 + embed_dim, 256),  # 输入为图像特征 + 噪声嵌入
            nn.ReLU(),
            nn.Linear(256, embed_dim)         # 输出一个嵌入向量
        )
    def forward(self, x_features, z_t):
        # 拼接图像特征和带噪向量，输入MLP
        return self.mlp(torch.cat([x_features, z_t], dim=1))

# -------------------- 定义 CNN 特征提取器 --------------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),   # 输出通道32, kernel=3x3
            nn.ReLU(),
            nn.MaxPool2d(2),          # 降采样
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),             # 展平为一维向量
            nn.Linear(1600, 128)      # 输出为128维的图像特征
        )
    def forward(self, x):
        return self.features(x)

# -------------------- 模型与优化器初始化 --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cnn = CNN().to(device)                                  # CNN 特征提取网络
mlps = nn.ModuleList([DenoisingMLP().to(device) for _ in range(T)])  # T个MLP去噪模块
optimizers = [optim.Adam(mlp.parameters(), lr=lr) for mlp in mlps]   # 对每个MLP使用一个优化器
cnn_optimizer = optim.Adam(cnn.parameters(), lr=lr)     # CNN的优化器

# -------------------- 数据加载 --------------------
transform = transforms.ToTensor()
train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# -------------------- 开始训练 --------------------
for epoch in range(epochs):
    cnn.train()
    for mlp in mlps:
        mlp.train()

    epoch_loss = 0.0
    batch_count = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        # 构造 one-hot 目标嵌入向量 u_y，大小为 (batch_size, embed_dim)
        u_y = torch.zeros(x.size(0), embed_dim, device=device).scatter_(1, y.unsqueeze(1), 1.0)

        # 逐步加入高斯噪声，得到 T 步的 z_t
        z = [u_y]
        for t in range(1, T + 1):
            eps = torch.randn_like(u_y)  # 标准正态噪声
            z_t = torch.sqrt(alpha[t-1]) * z[-1] + torch.sqrt(1 - alpha[t-1]) * eps
            z.append(z_t)

        x_features = cnn(x)  # 提取图像特征 (batch_size, 128)

        losses = []
        for t in range(T):
            # 使用第 t 个 MLP 去噪当前 z_t+1，预测目标向量 u_hat
            u_hat = mlps[t](x_features, z[t+1].detach())
            # 均方误差作为损失
            losses.append(torch.mean((u_hat - u_y) ** 2))

        total_loss = sum(losses)

        # 反向传播与优化
        cnn_optimizer.zero_grad()
        for opt in optimizers:
            opt.zero_grad()
        total_loss.backward()
        cnn_optimizer.step()
        for opt in optimizers:
            opt.step()

        epoch_loss += total_loss.item()
        batch_count += 1

    # 输出每轮训练损失
    print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {epoch_loss / batch_count:.4f}")

    # -------------------- 测试准确率 --------------------
    cnn.eval()
    for mlp in mlps:
        mlp.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x_features = cnn(x)
            z_t = torch.zeros(x.size(0), embed_dim, device=device)  # 初始向量设为全0

            # 倒序使用 MLP 模块逐步去噪，恢复预测的 one-hot 向量
            for t in reversed(range(T)):
                z_t = mlps[t](x_features, z_t)

            # 取最大值的位置作为预测类别
            pred = torch.argmax(z_t, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = correct / total
    print(f"Test Accuracy: {acc * 100:.2f}%")

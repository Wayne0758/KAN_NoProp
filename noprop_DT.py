import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# ----------------------------- Config -----------------------------
T = 10
eta = 0.1
embed_dim = 20
batch_size = 128
num_classes = 10
image_size = 28 * 28

class LabelEmbedding(nn.Module):
    def __init__(self, num_classes, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)
    def forward(self, y):
        return self.embedding(y)

def cosine_schedule(T, s=0.008):
    """
    Create a cosine noise schedule for T diffusion steps.
    s is a small offset to prevent alpha_bar from being 0.
    Returns a NumPy array 'betas' of shape [T].
    """
    steps = np.linspace(0, T, T+1)  # t from 0 to T
    
    # Calculate alpha_bar using the cosine formula
    # alpha_bar(t) = (cos((t/T + s) / (1+s) * (pi/2))^2) / (cos(s / (1+s) * (pi/2))^2)
    # The '1e-7' guards against numerical issues at boundaries.
    alphas_bar = (
        np.cos((steps / T + s) / (1 + s) * np.pi / 2) ** 2
        / np.cos(s / (1 + s) * np.pi / 2) ** 2
    )
    
    return alphas_bar

def get_alpha_bar(T):
    """生成 alpha_bar[0], ..., alpha_bar[T]，共 T+1 个值。"""
    return [cosine_schedule(t) for t in range(T + 1)]

def sample_noise(mean, var):
    """给定 N 维的 mean 和标量 var，为 batch 中每个样本加上高斯噪声。"""
    std = torch.sqrt(torch.tensor(var, device=mean.device))
    noise = torch.randn_like(mean)
    return mean + std * noise

class DenoiseBlock(nn.Module):
    """每个离散步 t 对应一个 block。"""
    def __init__(self, embed_dim):
        super().__init__()
        self.x_embed = nn.Sequential(
            nn.Linear(image_size, embed_dim),
            nn.ReLU()
        )
        self.z_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x, z):
        x = x.view(x.size(0), -1)
        x_feat = self.x_embed(x)
        z_feat = self.z_embed(z)
        return self.fc(torch.cat([x_feat, z_feat], dim=1))

class OutputClassifier(nn.Module):
    """最终输出层，用来估计 p(y | z_T)。"""
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)
    def forward(self, z):
        return self.fc(z)

def kl_divergence(mu_q, var_q, mu_p, var_p, eps=1e-5):
    """
    计算多维高斯之间的 KL 散度：
    DKL(q || p) = 0.5 * ∑ [log(var_p/var_q) + (var_q + (mu_q - mu_p)^2)/var_p - 1]
    """
    var_q = var_q + eps
    var_p = var_p + eps
    return 0.5 * torch.sum(
        torch.log(var_p / var_q) + (var_q + (mu_q - mu_p) ** 2) / var_p - 1,
        dim=1
    )

def evaluate_noprop_dt(blocks, output_head, data_loader, device, T):
    """
    对离散时间的 NoProp 模型做推理并统计准确率。
    1) 初始化 z_0 = N(0, I)
    2) 依次通过 blocks[0..T-1]: z_{t+1} = blocks[t](x, z_t)
    3) 在 z_T 上用 output_head 做分类
    返回正确数 / 总数。
    """
    for b in blocks:
        b.eval()
    output_head.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)

            # 初始 z_0 ~ N(0, I)
            z = torch.randn(batch_size, embed_dim, device=device)
            # 依次经过 T 个 block
            for t in range(T):
                z = blocks[t](x, z)

            # 输出层做分类
            logits = output_head(z)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += batch_size

    return correct / total

def train_noprop_dt():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([transforms.ToTensor()])

    # MNIST 数据集
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # 初始化网络
    WEmbed = LabelEmbedding(num_classes, embed_dim).to(device)
    blocks = [DenoiseBlock(embed_dim).to(device) for _ in range(T)]
    output_head = OutputClassifier(embed_dim, num_classes).to(device)

    optim = torch.optim.AdamW(
        list(WEmbed.parameters()) +
        list(output_head.parameters()) +
        [p for b in blocks for p in b.parameters()],
        lr=5e-4
    )

    alpha_bar = get_alpha_bar(T)
    epochs = 20

    for epoch in range(1, epochs + 1):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            uy = WEmbed(y)

            for t in range(T):

                mean_t = torch.sqrt(torch.tensor(alpha_bar[t], device=device)) * uy
                var_t = 1 - alpha_bar[t]
                z_tm1 = sample_noise(mean_t, var_t)
                # 1st part of loss function
                logits = blocks[t](x, z_tm1)
                log_probs = nn.functional.log_softmax(logits, dim=-1)
                neg_log_likelihood = -log_probs[range(len(y)), y]
                loss_cls = neg_log_likelihood.mean()
                # L2 去噪项
                if t == 0:
                    l2_loss = 0.0
                else:
                    u_pred = blocks[t - 1](x, z_tm1)
                    l2_loss = ((u_pred - uy) ** 2).sum(dim=1).mean()

                # KL 散度 (t=1 时)
                if t == 0:
                    kl_loss = 0.0
                else:
                    mu0 = torch.sqrt(torch.tensor(alpha_bar[0], device=device)) * uy
                    var0 = 1 - alpha_bar[0]
                    kl_vals = kl_divergence(mu0, var0, torch.zeros_like(mu0), torch.ones_like(mu0))
                    kl_loss = kl_vals.mean() * T

                # SNR 差值
                if t > 0:
                    snr_cur = alpha_bar[t - 1] / max(1e-5, 1 - alpha_bar[t - 1])
                else:
                    snr_cur = 0.0
                snr_next = alpha_bar[t] / max(1e-5, 1 - alpha_bar[t])
                delta_snr = max(snr_cur - snr_next, 0.0)

                l2_term = 0.5 * eta * T * delta_snr * l2_loss

                total_loss = loss_cls + kl_loss + l2_term

                optim.zero_grad()
                total_loss.backward()
                optim.step()

        # ---- 每个 epoch 结束后，分别计算“训练集”与“测试集”准确率 ----
        train_acc = evaluate_noprop_dt(blocks, output_head, train_loader, device, T)
        test_acc = evaluate_noprop_dt(blocks, output_head, test_loader, device, T)

        print(
            f"Epoch {epoch}, "
            f"Example total_loss = {total_loss.item():.4f}, "
            f"Train Acc = {train_acc*100:.2f}%, "
            f"Test Acc = {test_acc*100:.2f}%"
        )

if __name__ == "__main__":
    train_noprop_dt()
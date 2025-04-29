# NoProp.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from kan import KAN

# Assuming your CNN, DenoisingKAN, KAN etc are already imported or defined elsewhere
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Hyperparameters
T = 10  # Diffusion steps
embed_dim = 10  # Label embedding dimension(No. of Classes)
batch_size = 128
lr = 0.001
epochs = 50

# Noise schedule (linear)
alpha = torch.linspace(1.0, 0.1, T)  # α_t from 1.0 → 0.1

class DenoisingKAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.kan = nn.Sequential(
            KAN([128 + embed_dim, 256]),
            KAN([256, embed_dim])        # Output: denoised label
        )

    def forward(self, x_features, z_t):
        combined = torch.cat([x_features, z_t], dim=1)
        return self.kan(combined)
    
# CNN for image features
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1600, 128)  # MNIST: 28x28 → 1600-dim
        )

    def forward(self, x):
        return self.features(x)
    
# Create model and optimizer
cnn = CNN().to(device)
kans = nn.ModuleList([DenoisingKAN().to(device) for _ in range(T)])
optimizers = [
    optim.LBFGS(
        kan.parameters(), 
        lr=1.0,         # LBFGS usually needs a larger LR
        max_iter=20,    # How many inner optimization steps
        history_size=10
    ) 
    for kan in kans
]


# Define your train_loader here
# train_loader = ...

# Load MNIST
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(epochs):
    epoch_loss = 0.0
    batch_count = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        current_batch_size = x.shape[0]
        u_y = torch.zeros(current_batch_size, embed_dim, device=device).scatter_(1, y.unsqueeze(1), 1)

        # Forward diffusion(Adding Noise for each 'T')
        z = [u_y]
        for t in range(1, T + 1):
            eps = torch.randn_like(u_y)
            z_t = torch.sqrt(alpha[t-1]) * z[-1] + torch.sqrt(1 - alpha[t-1]) * eps
            z.append(z_t)

        x_features = cnn(x)
        batch_loss = 0.0

        for t in range(T):
            opt = optimizers[t]
            kan = kans[t]
            z_input = z[t+1].detach()

            def closure():
                opt.zero_grad()
                u_hat = kan(x_features, z_input)
                loss = torch.mean((u_hat - u_y) ** 2)
                loss.backward(retain_graph=True)
                return loss

            loss = opt.step(closure)
            batch_loss += loss.item()

        epoch_loss += batch_loss
        batch_count += 1

    avg_loss = epoch_loss / batch_count
    print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")

# Final message
print("Training complete!")

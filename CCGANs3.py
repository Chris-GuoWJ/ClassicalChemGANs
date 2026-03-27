import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def generate_3d_data(n_samples=2000, noise=0.5, seed=42):
    x = np.random.uniform(-10, 10, n_samples)
    y = np.random.uniform(-10, 10, n_samples)
    z = np.random.uniform(-10, 10, n_samples)
    # 目标：非线性组合 + 噪声
    target = (0.8*x + 0.3*y - 0.5*z + 
              0.02*x*y + 0.01*np.sin(z) + 
              0.1*x*z + np.random.normal(0, noise, n_samples))
    X = np.column_stack([x, y, z])  # (n_samples, 3)
    Y = target.reshape(-1, 1)        # (n_samples, 1)
    return X, Y


# X_raw, Y_raw = generate_3d_data(2000)
from read import DataLoad

data_loader = DataLoad()
X_raw, Y_raw = data_loader.load()

scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_scaled = scaler_X.fit_transform(X_raw)
Y_scaled = scaler_Y.fit_transform(Y_raw)

X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y_scaled, test_size=0.2, random_state=42
)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)
Y_test_t  = torch.tensor(Y_test, dtype=torch.float32)

class Generator(nn.Module):
    def __init__(self, input_dim=3, hidden_dims=[64, 128, 256, 512], output_dim=1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, hidden_dims=[512, 256, 128, 64]):
        super().__init__()
        total_input_dim = input_dim + output_dim
        layers = []
        prev_dim = total_input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.LeakyReLU(0.2))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        d_input = torch.cat([x, y], dim=1)
        return self.net(d_input)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

G = Generator().to(device)
D = Discriminator().to(device)

lr_G = 0.001
lr_D = 0.0001
optimizer_G = optim.Adam(G.parameters(), lr=lr_G, weight_decay=0)
optimizer_D = optim.Adam(D.parameters(), lr=lr_D, weight_decay=1e-8)

criterion = nn.BCELoss()

X_train_t = X_train_t.to(device)
Y_train_t = Y_train_t.to(device)
X_test_t  = X_test_t.to(device)
Y_test_t  = Y_test_t.to(device)

epochs = 5000
batch_size = 128
n_batches = len(X_train_t) // batch_size

G_losses = []
D_losses = []
test_rmse_list = []

best_test_rmse = float('inf')

for epoch in range(epochs):
    perm = torch.randperm(len(X_train_t))
    X_train_t = X_train_t[perm]
    Y_train_t = Y_train_t[perm]

    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        x_batch = X_train_t[start:end]
        y_real = Y_train_t[start:end]

        real_labels = torch.ones(x_batch.size(0), 1, device=device)
        fake_labels = torch.zeros(x_batch.size(0), 1, device=device)

        y_fake = G(x_batch)

        d_real = D(x_batch, y_real)
        loss_D_real = criterion(d_real, real_labels)

        d_fake = D(x_batch, y_fake.detach())
        loss_D_fake = criterion(d_fake, fake_labels)
        loss_D = (loss_D_real + loss_D_fake) / 2

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        y_fake = G(x_batch)
        d_fake_for_G = D(x_batch, y_fake)
        loss_G = criterion(d_fake_for_G, real_labels)

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        G_losses.append(loss_G.item())
        D_losses.append(loss_D.item())

    G.eval()
    with torch.no_grad():
        y_test_pred = G(X_test_t)
        test_mse = torch.mean((y_test_pred - Y_test_t) ** 2)
        test_rmse_std = torch.sqrt(test_mse).item()
        test_rmse_raw = test_rmse_std * scaler_Y.scale_[0]
        test_rmse_list.append(test_rmse_raw)

    if test_rmse_raw < best_test_rmse:
        best_test_rmse = test_rmse_raw
        torch.save(G.state_dict(), 'best_generator.pth')

    G.train()

    if (epoch+1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}]  D_loss: {loss_D.item():.4f}  G_loss: {loss_G.item():.4f}  Test RMSE: {test_rmse_raw:.4f}")

G.load_state_dict(torch.load('best_generator.pth'))
G.eval()

with torch.no_grad():
    y_train_pred = G(X_train_t)
    y_test_pred  = G(X_test_t)


Y_train_true_raw = scaler_Y.inverse_transform(Y_train_t.cpu().numpy())
Y_train_pred_raw = scaler_Y.inverse_transform(y_train_pred.cpu().numpy())
Y_test_true_raw  = scaler_Y.inverse_transform(Y_test_t.cpu().numpy())
Y_test_pred_raw  = scaler_Y.inverse_transform(y_test_pred.cpu().numpy())

from sklearn.metrics import mean_squared_error
train_rmse_raw = np.sqrt(mean_squared_error(Y_train_true_raw, Y_train_pred_raw))
test_rmse_raw = np.sqrt(mean_squared_error(Y_test_true_raw, Y_test_pred_raw))

print(f"\nTrain RMSE (original scale): {train_rmse_raw:.4f}")
print(f"Test RMSE  (original scale): {test_rmse_raw:.4f}")

from sklearn.metrics import r2_score, mean_squared_error
r2_train = r2_score(Y_train_true_raw, Y_train_pred_raw)
r2_test  = r2_score(Y_test_true_raw, Y_test_pred_raw)
rmse_train = np.sqrt(mean_squared_error(Y_train_true_raw, Y_train_pred_raw))
rmse_test  = np.sqrt(mean_squared_error(Y_test_true_raw, Y_test_pred_raw))

print("\n=== Final Evaluation (original scale) ===")
print(f"Train R² : {r2_train:.4f}, Train RMSE: {rmse_train:.4f}")
print(f"Test  R² : {r2_test:.4f}, Test  RMSE: {rmse_test:.4f}")

plt.figure(figsize=(15,5))


plt.subplot(1,3,1)
plt.plot(G_losses, label='Generator Loss')
plt.plot(D_losses, label='Discriminator Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.legend()
plt.grid(True)

plt.subplot(1,3,2)
plt.scatter(Y_test_true_raw, Y_test_pred_raw, alpha=0.5, s=10)
plt.plot([Y_test_true_raw.min(), Y_test_true_raw.max()],
         [Y_test_true_raw.min(), Y_test_true_raw.max()], 'r--', lw=2)
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.title(f'Test Set (R² = {r2_test:.4f})')
plt.grid(True)

plt.subplot(1,3,3)
plt.plot(test_rmse_list)
plt.xlabel('Epoch')
plt.ylabel('Test RMSE (original scale)')
plt.title('Test RMSE during training')
plt.grid(True)

plt.tight_layout()
plt.show()
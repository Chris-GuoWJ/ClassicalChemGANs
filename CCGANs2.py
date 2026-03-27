import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

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

class DiscriminatorNet(nn.Module):
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = 4
        n_out = 1
        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            nn.Linear(128, n_out),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

class GeneratorNet(nn.Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 3
        n_out = 1
        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )
        self.out = nn.Sequential(
            nn.Linear(1024, n_out)
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

discriminator = DiscriminatorNet().to(device)
generator = GeneratorNet().to(device)

d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

criterion = nn.BCELoss()

def real_data_target(size):
    """真实数据标签 = 1"""
    return torch.ones(size, 1, device=device)

def fake_data_target(size):
    """伪造数据标签 = 0"""
    return torch.zeros(size, 1, device=device)

batch_size = 128
num_epochs = 5000
d_steps = 1


dataset = torch.utils.data.TensorDataset(X_train_t, Y_train_t)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


best_test_rmse = float('inf')
patience = 50
patience_counter = 0


train_rmse_list = []
test_rmse_list = []


for epoch in range(num_epochs):
    for real_data, real_output in data_loader:
        real_data = real_data.to(device)
        real_output = real_output.to(device)

        fake_output = generator(real_data).detach()

        d_input_real = torch.cat([real_data, real_output], dim=1)
        d_input_fake = torch.cat([real_data, fake_output], dim=1)

        d_pred_real = discriminator(d_input_real)
        d_pred_fake = discriminator(d_input_fake)

        d_loss_real = criterion(d_pred_real, real_data_target(real_data.size(0)))
        d_loss_fake = criterion(d_pred_fake, fake_data_target(real_data.size(0)))
        d_loss = (d_loss_real + d_loss_fake) / 2

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        fake_output = generator(real_data)
        d_input_fake = torch.cat([real_data, fake_output], dim=1)
        d_pred_fake = discriminator(d_input_fake)

        g_loss = criterion(d_pred_fake, real_data_target(real_data.size(0)))

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()


    generator.eval()
    with torch.no_grad():
        y_train_pred = generator(X_train_t.to(device))
        train_mse = torch.mean((y_train_pred - Y_train_t.to(device)) ** 2)
        train_rmse_std = torch.sqrt(train_mse).item()
        train_rmse_raw = train_rmse_std * scaler_Y.scale_[0]

        y_test_pred = generator(X_test_t.to(device))
        test_mse = torch.mean((y_test_pred - Y_test_t.to(device)) ** 2)
        test_rmse_std = torch.sqrt(test_mse).item()
        test_rmse_raw = test_rmse_std * scaler_Y.scale_[0]

        train_rmse_list.append(train_rmse_raw)
        test_rmse_list.append(test_rmse_raw)

    if test_rmse_raw < best_test_rmse:
        best_test_rmse = test_rmse_raw
        torch.save(generator.state_dict(), 'best_generator.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    generator.train()

    if (epoch+1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}]  D_loss: {d_loss.item():.4f}  G_loss: {g_loss.item():.4f}  "
              f"Train RMSE: {train_rmse_raw:.6f}  Test RMSE: {test_rmse_raw:.6f}")

generator.load_state_dict(torch.load('best_generator.pth'))
generator.eval()

with torch.no_grad():
    y_train_pred = generator(X_train_t.to(device)).cpu().numpy()
    y_test_pred = generator(X_test_t.to(device)).cpu().numpy()

Y_train_true_raw = scaler_Y.inverse_transform(Y_train_t.numpy())
Y_train_pred_raw = scaler_Y.inverse_transform(y_train_pred)
Y_test_true_raw = scaler_Y.inverse_transform(Y_test_t.numpy())
Y_test_pred_raw = scaler_Y.inverse_transform(y_test_pred)

r2_train = r2_score(Y_train_true_raw, Y_train_pred_raw)
r2_test = r2_score(Y_test_true_raw, Y_test_pred_raw)
rmse_train = np.sqrt(mean_squared_error(Y_train_true_raw, Y_train_pred_raw))
rmse_test = np.sqrt(mean_squared_error(Y_test_true_raw, Y_test_pred_raw))

print("\n=== Final Evaluation ===")
print(f"Train R² : {r2_train:.6f}, Train RMSE: {rmse_train:.6f}")
print(f"Test  R² : {r2_test:.6f}, Test  RMSE: {rmse_test:.6f}")


plt.figure(figsize=(15,5))


plt.subplot(1,3,1)
plt.scatter(Y_test_true_raw, Y_test_pred_raw, alpha=0.5, s=10)
plt.plot([Y_test_true_raw.min(), Y_test_true_raw.max()],
         [Y_test_true_raw.min(), Y_test_true_raw.max()], 'r--', lw=2)
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.title(f'Test Set (R² = {r2_test:.4f})')
plt.grid(True)

plt.subplot(1,3,2)
plt.plot(train_rmse_list, label='Train RMSE')
plt.plot(test_rmse_list, label='Test RMSE')
plt.xlabel('Epoch')
plt.ylabel('RMSE (original scale)')
plt.title('RMSE during training')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

'''
# 预测新数据点
def predict_new_points(generator, new_points, scaler_X, scaler_Y):
    generator.eval()
    if new_points.ndim == 1:
        new_points = new_points.reshape(1, -1)
    new_scaled = scaler_X.transform(new_points)
    new_tensor = torch.tensor(new_scaled, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred_scaled = generator(new_tensor).cpu().numpy()
    pred = scaler_Y.inverse_transform(pred_scaled)
    return pred
''''''
# 测试
new_data = np.array([[2.5, -1.3, 4.7], [0.0, 0.0, 0.0]])
predictions = predict_new_points(generator, new_data, scaler_X, scaler_Y)
print("\nPredictions for new points:")
for pt, val in zip(new_data, predictions):
    print(f"  {pt} -> {val[0]:.6f}")
'''
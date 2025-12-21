# ***********************
# Chạy EKF với NF_model
# ***********************
import torch
from torch import nn
from torch.autograd.functional import jacobian
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
import time

class RealNVPNode(nn.Module):
    def __init__(self, mask, hidden_size, dropout_p=0.2):  # thêm dropout_p
        super(RealNVPNode, self).__init__()
        self.dim = len(mask)
        self.mask = nn.Parameter(mask, requires_grad=False)

        self.s_func = nn.Sequential(
            nn.Linear(in_features=self.dim, out_features=hidden_size), nn.Tanh(), nn.Dropout(dropout_p),
            nn.Linear(in_features=hidden_size, out_features=hidden_size), nn.Tanh(), nn.Dropout(dropout_p),
            nn.Linear(in_features=hidden_size, out_features=hidden_size), nn.Tanh(), nn.Dropout(dropout_p),
            nn.Linear(in_features=hidden_size, out_features=self.dim)
        )

        self.scale = nn.Parameter(torch.zeros(self.dim))

        self.t_func = nn.Sequential(
            nn.Linear(in_features=self.dim, out_features=hidden_size), nn.Tanh(), nn.Dropout(dropout_p),
            nn.Linear(in_features=hidden_size, out_features=hidden_size), nn.Tanh(), nn.Dropout(dropout_p),
            nn.Linear(in_features=hidden_size, out_features=hidden_size), nn.Tanh(), nn.Dropout(dropout_p),
            nn.Linear(in_features=hidden_size, out_features=self.dim)
        )

    def forward(self, x):
        x_mask = x * self.mask
        s = self.s_func(x_mask) * self.scale
        t = self.t_func(x_mask)

        y = x_mask + (1 - self.mask) * (x * torch.exp(s) + t)
        log_det_jac = ((1 - self.mask) * s).sum(-1)
        return y

    def inverse(self, y):
        y_mask = y * self.mask
        s = self.s_func(y_mask) * self.scale
        t = self.t_func(y_mask)

        x = y_mask + (1 - self.mask) * (y - t) * torch.exp(-s)
        inv_log_det_jac = ((1 - self.mask) * -s).sum(-1)
        return x


class RealNVP(nn.Module):
    def __init__(self, input, masks, hidden_size, input_dim=2, output_dim=4, dropout_p=0.2):  # thêm dropout_p
        super(RealNVP, self).__init__()

        self.output_dim = output_dim
        self.hidden_size = hidden_size

        # Linear transformation to upscale input (2x1 → 4x1)
        self.upscale_net = nn.Linear(input_dim, output_dim)

        self.masks = masks
        self.layers = nn.ModuleList([
            RealNVPNode(mask, self.hidden_size, dropout_p=dropout_p) for mask in self.masks
        ])

    def forward(self, input):
        upscale = self.upscale_net(input)
        for layer in self.layers:
            upscale = layer.forward(upscale)  # Apply each RealNVPNode sequentially
        return upscale

    def invert_linear_layer(self, linear_layer, y):
        if not isinstance(linear_layer, nn.Linear):
            raise TypeError("Expected an nn.Linear layer.")

        W = linear_layer.weight  # Shape: (output_dim, input_dim)
        b = linear_layer.bias  # Shape: (output_dim,)
        W_inv = torch.linalg.pinv(W)
        x = torch.matmul(W_inv, (y - b).T).T
        return x

    def invert(self, output):
        for layer in reversed(self.layers):
            output = layer.inverse(output)  # Apply inverse of each RealNVPNode
        downscale = self.invert_linear_layer(self.upscale_net, output)
        return downscale


def transform(X, Y, std):
    X = torch.tensor(X, dtype=torch.float32, requires_grad=True) if not torch.is_tensor(X) else X
    Y = torch.tensor(Y, dtype=torch.float32, requires_grad=True) if not torch.is_tensor(Y) else Y

    points = torch.stack([X, Y], dim=-1) if X.ndim == 0 else torch.stack([X, Y], dim=-1)

    pos_X = torch.tensor([0, 8], dtype=torch.float32)
    pos_Y = torch.tensor([0, 8], dtype=torch.float32)
    predefined_positions = torch.stack([
        torch.tensor([pos_X[0], pos_Y[0]]),
        torch.tensor([pos_X[1], pos_Y[0]]),
        torch.tensor([pos_X[0], pos_Y[1]]),
        torch.tensor([pos_X[1], pos_Y[1]])
    ])

    distances = torch.norm(points[:, None, :] - predefined_positions, dim=2) + 1e-6

    d0 = torch.tensor(1.0, dtype=torch.float32)
    power_transmit = torch.tensor(-14.32, dtype=torch.float32)
    frequency = torch.tensor(3993.6 * 1e6, dtype=torch.float32)
    wavelength = torch.tensor(3e8, dtype=torch.float32) / frequency
    K_dB = 20 * torch.log10(wavelength / (4 * torch.pi * d0))                # suy hao tại điểm reference point
    pathloss_exponent = torch.tensor(3, dtype=torch.float32)          # thường là 1.6 đến 1.8; free space l 2 ở đây đang là 2 trong khi train là 3
    shadowfading_deviation_dB = torch.normal(mean=torch.tensor(0, dtype=torch.float32), std=torch.tensor(std, dtype=torch.float32), size=distances.shape)

    pathloss_attenuation_dB = 10 * pathloss_exponent * torch.log10(distances / d0)

    power_received = power_transmit + K_dB - pathloss_attenuation_dB - shadowfading_deviation_dB
    return power_received


def fx(x, dt):
    F = torch.tensor([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1,  0],
                      [0, 0, 0,  1]], dtype=torch.float32)
    return torch.matmul(F, x)


def hx(x):
    x = torch.tensor([x[0], x[1]], requires_grad=True, dtype=torch.float32)
    x = torch.reshape(x, (1, 2))
    power_received = model(x).detach()
    return power_received

def H_jacobian(x):
    mea_beacon = 4
    # Extract inputs as tensors

    X = torch.tensor(x[0], requires_grad=True, dtype=torch.float32)
    Y = torch.tensor(x[1], requires_grad=True, dtype=torch.float32)

    input_tensor = torch.stack([X, Y])

    input_tensor = torch.tensor(input_tensor, requires_grad=True, dtype=torch.float32)
    input_tensor = torch.reshape(input_tensor, (1, 2))
    J = jacobian(model, input_tensor)

    # Convert to NumPy and append zero-velocity derivatives
    J = J.detach().numpy()  # Convert PyTorch tensor to NumPy array
    J = np.squeeze(J)
    derivative_velocity = np.zeros((mea_beacon, 2))  # Add zero columns
    J_combined = np.append(J, derivative_velocity, axis=1)

    return torch.tensor(J_combined, requires_grad=True, dtype=torch.float32)

model = torch.load('trained_model/v41NVP_4Aff_4hiddlayer_64hid_size8x8_100x100samples_200001_drop0_decay1e-5_8train2val_OptLR.pth', map_location=torch.device('cpu'))
# model = torch.load('trained_model/v6newNVP_4Aff_4hidd_size8x8_10samples_std15_10000.pth', map_location=torch.device('cpu'))
model.eval()
std = 1.5
samples = 200
X = torch.linspace(1, 7, samples)
# linear

# Y = X
# file1_path = "RSS_vanilla_linear.csv"
# file2_path = "RSS_NF_linear.csv"
# sin trajectory
# Y = 4 + 3 * torch.sin(2 * np.pi * X / 8)
# file1_path = "RSS_vanilla_sinusoidal.csv"
# file2_path = "RSS_NF_sinusoidal.csv"
# non-linear trajectory
# Tham số xoắn ốc
a = 0.001  # Bán kính ban đầu
b = 0.3  # Độ mở rộng của vòng xoắn
theta = torch.linspace(0, 4 * np.pi, 200)  # Góc từ 0 đến 4 vòng (8π)

# Tính tọa độ X, Y
r = a + b * theta
X = r * torch.cos(theta) + 4
Y = r * torch.sin(theta) + 4
# file1_path = "RSS_vanilla_spiral.csv"
# file2_path = "RSS_NF_spiral.csv"
location_array = torch.stack([X, Y], dim=1)
RSS = transform(X, Y, std)

mea_beacon = 4
initial_position = location_array[0]
x = torch.tensor([initial_position[0], initial_position[1], 0, 0], dtype=torch.float32).reshape(4, 1)
dt = 1 / samples
estimation_cov = 0.0001 * torch.eye(4, dtype=torch.float32)
motion_cov = 0.005 * torch.eye(4, dtype=torch.float32)
measurement_cov = torch.sqrt(torch.tensor(std)) * torch.sqrt(torch.tensor(std)) *torch.eye(mea_beacon, dtype=torch.float32)

F = torch.tensor([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=torch.float32)

pos = x[:2].T.clone()

idx = 1
rss_predictions = []
ys = []

start_time = time.time()
while idx < len(location_array):
    predictedState = fx(x, dt)
    estimation_cov = torch.matmul(torch.matmul(F, estimation_cov), F.T) + motion_cov
    rss_measurement = RSS[idx].reshape(mea_beacon, 1)
    rss_prediction = hx(predictedState.squeeze()).reshape(mea_beacon, 1)
    H = H_jacobian(x)
    latent_matrix = torch.matmul(torch.matmul(H, estimation_cov), H.T) + measurement_cov
    K = torch.matmul(torch.matmul(estimation_cov, H.T), torch.linalg.inv(latent_matrix))
    y = (rss_measurement - rss_prediction)
    x = predictedState + torch.matmul(K, y)
    estimation_cov = torch.matmul(torch.eye(4) - torch.matmul(K, H), estimation_cov)
    pos = torch.cat([pos, x[:2].T], dim=0)
    idx += 1

    # rss_predictions.append(rss_prediction.detach().numpy().flatten())
    # ys.append(y.detach().numpy().flatten())
end_time = time.time()
print(f"Runtime: {end_time - start_time:.5f} seconds")

rss_predictions = torch.tensor(rss_predictions).numpy()
rss_std = rss_predictions.std()
ys = torch.tensor(ys).numpy()
ys_std = ys.std()

pos_true = torch.flatten(location_array, start_dim=0, end_dim=-1)
pos_pred = torch.flatten(pos, start_dim=0, end_dim=-1)
loss = nn.MSELoss()
mse = loss(pos_pred, pos_true)
dB = 10*torch.log10(mse)
print(f"mse Flow = {mse:.6f}")
print(f"mse Flow dB = {dB:.6f}")

pos_vanilla = model.invert(RSS)
mse_vanilla = loss(pos_vanilla, location_array)
vanilla_dB = 10*torch.log10(mse_vanilla)
print(f"mse_vanilla = {mse_vanilla:.6f}")
print(f"mse_vanilla dB = {vanilla_dB:.6f}")

# file1_df = pd.DataFrame({"location_x": pos_vanilla[:, 0].detach().numpy(), "location_y": pos_vanilla[:, 1].detach().numpy()})
# file2_df = pd.DataFrame({"location_x": pos[:, 0].detach().numpy(), "location_y": pos[:, 1].detach().numpy()})
# file1_df.to_csv(file1_path, index=False)
# file2_df.to_csv(file2_path, index=False)
# Vẽ groundtruth


# Convert tensors to numpy arrays
X_np = X.numpy()
Y_np = Y.numpy()
pos_np = pos.detach().numpy()
pos_vanilla_np = pos_vanilla.detach().numpy()

# Define step for markers (e.g., show markers every 10th point)
marker_step = 10


fig, ax = plt.subplots()

# Plot with reduced marker frequency
line1, = ax.plot(X_np, Y_np, label='Groundtruth', color='blue', linestyle='--', linewidth=1.5)
marker1, = ax.plot(X_np[::marker_step], Y_np[::marker_step], 'bo', markersize=4)

line2, = ax.plot(pos_np[:, 0], pos_np[:, 1], label='Neural Kalman', color='red', linestyle='--', linewidth=1.5)
marker2, = ax.plot(pos_np[::marker_step, 0], pos_np[::marker_step, 1], 'rs', markersize=4)

# line3, = ax.plot(pos_vanilla_np[:, 0], pos_vanilla_np[:, 1], label='Vanilla NF', color='black', linestyle='--', linewidth=1.5)
# marker3, = ax.plot(pos_vanilla_np[::marker_step, 0], pos_vanilla_np[::marker_step, 1], 'k^', markersize=4)

# Enhancing the visualization
ax.legend([(line1, marker1), (line2, marker2)], ['Groundtruth', 'Flow-based EKF'], fontsize=10, loc='best', frameon=True, edgecolor='gray', handler_map={tuple: matplotlib.legend_handler.HandlerTuple(ndivide=None)})

ax.set_xlim(0, 8)
ax.set_ylim(0, 8)
ax.set_xlabel("X-axis", fontsize=12)
ax.set_ylabel("Y-axis", fontsize=12)
ax.set_title("Flow-based EKF", fontsize=14, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.6)

# Save the professional plot
# plt.savefig("RSS sinusoidal.png", bbox_inches='tight', dpi=300)
plt.show()
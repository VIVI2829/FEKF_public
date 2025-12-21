import torch
from torch import nn
from torch.autograd.functional import jacobian
import matplotlib.pyplot as plt
import numpy as np
from torch.onnx.symbolic_opset9 import detach
from torch.utils.tensorboard import SummaryWriter
import os
import pandas as pd

# MODEL
class RealNVPNode(nn.Module):
    def __init__(self, mask, hidden_size):
        super(RealNVPNode, self).__init__()
        self.dim = len(mask)
        self.mask = nn.Parameter(mask, requires_grad=False)

        self.s_func = nn.Sequential(
            nn.Linear(in_features=self.dim, out_features=hidden_size), nn.Tanh(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size), nn.Tanh(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size), nn.Tanh(),
            nn.Linear(in_features=hidden_size, out_features=self.dim)
        )

        self.scale = nn.Parameter(torch.zeros(self.dim))

        self.t_func = nn.Sequential(
            nn.Linear(in_features=self.dim, out_features=hidden_size), nn.Tanh(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size), nn.Tanh(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size), nn.Tanh(),
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
    def __init__(self, input, masks, hidden_size, input_dim=2, output_dim=4):
        super(RealNVP, self).__init__()

        # self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size

        # Linear transformation to upscale input (2x1 → 4x1)
        self.upscale_net = nn.Linear(input_dim, output_dim)

        self.masks = masks
        self.layers = nn.ModuleList([
            RealNVPNode(mask, self.hidden_size) for mask in self.masks
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

        W_inv = torch.linalg.pinv(W)  # Use pseudo-inverse for all cases
        x = torch.matmul(W_inv, (y - b).T).T  # Correct order of multiplication

        return x

    def invert(self, output):
        for layer in reversed(self.layers):
            output = layer.inverse(output)  # Apply inverse of each RealNVPNode
        downscale = self.invert_linear_layer(self.upscale_net, output)
        return downscale



def transform(X, Y, std):
    X = torch.tensor(X, dtype=torch.float64, requires_grad=True) if not torch.is_tensor(X) else X
    Y = torch.tensor(Y, dtype=torch.float64, requires_grad=True) if not torch.is_tensor(Y) else Y
    points = torch.stack([X, Y], dim=-1) if X.ndim == 0 else torch.stack([X, Y], dim=-1)
    points = torch.reshape(points, (-1, 2))
    pos_X = torch.tensor([0, 8], dtype=torch.float64)
    pos_Y = torch.tensor([0, 8], dtype=torch.float64)
    predefined_positions = torch.stack([torch.stack([pos_X[0], pos_Y[0]]),
                                        torch.stack([pos_X[1], pos_Y[0]]),
                                        torch.stack([pos_X[0], pos_Y[1]]),
                                        torch.stack([pos_X[1], pos_Y[1]]), ])
    distances = torch.norm(points[:, None, :] - predefined_positions[None, :, :], dim=2) + 1e-6
    d0 = torch.tensor(1.0, dtype=torch.float64)
    power_transmit = torch.tensor(-14.32, dtype=torch.float64)
    frequency = torch.tensor(3993.6 * 1e6, dtype=torch.float64)
    wavelength = torch.tensor(3e8, dtype=torch.float64) / frequency
    K_dB = 20 * torch.log10(wavelength / (4 * torch.pi * d0))
    pathloss_exponent = torch.tensor(3.0, dtype=torch.float64)
    shadowfading_deviation_dB = torch.normal(mean=torch.tensor(0, dtype=torch.float64),
                                             std=torch.tensor(std, dtype=torch.float64), size=distances.shape)
    pathloss_attenuation_dB = 10 * pathloss_exponent * torch.log10(distances / d0)
    power_received = power_transmit + K_dB - pathloss_attenuation_dB - shadowfading_deviation_dB
    return power_received


class ExtendedKalmanFilter:
    def __init__(self,model, dt, std = 1.5):
        self.model = model
        self.std = torch.tensor(std, dtype=torch.float64)
        self.mea_beacon = 4
        self.motion_cov = 0.005 * torch.eye(4, dtype=torch.float64)
        self.measurement_cov = torch.tensor((self.std)**2) *torch.eye(self.mea_beacon, dtype=torch.float64)
        #
        self.F = torch.tensor([[1, 0, dt, 0],
                               [0, 1, 0, dt],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]], dtype=torch.float64)

    def hx(self, x):
        x = torch.tensor([x[0], x[1]], requires_grad=True, dtype=torch.float64).reshape(1, 2)
        power_received = self.model(x)  # Không detach để giữ gradient
        return power_received

    def H_jacobian(self, x):
        """ Tính ma trận Jacobian của mô hình đo """
        X = torch.tensor(x[0], requires_grad=True, dtype=torch.float64)
        Y = torch.tensor(x[1], requires_grad=True, dtype=torch.float64)
        input_tensor = torch.stack([X, Y]).reshape(1, 2)

        J = jacobian(self.model, input_tensor).detach().numpy()  # Chuyển sang NumPy để xử lý
        J = np.squeeze(J)

        derivative_velocity = np.zeros((self.mea_beacon, 2))
        J_combined = np.append(J, derivative_velocity, axis=1)

        return torch.tensor(J_combined, dtype=torch.float64)

    def run_filter(self, initial_position, RSS):
        x = torch.tensor([initial_position[0], initial_position[1], 0, 0], dtype=torch.float64).view(4, 1)
        estimation_cov = torch.tensor(0.0001, dtype=torch.float64) * torch.eye(4, dtype=torch.float64)
        pos = initial_position.reshape(1, 2)
        idx = 1
        while idx < len(RSS):
            predictedState = self.F @ x
            estimation_cov = self.F @ estimation_cov @ self.F.T + self.motion_cov
            rss_measurement = RSS[idx].view(self.mea_beacon, 1)
            rss_prediction = self.hx(predictedState.squeeze()).reshape(self.mea_beacon, 1)
            H = self.H_jacobian(predictedState.flatten())
            latent_matrix = H @ estimation_cov @ H.T + self.measurement_cov
            K = estimation_cov @ H.T @ torch.inverse(latent_matrix)
            y = rss_measurement - rss_prediction
            x = predictedState + K @ y
            estimation_cov = (torch.eye(4, dtype=torch.float64) - K @ H) @ estimation_cov
            pos = torch.cat((pos, x[:2].T), dim=0)
            idx += 1
        return pos

model = torch.load('trained_model/v41NVP_4Aff_4hiddlayer_64hid_size8x8_100x100samples_200001_drop0_decay1e-5_8train2val_OptLR.pth', map_location=torch.device('cpu')).double()

model.double()
model.eval()
std = 1.5
traj_values = 3
samples = 400  # Số lượng mẫu khởi tạo
trials = 5  # Mặc định số lần thử nghiệm Monte Carlo

NF_mse_results = []
Vanilla_mse_result = []
loss = nn.MSELoss()
for traj in range(traj_values):
    if traj == 0:
        X = torch.linspace(1, 7, samples)
        Y = X
        location_array = torch.stack([X, Y], dim=1)
    elif traj == 1:
        X = torch.linspace(1, 7, samples)
        Y = 4 + 3 * torch.sin(2 * np.pi * X / 8)
        location_array = torch.stack([X, Y], dim=1)
    elif traj == 2:
        a = 0  # Bán kính ban đầu
        b = 0.3  # Độ mở rộng của vòng xoắn
        theta = torch.linspace(0, 4 * np.pi, samples)  # Góc từ 0 đến 4 vòng (8π)
        # Tính tọa độ X, Y
        r = a + b * theta
        X = r * torch.cos(theta) + 4
        Y = r * torch.sin(theta) + 4
        location_array = torch.stack([X, Y], dim=1)
    else:
        print("Expect traj from 0 to 2")
        break
    print("Trajectory:", traj)
    NF_mse_trials = []
    Vanilla_mse_trials = []
    for _ in range(trials):
        RSS = transform(X, Y, std)
        # Vanilla pos
        pos_vanilla = model.invert(RSS)

        # EKF pos
        ekf = ExtendedKalmanFilter(model, 1/samples, std)
        pos = ekf.run_filter(location_array[0], RSS)
        NF_mse = loss(pos, location_array)
        if NF_mse > 10:
            continue
        NF_mse_trials.append(NF_mse)
        Vanilla_mse = loss(pos_vanilla, location_array)
        Vanilla_mse_trials.append(Vanilla_mse)
    # del RSS, pos_vanilla, Vanilla_mse, pos, NF_mse
    del RSS, pos, NF_mse
    NF_mse_mean = torch.tensor(NF_mse_trials).mean()
    print("MSE[dB]", 10*torch.log10(NF_mse_mean))
    NF_mse_results.append(NF_mse_mean)
    Vanilla_mse_mean = torch.tensor(Vanilla_mse_trials).mean()
    Vanilla_mse_result.append(Vanilla_mse_mean)

# Lưu file csv
NF_mse_results = pd.DataFrame(10*np.log10(NF_mse_results))
Vanilla_mse_result = pd.DataFrame(10*np.log10(Vanilla_mse_result))
NF_mse_results.to_csv("v41Monte_Carlo_EKF_NF_msedB.csv", header=False)
Vanilla_mse_result.to_csv("v41Monte_Carlo_EKF_Vanilla_msedB.csv", header=False)

# NF_mean_mse = torch.tensor(NF_mse_results).mean(dim=1)
# NF_std_mse = torch.tensor(NF_mse_results).std(dim=1)
# Vanilla_mean_mse = torch.tensor(Vanilla_mse_result).mean(dim=1)
# Vanilla_std_mse = torch.tensor(Vanilla_mse_result).std(dim=1)
print("NF mean_mse", NF_mse_results)
# print("NF std_mse", NF_std_mse)
print("Vanilla mean_mse", Vanilla_mse_result)
# print("Vanilla std_mse", Vanilla_std_mse)

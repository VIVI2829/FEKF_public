# ***********************
# Chạy EKF với NF_model
# ***********************
import torch
from torch import nn
from torch.autograd.functional import jacobian
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functorch import jacrev


class RealNVPNode(nn.Module):
    def __init__(self, mask, hidden_size):
        super(RealNVPNode, self).__init__()
        self.dim = len(mask)
        self.mask = nn.Parameter(mask, requires_grad=False)

        self.s_func = nn.Sequential(
            nn.Linear(in_features=self.dim, out_features=hidden_size), nn.Tanh(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size), nn.Tanh(),
            nn.Linear(in_features=hidden_size, out_features=self.dim)
        )

        self.scale = nn.Parameter(torch.zeros(self.dim))

        self.t_func = nn.Sequential(
            nn.Linear(in_features=self.dim, out_features=hidden_size), nn.Tanh(),
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

        # inv_log_det_jac = ((1 - self.mask) * -s).sum(-1)
        return x


class RealNVP(nn.Module):
    def __init__(self, masks, hidden_size, input_dim=2, output_dim=2):
        super(RealNVP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size

        # Linear transformation to upscale input (2x1 → 4x1)
        # self.upscale_net = nn.Linear(input_dim, output_dim)

        self.masks = masks
        self.layers = nn.ModuleList([
            RealNVPNode(mask, self.hidden_size) for mask in self.masks
        ])

    def forward(self, x):
        # upscale = self.upscale_net(input)
        for layer in self.layers:
            x = layer.forward(x)  # Apply each RealNVPNode sequentially
        return x

    def invert_linear_layer(self, linear_layer, y):
        if not isinstance(linear_layer, nn.Linear):
            raise TypeError("Expected an nn.Linear layer.")

        W = linear_layer.weight  # Shape: (output_dim, input_dim)
        b = linear_layer.bias  # Shape: (output_dim,)

        W_inv = torch.linalg.pinv(W)  # Use pseudo-inverse for all cases
        x = torch.matmul(W_inv, (y - b).T).T  # Correct order of multiplication

        return x

    def invert(self, y):
        for layer in reversed(self.layers):
                y = layer.inverse(y)  # Apply inverse of each RealNVPNode
        # downscale = self.invert_linear_layer(self.upscale_net, output)
        return y


class ExtendedKalmanFilter:
    def __init__(self, model):
        self.state_dim = 4
        self.observation_dim = 2
        self.motion_cov = torch.eye(self.state_dim, dtype=torch.float64)
        self.measurement_cov = torch.eye(self.observation_dim, dtype=torch.float64)
        self.estimation_cov = torch.eye(self.state_dim, dtype=torch.float64)
        self.F = torch.tensor([[1, 0, 1, 0],
                               [0, 1, 0, 1],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]], dtype=torch.float64)
        self.model = model

    def fx(self, x, dt):
        self.F = torch.tensor([[1, 0, dt, 0],
                               [0, 1, 0, dt],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]], dtype=torch.float64)
        return torch.matmul(self.F, x)

    def hx(self, x):
        x = torch.tensor([x[0], x[1]], requires_grad=True, dtype=torch.float64)
        x = torch.reshape(x, (1, self.state_dim - 2))
        return self.model(x).detach()

    def invert(self, y):
        y = torch.tensor(y, requires_grad=True, dtype=torch.float64)
        return self.model.invert(y)

# Không sử dụng khi train end2end
    # def H_jacobian(self, x):
    #     X = torch.tensor(x[0], requires_grad=True, dtype=torch.float64)
    #     Y = torch.tensor(x[1], requires_grad=True, dtype=torch.float64)
    #     input_tensor = torch.stack([X, Y]).reshape(1, self.observation_dim)
    #     J = jacobian(self.model, input_tensor).detach().numpy()
    #     J = np.squeeze(J)
    #     derivative_velocity = np.zeros((self.observation_dim, 2))
    #     J_combined = np.append(J, derivative_velocity, axis=1)
    #     return torch.tensor(J_combined, requires_grad=True, dtype=torch.float64)

    def H_jacobian(self, x):
        X = x[0:2].clone().detach().requires_grad_(True)
        input_tensor = X.reshape(1, self.observation_dim)

        jac_fn = jacrev(self.model)  # tạo hàm đạo hàm
        J = jac_fn(input_tensor)  # tính đạo hàm tại input
        J = J.reshape(self.observation_dim, self.observation_dim)

        derivative_velocity = torch.zeros((self.observation_dim, 2), dtype=torch.float64, device=x.device)
        J_combined = torch.cat([J, derivative_velocity], dim=1)

        return J_combined

    def run_filter(self, initial_state, MEA, dt):
        idx = 1
        # pos lưu chuỗi tọa độ
        pos = torch.tensor([initial_state[0], initial_state[1]]).reshape(1, 2)
        pos_vanillas = torch.tensor([initial_state[0], initial_state[1]]).reshape(1, 2)
        x = torch.tensor(initial_state, dtype=torch.float64).view(self.state_dim, 1)
        while idx < len(MEA):
            predictedState = self.fx(x, dt[idx])
            self.estimation_cov = torch.matmul(torch.matmul(self.F, self.estimation_cov), self.F.T) + self.motion_cov
            pos_measurement = MEA[idx].view(self.observation_dim, 1)
            pos_prediction = self.hx(predictedState.squeeze()).reshape(self.observation_dim, 1)
            # pos_vanilla = self.invert(pos_measurement.reshape(1, 2))
            H = self.H_jacobian(predictedState)
            # print(H)
            latent_matrix = H @ self.estimation_cov @ H.T + self.measurement_cov + 1e-6 * torch.eye(self.observation_dim, dtype=torch.float64) # ổn định nghịch đảo
            K = self.estimation_cov @ H.T @ torch.linalg.inv(latent_matrix) # instead of torch.inverse(latent_matrix)
            y = pos_measurement - pos_prediction
            x = predictedState + K @ y
            I = torch.eye(self.state_dim, dtype=torch.float64)
            self.estimation_cov = (I - K @ H) @ self.estimation_cov
            # self.estimation_cov = (I - K @ H) @ self.estimation_cov @ (I - K @ H).T + K @ self.measurement_cov @ K.T  # Josep form đảm bảo dương xác định cho P
            pos = torch.cat((pos, x[:2].T), dim=0)
            # pos_vanillas = torch.cat((pos_vanillas, pos_vanilla), dim = 0)
            idx += 1
        # return pos, pos_vanillas
        return pos

layers = 4  # Number of masks (adjustable)

# Define the base masks
mask1 = torch.tensor([1, 0], dtype=torch.float64)
mask2 = torch.tensor([0, 1], dtype=torch.float64)

masks = [mask1 if i % 2 == 0 else mask2 for i in range(layers)]

masks = torch.stack(masks)

hidden_size = 64
model = RealNVP(masks, hidden_size, 2, 2)
# LOAD 2201
# model = torch.load('trained_model/v1_2201NVP_4Aff_64hidd_cal_200kiter.pth',
#                    map_location=torch.device('cpu'))
# LOAD 2904
model = torch.load('trained_model/v1_2904NVP_4Aff_64hidd_cal2_200kiter.pth',
                    map_location=torch.device('cpu'))
# LOAD MIX
# model = torch.load('trained_model/v1_mix_4Aff_64hidd_cal_200kiterdecay1e5.pth',
#                    map_location=torch.device('cpu'))

model = model.double()
model.eval()

test_location_array = torch.load("data/2904/cal_test_groundtruth.pt")
test_location_array = torch.tensor(test_location_array, dtype=torch.float64)
test_measurement_array = torch.load("data/2904/cal_test_odo.pt")
test_measurement_array = torch.tensor(test_measurement_array, dtype=torch.float64)
dt = torch.load("data/2904/cal_dt.pt").double()

# test_location_array = torch.load("cal_test_groundtruth.csv", map_location="cpu")
groundtruth_x = test_location_array[:,0]
groundtruth_y = test_location_array[:,1]
initial_state = torch.stack([groundtruth_x[0], groundtruth_y[0], torch.tensor(0), torch.tensor(0)], dim=0)

# Run Kalman Filter
ekf = ExtendedKalmanFilter(model)
pos = ekf.run_filter(initial_state, test_measurement_array, dt)
# vanilla_pos = model.invert(test_measurement_array)
Flow_EKF = pd.DataFrame(pos.detach().numpy())
# Vanilla = pd.DataFrame(vanilla_pos.detach().numpy())

# Lưu ra file CSV, không header, không index
# Flow_EKF.to_csv("data/2201/fullcrossFEKF.csv", index=False, header=False)
# Vanilla.to_csv("data/2201/fullcrossVanilla.csv", index=False, header=False)
# Compute MSE
loss = nn.MSELoss()
mse = loss(pos, test_location_array[:,:])
mse_mea = loss(test_measurement_array, test_location_array[:,:])
# mse_vanilla = loss(vanilla_pos, test_location_array[:,:])

print("mseFEKF(dB)", 10 * torch.log10(mse))
print(f"mseFEKF = {mse:.6f}")
# print("mse_vanilla(dB)", 10 * torch.log10(mse_vanilla))
# print(f"mse_vanilla = {mse_vanilla:.6f}")

pos_np = pos.detach().cpu().numpy()
# vanilla_pos = vanilla_pos.detach().cpu().numpy()
gt = test_location_array[:,:]

# Plot
plt.figure(figsize=(5, 5))
plt.plot(gt[:, 0], gt[:, 1], 'black', label='Groundtruth')
# plt.plot(vanilla_pos[:, 0], vanilla_pos[:, 1], 'g-', label='Vanilla')  # 'ro' = red dot
plt.plot(pos_np[:, 0], pos_np[:, 1], 'r-', label='Flow-based')  # 'ro' = red dot
plt.plot(test_measurement_array[:,0].detach().numpy(), test_measurement_array[:,1].detach().numpy(), 'y', label='Odometry')
# plt.title("Data NCLT 2904")
plt.xlabel("X-axis[m]")
plt.ylabel("Y-axis[m]")
plt.grid(True)
plt.axis('equal')  # Keep aspect ratio square
plt.legend()
plt.show()

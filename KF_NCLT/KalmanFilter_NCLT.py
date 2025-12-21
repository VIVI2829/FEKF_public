import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.autograd.functional import jacobian
import pandas as pd

class KalmanFilter:
    def __init__(self, f):
        self.state_variable = 4
        self.observation_variable = 2
        self.dt = torch.tensor(1/f,  dtype=torch.float64)
        # self.motion_cov = torch.eye(self.state_variable, dtype=torch.float64)
        # 0.4571285761615131 [0.1; 0.5; 1]
        self.motion_cov = torch.eye(self.state_variable, dtype=torch.float64)
        self.measurement_cov = torch.eye(self.observation_variable, dtype=torch.float64)
        self.measurement_cov = torch.eye(self.observation_variable, dtype=torch.float64)
        self.estimation_cov = torch.eye(self.state_variable, dtype=torch.float64)
        # x = [x, y, delta_x, delta_y]
        self.F = torch.tensor([[1, 0, self.dt, 0],
                               [0, 1, 0, self.dt],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]], dtype=torch.float64)
        self.H = torch.tensor([[1, 0, 0, 0],
                               [0, 1, 0, 0]], dtype=torch.float64)
    def fx(self, x):
        return torch.matmul(self.F, x)

    def run_filter(self, initial_state, MEA):
        x = torch.tensor(initial_state, dtype=torch.float64).view(self.state_variable, 1)
        pos = torch.tensor([initial_state[0], initial_state[1]]).reshape(1, 2)
        idx = 1
        while idx < len(MEA):
            predictedState = self.fx(x)
            self.estimation_cov = torch.matmul(torch.matmul(self.F, self.estimation_cov), self.F.T) + self.motion_cov
            pos_measurement = MEA[idx].view(self.observation_variable, 1)
            # mea_prediction = transform(predictedState[2], predictedState[3])
            mea_prediction = self.H @ x
            # H = self.H_jacobian(predictedState.flatten())                           # 2x4
            latent_matrix = self.H @ self.estimation_cov @ self.H.T + self.measurement_cov    # 2x2
            K = self.estimation_cov @ self.H.T @ torch.inverse(latent_matrix)            # 4x2
            y = pos_measurement - mea_prediction                                 # 2x1
            x = predictedState + K @ y                                              # 4x1
            # print(x)
            self.estimation_cov = (torch.eye(self.state_variable) - K @ self.H) @ self.estimation_cov
            pos = torch.cat((pos, x[:2].T), dim=0)
            idx += 1
        return pos


f = 1 # (Hz)

test_location_array = torch.load("INPUT/2201/cal_test_groundtruth.pt")
test_location_array = torch.tensor(test_location_array)
measurement_array = torch.load("INPUT/2201/cal_test_odo.pt")
measurement_array = torch.tensor(measurement_array)
gt = test_location_array

# Chuyển đổi thành torch tensor
X_true = test_location_array[:, 0]
Y_true = test_location_array[:, 1]
location_array = torch.stack([X_true, Y_true], dim=1)

initial_state = torch.tensor([X_true[0], Y_true[0], 0, 0], dtype = torch.float64)
ekf = KalmanFilter(f)
pos = ekf.run_filter(initial_state, measurement_array)
loss = nn.MSELoss()
mse = loss(pos, location_array)
dB = 10*torch.log10(mse)
file2_df = pd.DataFrame({"location_x": pos[:, 0].detach().numpy(), "location_y": pos[:, 1].detach().numpy()})
file2_df.to_csv("KF2201.csv", index=False)
print(f"mse = {mse:.6f}")
print("mse dB", dB)
plt.plot(gt[:, 0], gt[:, 1], 'black', label='Groundtruth')
plt.plot(pos[:, 0].detach().numpy(), pos[:, 1].detach().numpy(), color='r', label='Prediction')
plt.legend()
plt.show()


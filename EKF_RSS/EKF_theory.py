import numpy as np
import torch
from torch import nn
import pandas as pd
from torch.autograd import grad
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from torch.autograd.functional import jacobian


def transform(X, Y):
    """
    Calculate the received power for given X and Y positions (scalar or array).
    """
    # Ensure X and Y are tensors
    X = torch.tensor(X, dtype=torch.float32, requires_grad=True) if not torch.is_tensor(X) else X
    Y = torch.tensor(Y, dtype=torch.float32, requires_grad=True) if not torch.is_tensor(Y) else Y
    # Combine X and Y into a single tensor of points
    points = torch.stack([X, Y], dim=-1) if X.ndim == 0 else torch.stack([X, Y], dim=-1)
    points = torch.reshape(points, (-1,2))
    # Define the predefined positions
    pos_X = torch.tensor([0, 8], dtype=torch.float32)
    pos_Y = torch.tensor([0, 8], dtype=torch.float32)
    predefined_positions = torch.stack([
        torch.stack([pos_X[0], pos_Y[0]]),  # (0, 0)
        torch.stack([pos_X[1], pos_Y[0]]),  # (1, 0)
        torch.stack([pos_X[0], pos_Y[1]]),  # (0, 1)
        torch.stack([pos_X[1], pos_Y[1]])  # (1, 1)
    ])

    # Tính distances
    distances = torch.norm(points[:, None, :] - predefined_positions[None, :, :], dim=2) + 1e-6

    # Constants (ensure they are tensors)
    d0 = torch.tensor(1.0, dtype=torch.float32)  # Reference distance in meters
    power_transmit = torch.tensor(-14.32, dtype=torch.float32)  # Transmit power in dBm
    frequency = torch.tensor(3993.6 * 1e6, dtype=torch.float32)  # Frequency in Hz
    wavelength = torch.tensor(3e8, dtype=torch.float32) / frequency  # Wavelength (as a tensor)
    K_dB = 20 * torch.log10(wavelength / (4 * torch.pi * d0))  # Free-space path loss gain
    pathloss_exponent = torch.tensor(3.0, dtype=torch.float32)  # Path loss exponent
    shadowfading_deviation_dB = torch.normal(mean=torch.tensor(0, dtype=torch.float32), std=torch.tensor(1.5, dtype=torch.float32), size=distances.shape)  # Shadow fading
    # Path loss attenuation in dB
    pathloss_attenuation_dB = 10 * pathloss_exponent * torch.log10(distances / d0)
    # Received power in dBm
    power_received = power_transmit + K_dB - pathloss_attenuation_dB - shadowfading_deviation_dB
    return power_received

def fx(x, dt):
    # State transition function for a constant velocity model
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1,  0],
                  [0, 0, 0,  1]])
    # print(np.dot(F, x))
    return np.dot(F, x)

def H_jacobian(x):
    mea_beacon = 4
    # Extract inputs as tensors
    X = torch.tensor(x[0], requires_grad=True, dtype=torch.float32)
    Y = torch.tensor(x[1], requires_grad=True, dtype=torch.float32)
    # Define a wrapper for `transform` to accept a single tensor input
    def wrapper(inputs):
        return transform(inputs[0], inputs[1])
    # Stack inputs for compatibility with `jacobian`
    input_tensor = torch.stack([X, Y])
    # Compute Jacobian using `torch.autograd.functional.jacobian`
    J = jacobian(wrapper, input_tensor)
    # Convert to NumPy and append zero-velocity derivatives
    J = J.detach().numpy()  # Convert PyTorch tensor to NumPy array
    J = np.squeeze(J)
    # mea_beacon = J.shape[0]  # Assuming number of rows corresponds to beacons
    # print("J", J)
    derivative_velocity = np.zeros((mea_beacon, 2))  # Add zero columns
    J_combined = np.append(J, derivative_velocity, axis=1)
    return torch.tensor(J_combined, dtype=torch.float32, requires_grad=True)
std = 1.5
samples = 400
X = torch.linspace(1, 7, samples)
# linear

# Y = X
# file1_path = "RSS_EKF_linear.csv"

# sin trajectory
# Y = 4 + 3 * torch.sin(2 * np.pi * X / 8)
# file1_path = "RSS_EKF_sinusoidal.csv"

# non-linear trajectory
# Tham số xoắn ốc
a = 0.001  # Bán kính ban đầu
b = 0.3  # Độ mở rộng của vòng xoắn
theta = torch.linspace(0, 4 * np.pi, 200)  # Góc từ 0 đến 4 vòng (8π)

# Tính tọa độ X, Y
r = a + b * theta
X = r * torch.cos(theta) + 4
Y = r * torch.sin(theta) + 4
file1_path = "RSS_EKF_spiral.csv"


location_array = torch.stack([X, Y], dim=1)
RSS = transform(X, Y)

mea_beacon = 4
initial_position = location_array[0]
x = torch.tensor([initial_position[0], initial_position[1], 0, 0], dtype=torch.float32).view(4, 1)
dt = 1 / samples
std = 1.5
estimation_cov = 0.0001 * torch.eye(4)
motion_cov = 0.005 * torch.eye(4)
measurement_cov = std**2 * torch.eye(mea_beacon)
# measurement_cov = torch.eye(mea_beacon)

F = torch.tensor([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=torch.float32)

pos = location_array[0].reshape(1, 2)
idx = 1
while idx < len(location_array):
    predictedState = F @ x
    estimation_cov = F @ estimation_cov @ F.T + motion_cov

    rss_measurement = RSS[idx].view(mea_beacon, 1)
    rss_prediction = transform(predictedState[0], predictedState[1]).view(mea_beacon, 1)
    H = H_jacobian(predictedState.flatten())

    latent_matrix = H @ estimation_cov @ H.T + measurement_cov
    K = estimation_cov @ H.T @ torch.inverse(latent_matrix)
    y = rss_measurement - rss_prediction

    x = predictedState + K @ y
    estimation_cov = (torch.eye(4) - K @ H) @ estimation_cov

    pos = torch.cat((pos, x[:2].T), dim=0)
    idx += 1

mse = mean_squared_error(location_array.numpy().flatten(), pos.detach().numpy().flatten())
print("MSE = ", mse)
print("MSE[dB", 10*np.log10(mse))
plt.plot(X.detach().numpy(), Y.detach().numpy(), label='True trajectory')
plt.plot(pos[:, 0].detach().numpy(), pos[:, 1].detach().numpy(), color = 'r', label='Prediction')
plt.grid()
plt.title("EKF_RSS")
plt.legend()
plt.show()

file1_df = pd.DataFrame({"location_x": pos[:, 0].detach().numpy(), "location_y": pos[:, 1].detach().numpy()})

# file1_df.to_csv(file1_path, index=False)
# loại bỏ

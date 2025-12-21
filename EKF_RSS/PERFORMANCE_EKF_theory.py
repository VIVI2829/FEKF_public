#****************************
# Chạy kiểm tra performance EKF theo CTLT
# Hàm run_ekf trả về mean của samples
# Vòng for trả về Monte Carlo theo trials
# Log1: mse/std with trial lines
# Log2: mse/std with sample lines
#****************************

import numpy as np
import torch
from torch import nn
import pandas as pd
from torch.autograd.functional import jacobian
from sklearn.metrics import mean_squared_error
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import os
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# np.random.seed(28)
def transform(X, Y):
    """
    Calculate the received power for given X and Y positions (scalar or array).
    """
    # Ensure X and Y are tensors
    X = torch.tensor(X, dtype=torch.float32, requires_grad=True) if not torch.is_tensor(X) else X
    Y = torch.tensor(Y, dtype=torch.float32, requires_grad=True) if not torch.is_tensor(Y) else Y
    # Combine X and Y into a single tensor of points
    points = torch.stack([X, Y], dim=-1) if X.ndim == 0 else torch.stack([X, Y], dim=-1)
    # Define the predefined positions
    pos_X = torch.tensor([0, 8], dtype=torch.float32)
    pos_Y = torch.tensor([0, 8], dtype=torch.float32)
    predefined_positions = torch.stack([
        torch.stack([pos_X[0], pos_Y[0]]),  # (0, 0)
        torch.stack([pos_X[1], pos_Y[0]]),  # (1, 0)
        torch.stack([pos_X[0], pos_Y[1]]),  # (0, 1)
        torch.stack([pos_X[1], pos_Y[1]])  # (1, 1)
    ])
    # Calculate Euclidean distances
    distances = torch.norm(points[:, None, :] - predefined_positions, dim=2) + 1e-6  # Avoid in-place modification
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
    return J_combined

# Define constants
std_values = [1.5]

def run_ekf(samples, X, Y):


    location_array = np.column_stack((X, Y))
    RSS = transform(X, Y)
    pos = location_array[0].reshape(1, 2)
    dt = 1 / samples
    mea_beacon = 4

    motion_cov = 0.005 * np.identity(4)
    measurement_cov = 1.5**2 * np.identity(4)
    estimation_cov = 0.0001 * np.identity(4)

    # State initialization
    initial_position = location_array[0]
    x = np.array([location_array[0, 0], location_array[0, 1], 0, 0]).reshape(4, 1)

    # Simulate EKF
    for idx in range(1, len(location_array)):
        # Prediction step
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        predictedState = fx(x, dt)
        estimation_cov =  np.dot(np.dot(F, estimation_cov), F.T) + motion_cov

        # Measurement step
        rss_measurement = RSS[idx].reshape(mea_beacon, 1)  # Simulated RSS measurement
        rss_prediction = transform(predictedState[0], predictedState[1]).reshape(mea_beacon,1)  # Simulated predicted RSS
        H = H_jacobian(x)  # Simple measurement model

        # Kalman gain
        latent_matrix = np.dot(np.dot(H, estimation_cov), H.T) + measurement_cov  # account for mea_cov
        K = np.dot(np.dot(estimation_cov, H.T), np.linalg.inv(latent_matrix))

        # Update state
        y = (rss_measurement - rss_prediction).detach().numpy()  # Measurement residual (mea_beaconx1) # checked
        x = predictedState + np.dot(K, y)  # Updated state (4x1)
        I = np.identity(4)
        latent_matrix1 = I - np.dot(K, H)
        latent_matrix2 = np.dot(np.dot(measurement_cov, K), K.T)  # acount for mea covariance
        estimation_cov = np.dot(np.dot(latent_matrix1, estimation_cov),
                                latent_matrix1.T) + latent_matrix2
        # trackedLocation = np.array([x[0], x[1]]).reshape(1, 2)
        # Store position
        pos = np.vstack([pos, x[:2].flatten()])
        # print(X[idx], Y[idx], pos[idx])

    # Compute metrics
    mse = mean_squared_error(location_array.flatten(), pos.flatten())
    # print("mse[dB]", 10 * np.log10(mse))
    # plt.plot(X, Y, color = 'b', label='True trajectory')
    # plt.plot(pos[:, 0], pos[:, 1], color = 'r', label='Prediction')
    # plt.legend()
    # plt.show()
    diff = np.sqrt(np.sum((location_array - pos)**2, axis=1))
    avg_dist = np.mean(diff)
    # sau mỗi 1 chương trình ekf sẽ trả về mse và độ lệch trung bình
    return avg_dist, mse

# Monte Carlo simulations
samples = 400
num_trials = [5]
results = []
std = 1.5

traj_values = 3
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

    mse_mean = []
    for trials in num_trials:
        # Initialize Summary Writer
        mse_trials = []
        for _ in range(trials):
            avg_dist, mse = run_ekf(samples, X, Y) # Lưu từng trials
            mse_trials.append(mse)

        mse_mean = np.mean(mse_trials)
        print("mse[dB]", 10*np.log10(mse_mean))

        results.append({
        "std": std,
        "trials": trials,
        "samples": samples,
        "mse[[dB]": 10*np.log10(mse_mean)
    })
print(results)


# Save results to CSV
df_results = pd.DataFrame(results)
df_results.to_csv("MonteCarloEKF_RSS.csv", index=False)

#v19 200 samples
# Trajectory: 0
#               -5.077331276338338}
# Trajectory: 1
#               -5.008
# Trajectory: 2
#               -4.90

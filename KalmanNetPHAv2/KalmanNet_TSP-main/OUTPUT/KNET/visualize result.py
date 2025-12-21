import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np

# Đọc dữ liệu từ file CSV (KNet và KF)
KNet_x = pd.read_csv("2201/2daysKNEToutx.csv", header=None).iloc[:, 0]
KNet_y = pd.read_csv("2201/2daysKNETouty.csv", header=None).iloc[:, 0]
KNET_x = np.array(KNet_x[1:])
KNET_y = np.array(KNet_y[1:])
# KF_x = pd.read_csv("KNet_output/NCLT_result/KF_outx.csv", header=None).iloc[:, 0]
# KF_y = pd.read_csv("KNet_output/NCLT_result/KF_outy.csv", header=None).iloc[:, 0]

test_measurement_array = torch.load("2201/cal_test_odo.pt", map_location="cpu")
test_location_array = torch.load("2201/cal_test_groundtruth.pt", map_location="cpu")

groundtruth_x = test_location_array[:, 0].squeeze()
groundtruth_y = test_location_array[:, 1].squeeze()
odo_x = test_measurement_array[:, 0].squeeze()
odo_y = test_measurement_array[:, 1].squeeze()

# Tính MSE
knet_traj = np.stack((KNET_x, KNET_y), axis=1)[1:]
gt_traj = np.stack((groundtruth_x, groundtruth_y), axis=1)[1:]
KNET_mse = mean_squared_error(gt_traj, knet_traj)

print("KNET mse[dB]", 10*np.log10(KNET_mse))
# Vẽ biểu đồ
plt.figure(figsize=(8, 6))
plt.plot(groundtruth_x,groundtruth_y, 'k-', label="Groundtruth")  # Màu đen
plt.plot(KNet_x[1:], KNet_y[1:], 'r--', label="KNet Output")  # Màu đỏ nét đứt
plt.plot(odo_x, odo_y, 'b:', label="Odometry")  # Màu xanh nét chấm chấm

plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Trajectory Comparison")
plt.legend()
plt.grid(True)
plt.show()


# KNET mse[dB] 20.566177302209038 # v1 best model
# KNET mse[dB] 13.191965151409903 # 2201
# KNET mse[dB] 17.09672592171799
# KNET mse[dB] 22.328839818440645 # predict 2904 using 2201

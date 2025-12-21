import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
file_path = "ekf_CTLT_performance.csv"
data = pd.read_csv(file_path)

# Filter data by sample sizes and trials
# samples_colors = {100: 'orange', 500: 'green', 1000: 'blue', 50: 'red'}
trial_colors = {2000: 'green', 1500: 'red', 1000: 'orange', 500: 'yellow', 100: 'gray', 50: 'blue'}
# trial_markers = {500: 's', 2000: 'o', 100: 'v'}
samples_markers = {500: 's'}
# Create the plot
plt.figure(figsize=(10, 6))
for sample_size, marker in samples_markers.items():
    for trial, color in trial_colors.items():
        subset = data[(data['samples'] == sample_size) & (data['trials'] == trial)]
        plt.plot(subset['std'], subset['mse'], marker=marker, label=f'Samples: {sample_size}, Trials: {trial}', color=color)

# Add labels, legend, and title
plt.xlabel('std')
plt.ylabel('MSE')
plt.title('MSE vs Std for Different Sample Sizes and Trials')
plt.legend()
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Đọc dữ liệu từ file CSV
# file_path = "ekf_CTLT_performance.csv"
# data = pd.read_csv(file_path)
#
# # Filter data by sample sizes and trials
# samples_colors = {100: 'orange', 500: 'green', 1000: 'blue', 50: 'red'}
# # trial_colors = {2000: 'green', 1000: 'yellow', 500: 'blue', 100: 'red'}
# trial_markers = {500: 's', 2000: 'o', 100: 'v'}
# # samples_markers = {500: 's'}
# # Create the plot
# plt.figure(figsize=(10, 6))
# for sample_size, color in samples_colors.items():
#     for trial, marker in trial_markers.items():
#         subset = data[(data['samples'] == sample_size) & (data['trials'] == trial)]
#         plt.plot(subset['std'], subset['mse'], marker=marker, label=f'Samples: {sample_size}, Trials: {trial}', color=color)
#
# # Add labels, legend, and title
# plt.xlabel('std')
# plt.ylabel('MSE')
# plt.title('MSE vs Std for Different Sample Sizes and Trials')
# plt.legend()
# plt.grid(True)
#
# # Show the plot
# plt.tight_layout()
# plt.show()

import pandas as pd
import torch

# Load the data files
full_data_path = "fulldata2904xpredictedbyKNETx2201.csv"
mix_data_path = "mixKNETx2904.csv"

full_data = pd.read_csv(full_data_path)
mix_data = pd.read_csv(mix_data_path)

# Calculate sizes
total_samples = len(full_data)
num_train = round(total_samples * 0.85)
num_cv = round(total_samples * 0.10)
num_test = total_samples - num_cv - num_train

# Indices for splitting
cv_start_idx = 0
cv_end_idx = cv_start_idx + num_cv
train_start_idx = num_cv
train_end_idx = train_start_idx + num_train
test_start_idx = num_train + num_cv
test_end_idx = test_start_idx + num_test

# Split the data
cv_data = full_data.iloc[cv_start_idx:cv_end_idx]
test_data = full_data.iloc[test_start_idx:test_end_idx]
train_data = full_data.iloc[train_start_idx:train_end_idx]

# Check if test size matches mixKNETx2201
test_equal = len(test_data) == len(mix_data)

(test_equal, len(test_data), len(mix_data))

if test_equal:
    test_data.to_csv("crossdayKNETx2904.csv", index=False)
    # torch.save(test_data, "crossdayKNETy2201.csv")
    print("File saved.")
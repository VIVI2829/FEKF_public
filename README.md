# FlowEKF: Flow-based Extended Kalman Filter

Implementation of **"FlowEKF: Flow-based Extended Kalman Filter"** published at APSIPA ASC 2025.

## 📋 Overview

**FlowEKF** is a hybrid Kalman filtering framework that addresses the challenge of unknown and nonlinear observation models in state estimation. By leveraging Real-valued Non-Volume Preserving (Real-NVP) flow-based neural networks, FlowEKF learns complex observation mappings while maintaining lower computational complexity compared to existing methods.

## 🏗️ Repository Structure (key files)

```
FEKF_public/
├── EKF_RSS/				# Extended Kalman Filter for RSS-based localization
│   ├── EKF_theory.py			# The original extended Kalman filer
│   ├── PERFORMANCE_EKF_theory.py	# Evaluate average performance of 3 trajectories.
│   └── Visualize.py            	# 
│
├── KF_NCLT/				# Classical Kalman Filter baseline for NCLT
│   ├── KalmanFilter_NCLT.py		# Run Kalman filter on NCLT dataset
│   └── INPUT				# NCLT data for Kalman filter
│
├── KalmanNetPHAv2/            		# KalmanNet implementation (baseline comparison)
│   └── KalmanNet_TSP-main/
│       ├── main_linear_PHA.py		# Run KalmanNet for NCLT data
│
├── NF_NCLT/                  		# NCLT Dataset
│   └── EKF_NF_NCLT.py			# FEKF for NCLT experiments.
│   └── NF_NCLT_train.py		# Training model for NCLT experiments.
│
├── NCLT_data/                    	# Raw NCLT data
│
└── NF_RSS/                    
    ├── EKF_NF_RSS.py			# FEKF for RSS simulations
    └── NF_RSS_train.py			# Training model for RSS simulations
```

## 🔬 Method

### FlowEKF Architecture

FlowEKF extends the standard Extended Kalman Filter with a Real-NVP neural network to learn unknown observation models:

**Standard EKF Process:**
1. **Prediction Step**: Predict next state using motion model
2. **Update Step**: Correct prediction using observations

**FlowEKF Innovation:**
- Replaces analytical observation model h(·) with learned Real-NVP
- Real-NVP learns mapping: state → observations
- Outputs both predicted measurements and Jacobian matrix for Kalman gain

**Complexity Comparison:**
- KalmanNet: O(n⁴ + m⁴ + n³m + nm³ + n²m²)
- **FlowEKF**: O(n² + mn)

Where n = observation dimension, m = state dimension

## 💻 Main Files Usage

### 1. EKF_RSS - Indoor Simulation

**Purpose**: Evaluate classical Kalman filter on synthetic RSS (Received Signal Strength) data

**Data**: 
- Simulated 8×8m indoor environment with 4 beacons
- RSS values calculated using path loss model
- 10,000 training points, 3 test trajectories (linear, sinusoidal, spiral)

### 2. KF_NCLT - Baseline Kalman Filter

**Purpose**: Classical Kalman Filter baseline on real-world data

### 3. KalmanNetPHAv2 - Baseline Comparison

**Purpose**: KalmanNet baseline (uses RNN to learn Kalman gain)

**Key Files**:
- `KalmanNet_TSP-main/main_linear_PHA.py: The code train x and y axis separately.

### 4. NF_NCLT - FlowEKF for NCLT

**Purpose**: Main FlowEKF implementation for real-world data

**Key Files**:
- `NF_NCLT_train.py`: Provide real-NVP model and vanilla (the inverse model of real-NVP) model
- `EKF_NF_NCLT.py`: integrate trained model to evaluate FEKF.
  - Supports: same-day, cross-day, mixed-temporal training
- `evaluate.py`: Evaluate FEKF and compute RMSE

**Training Modes**:
- **Same-day**: Train and test on same session
- **Cross-day**: Train on one day, test on another
- **Mixed-temporal**: Train on multiple days

### 5. NCLT_data - Real-World Dataset

**Purpose**: Real robot navigation data for evaluation

**Data Contains**:
- **Odometry**: Wheel encoder measurements (noisy position estimates)
- **Ground truth**: Reference trajectory
- Used to test FlowEKF on real-world scenarios with unknown observation noise

**Sessions**:
- `20120122/`: January 22, 2012 session
- `20120429/`: April 29, 2012 session

### 6. NF_RSS - Neural Filter for RSS

**Purpose**: Evaluate FEKF on synthetic RSS (Received Signal Strength) data

**Key Files**:
- `EKF_NF_RSS`: FEKF on synthetic RSS.
- `NF_RSS_train.py`: Train flow-based model on RSS data

**Note**: This is the "Vanilla NF" mentioned in paper - uses only inverse Real-NVP without Kalman filtering

## 📊 Model Configuration

Default parameters used in experiments:

```python
# Real-NVP Architecture
hidden_layers = 4
hidden_dim = 64
activation = 'tanh'

# Training
optimizer = 'Adam'
learning_rate = 1e-3
weight_decay = 1e-5
gradient_clip = 1.0
iterations = 200000

# Data Split
train_split = 0.85
val_split = 0.10
test_split = 0.05
```

## 📖 Citation

```bibtex
@inproceedings{pham2025flowekf,
  title={FlowEKF: Flow-based Extended Kalman Filter},
  author={Pham, Hai Anh and Tran, Trong Duy and Do, Hai Son and Abed-Meraim, Karim and Nguyen, Linh Trung},
  booktitle={2025 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)},
  pages={1851--1856},
  year={2025},
  organization={IEEE}
}
```

## 👥 Authors

- **Pham Hai Anh** - University of Engineering and Technology, VNU Hanoi
- **Tran Trong Duy** - CentraleSup´elec, Université Paris-Saclay & VNU Hanoi
- **Do Hai Son** - Curtin University & VNU Hanoi
- **Karim Abed-Meraim** - PRISME Laboratory, University of Orléans
- **Nguyen Linh Trung** - University of Engineering and Technology, VNU Hanoi

**Contact**: linhtrung@vnu.edu.vn

## 🙏 Acknowledgments

This research was supported by project **QG.25.08** at Vietnam National University, Hanoi.

## 📄 License

This project is for research and educational purposes.

---

**Note**: For detailed experimental results and performance comparisons, please refer to the paper.

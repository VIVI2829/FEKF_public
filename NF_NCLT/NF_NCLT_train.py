import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
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

        inv_log_det_jac = ((1 - self.mask) * -s).sum(-1)
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


def train(model, input, output, X_val_tensor, y_val_tensor, iter=1000):

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=25)
    loss_function = nn.MSELoss()
    losses = []

    train_log_dir = './lossNCLT/train/v1_2201vs2904_4Aff_64hidd_pt_calibrate'
    # train_log_dir = './test/calibrated'
    val_log_dir = './lossNCLT/val/v1_2201vs2904_4Aff_64hidd_pt_calibrate'
    # val_log_dir = './test/calibrated_ver3'
    train_summary_writer = SummaryWriter(log_dir=train_log_dir)
    val_summary_writer = SummaryWriter(log_dir=val_log_dir)
    os.makedirs(train_log_dir, exist_ok=True)
    os.makedirs(val_log_dir, exist_ok=True)

    for i in range(iter):
        y = model(input)  # Forward
        inverted = model.invert(y)

        loss = loss_function(y, output)

        train_summary_writer.add_scalar('Loss/train', loss, i)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # losses.append(loss.item())

        if i % 100 == 0:
            train_summary_writer.add_scalar('Loss/train', loss, i)
            model.eval()
            with torch.no_grad():
                val_output = model(X_val_tensor)
                val_loss = loss_function(val_output, y_val_tensor)
                val_summary_writer.add_scalar('Loss/val', val_loss, i)
                print(f'Epoch {i}/{iter}, Loss: {loss:.9f}, Validation Loss: {val_loss:.9f}')
    train_summary_writer.close()
    val_summary_writer.close()
    return model

# validate trên 2904
val_input   = torch.load("data/2201/cal_val_groundtruth.pt")
val_output  = torch.load("data/2201/cal_val_odo.pt")
val_location_array = torch.tensor(val_input, dtype=torch.float64, device=device)
val_measurement_array = torch.tensor(val_output, dtype=torch.float64, device=device)

# Ghép train 2201 và 2904
location_array1 = torch.load("data/2201/cal_train_groundtruth.pt")
measurement_array1 = torch.load("data/2201/cal_train_odo.pt")
location_array2 = torch.load("data/2201/cal_train_groundtruth.pt")
measurement_array2 = torch.load("data/2201/cal_train_odo.pt")

location_array = torch.cat([torch.tensor(location_array1), torch.tensor(location_array2)], dim=0).to(device)
measurement_array = torch.cat([torch.tensor(measurement_array1), torch.tensor(measurement_array2)], dim=0).to(device)

layers = 4  # Number of masks (adjustable)

# Define the base masks
mask1 = torch.tensor([1, 0], dtype=torch.float64)
mask2 = torch.tensor([0, 1], dtype=torch.float64)

masks = [mask1 if i % 2 == 0 else mask2 for i in range(layers)]

masks = torch.stack(masks)

hidden_size = 64
NVP_model = RealNVP(masks, hidden_size, 2, 2)  # Initialize
NVP_model = NVP_model.double().to(device)

trained_model = train(NVP_model,torch.tensor(location_array, dtype=torch.float64), torch.tensor(measurement_array, dtype=torch.float64), torch.tensor(val_location_array, dtype=torch.float64), torch.tensor(val_measurement_array, dtype=torch.float64), 200000)
torch.save(trained_model, 'trained_model/v1_2201vs2904NVP_4Aff_64hidd_cal_200kiter.pth')
# torch.save(trained_model, 'trained_model/testsave')




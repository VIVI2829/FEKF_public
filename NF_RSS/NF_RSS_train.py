import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
device = torch.device("cpu")
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class RealNVPNode(nn.Module):
    def __init__(self, mask, hidden_size, dropout_p=0.2):  # thêm dropout_p
        super(RealNVPNode, self).__init__()
        self.dim = len(mask)
        self.mask = nn.Parameter(mask, requires_grad=False)

        self.s_func = nn.Sequential(
            nn.Linear(in_features=self.dim, out_features=hidden_size), nn.Tanh(), nn.Dropout(dropout_p),
            nn.Linear(in_features=hidden_size, out_features=hidden_size), nn.Tanh(), nn.Dropout(dropout_p),
            nn.Linear(in_features=hidden_size, out_features=hidden_size), nn.Tanh(), nn.Dropout(dropout_p),
            nn.Linear(in_features=hidden_size, out_features=self.dim)
        )

        self.scale = nn.Parameter(torch.zeros(self.dim))

        self.t_func = nn.Sequential(
            nn.Linear(in_features=self.dim, out_features=hidden_size), nn.Tanh(), nn.Dropout(dropout_p),
            nn.Linear(in_features=hidden_size, out_features=hidden_size), nn.Tanh(), nn.Dropout(dropout_p),
            nn.Linear(in_features=hidden_size, out_features=hidden_size), nn.Tanh(), nn.Dropout(dropout_p),
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
    def __init__(self, input, masks, hidden_size, input_dim=2, output_dim=4, dropout_p=0.2):  # thêm dropout_p
        super(RealNVP, self).__init__()

        self.output_dim = output_dim
        self.hidden_size = hidden_size

        # Linear transformation to upscale input (2x1 → 4x1)
        self.upscale_net = nn.Linear(input_dim, output_dim)

        self.masks = masks
        self.layers = nn.ModuleList([
            RealNVPNode(mask, self.hidden_size, dropout_p=dropout_p) for mask in self.masks
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
        W_inv = torch.linalg.pinv(W)
        x = torch.matmul(W_inv, (y - b).T).T
        return x

    def invert(self, output):
        for layer in reversed(self.layers):
            output = layer.inverse(output)  # Apply inverse of each RealNVPNode
        downscale = self.invert_linear_layer(self.upscale_net, output)
        return downscale


def train(model, input, output, X_val_tensor, y_val_tensor, iter=1000):
    # Thêm weight_decay để regularize
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    # , weight_decay=1e-4
    loss_function = nn.MSELoss()
    losses = []

    train_log_dir = './loss_8x8_4noise/v39'
    val_log_dir = './loss_8x8_4noise/v39'
    # train_log_dir = './test/v17'
    # val_log_dir = './test/v17'
    train_summary_writer = SummaryWriter(log_dir=train_log_dir)
    val_summary_writer = SummaryWriter(log_dir=val_log_dir)
    os.makedirs(train_log_dir, exist_ok=True)

    for i in range(iter):
        y = model(input)  # Forward

        loss = loss_function(y, output)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())

        if i % 10 == 0:
            train_summary_writer.add_scalar('Loss/train', loss, i)
            model.eval()
            with torch.no_grad():
                val_output = model(X_val_tensor)
                val_loss = loss_function(val_output, y_val_tensor)
                # scheduler.step(val_loss)
                val_summary_writer.add_scalar('Loss/val', val_loss, i)
                print(f'Epoch {i}/{iter}, Loss: {loss:.9f}, Validation Loss: {val_loss:.9f}')
    train_summary_writer.close()
    val_summary_writer.close()
    return model, losses



def transform(points, std):
    # X = torch.tensor(X, dtype=torch.float32, requires_grad=True) if not torch.is_tensor(X) else X
    # Y = torch.tensor(Y, dtype=torch.float32, requires_grad=True) if not torch.is_tensor(Y) else Y
    #
    # points = torch.stack([X, Y], dim=-1) if X.ndim == 0 else torch.stack([X, Y], dim=-1)

    pos_X = torch.tensor([0, 8], dtype=torch.float32)
    pos_Y = torch.tensor([0, 8], dtype=torch.float32)
    predefined_positions = torch.stack([
        torch.tensor([pos_X[0], pos_Y[0]]),
        torch.tensor([pos_X[1], pos_Y[0]]),
        torch.tensor([pos_X[0], pos_Y[1]]),
        torch.tensor([pos_X[1], pos_Y[1]])
    ])
    # a.unsqueeze(1) - b.unsqueeze(0)
    distances = torch.norm(points.unsqueeze(1) - predefined_positions.unsqueeze(0), dim=-1) + 1e-6

    d0 = torch.tensor(1.0, dtype=torch.float32)
    power_transmit = torch.tensor(-14.32, dtype=torch.float32)
    frequency = torch.tensor(3993.6 * 1e6, dtype=torch.float32)
    wavelength = torch.tensor(3e8, dtype=torch.float32) / frequency
    K_dB = 20 * torch.log10(wavelength / (4 * torch.pi * d0))
    pathloss_exponent = torch.tensor(3.0, dtype=torch.float32)
    shadowfading_deviation_dB = torch.normal(mean=torch.tensor(0, dtype=torch.float32), std=torch.tensor(std, dtype=torch.float32), size=distances.shape)

    pathloss_attenuation_dB = 10 * pathloss_exponent * torch.log10(distances / d0)

    power_received = power_transmit + K_dB - pathloss_attenuation_dB - shadowfading_deviation_dB
    return power_received

# DATA GENERATION
std = 1.5
samples = 10
x = torch.linspace(0, 8, samples)
y = torch.linspace(0, 8, samples)
X, Y = torch.meshgrid(x, y, indexing='ij')
points = torch.stack((X, Y), dim=-1)
points_flat = points.view(-1, 2)
# print(points_flat.shape)
Power_received = transform(points_flat, std)
total_count = Power_received.shape[0]
train_count = int(0.6 * total_count)
val_count = int(0.2 * total_count)
test_count = total_count - train_count - val_count
def subset_to_tensor(subset):
    return torch.stack([subset.dataset[i] for i in subset.indices])
X_train, X_val, X_test = torch.utils.data.random_split(
    points_flat, (train_count, val_count, test_count),torch.Generator().manual_seed(42))
y_train, y_val, y_test = torch.utils.data.random_split(
    Power_received, (train_count, val_count, test_count), torch.Generator().manual_seed(42))

X_train_tensor = torch.tensor(subset_to_tensor(X_train), dtype = torch.float32, requires_grad=False).to(device)
print("X_train size: ",X_train_tensor.size())
X_val_tensor = torch.tensor(subset_to_tensor(X_val), dtype = torch.float32, requires_grad=False).to(device)
# X_test_tensor = torch.tensor(subset_to_tensor(X_test), dtype = torch.float32, requires_grad=False).to(device)

y_train_tensor = torch.tensor(subset_to_tensor(y_train), dtype = torch.float32, requires_grad=False).to(device)
print("Y_train size: ",y_train_tensor.size())
y_val_tensor = torch.tensor(subset_to_tensor(y_val), dtype = torch.float32, requires_grad=False).to(device)
# y_test_tensor = torch.tensor(subset_to_tensor(y_test), dtype = torch.float32, requires_grad=False).to(device)

layers = 4  # Number of masks (adjustable)

# Define the base masks
mask1 = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
mask2 = torch.tensor([0, 1, 0, 1], dtype=torch.float32)

masks = [mask1 if i % 2 == 0 else mask2 for i in range(layers)]

# masks = torch.stack(masks)

hidden_size = 32
NVP_model = RealNVP(points_flat, masks, hidden_size, 2, 4, 0.1)  # Initialize
trained_model, loss = train(NVP_model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, 10001)
model_name = "trained_model/v39NVP_4Aff_4hiddlayer_64hid_size8x8_10x10samples_10001_drop0_decay0_6train2val.pth"
print(model_name)
torch.save(trained_model, model_name)

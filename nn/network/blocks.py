import torch as th
import torch.nn.functional as F
from unet import UNet, ShallowUNet


class ConvEncoder(th.nn.Module):

    def __init__(self, input_shape, n_objs, num_outputs, hidden_dim=200, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_shape = input_shape
        self.n_objs = n_objs
        self.hidden_dim = hidden_dim
        self.num_outputs = num_outputs

        self.unet = ShallowUNet(in_channels=input_shape[0], base_channels=8, out_channels=n_objs, upsample=True)
        self.dense0 = th.nn.Linear(input_shape[0] * input_shape[1] * input_shape[2], hidden_dim)
        self.dense1 = th.nn.Linear(hidden_dim, hidden_dim)
        self.dense2 = th.nn.Linear(hidden_dim, num_outputs)

        self.enc_masks = None
        self.masked_objs = None

    def forward(self, x):
        h = self.unet(x)
        h = th.concat([h, th.ones_like(h[:, :1, :, :])], dim=1)
        h = F.softmax(h, dim=1)

        self.enc_masks = h
        self.masked_objs = [self.enc_masks[:, i:i+1, :, :] * x for i in range(self.n_objs)]

        h = th.concat(self.masked_objs, dim=0)
        h = th.flatten(h, start_dim=1)
        h = F.relu(self.dense0(h))
        h = F.relu(self.dense1(h))
        h = self.dense2(h)

        # in PyTorch the size of each split, in TF the number of splits
        h = th.split(h, h.shape[0] // self.n_objs, dim=0)
        h = th.concat(h, dim=1)
        h = F.tanh(h) * (self.input_shape[1] / 2) + (self.input_shape[1] / 2)
        return h


# only support the version with alt_vel=False (for now)
class VelocityEncoder(th.nn.Module):

    def __init__(self, n_objs, input_steps, coord_units, hidden_dim=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_objs = n_objs
        self.input_steps = input_steps
        self.coord_units = coord_units
        self.hidden_dim = hidden_dim

        self.dense0 = th.nn.Linear(input_steps * coord_units // n_objs // 2, hidden_dim)
        self.dense1 = th.nn.Linear(hidden_dim, hidden_dim)
        self.dense2 = th.nn.Linear(hidden_dim, coord_units // n_objs // 2)

    def forward(self, x):
        # assuming cartesian (2D) coordinates
        # x: B x input_steps x (2 * n_objs)
        h = th.split(x, x.shape[2] // self.n_objs, dim=2)
        h = th.concat(h, dim=0)  # h: n_objs * B x input_steps x 2
        h = th.flatten(h, start_dim=1)  # instead of the .resize from original code (same output)

        h = F.tanh(self.dense0(h))
        h = F.tanh(self.dense1(h))
        h = self.dense2(h)  # h: n_objs * B x 2

        h = th.split(h, h.shape[0] // self.n_objs, dim=0)
        h = th.concat(h, dim=1)  # B x 2 * n_objs
        return h


net = VelocityEncoder(3, 4, 12)
x = th.randn((5, 4, 2 * 3))
print(net(x).shape)

# from six.moves import xrange
# import tensorflow as tf
import torch as th
import torch.nn.functional as F


# implementation from https://github.com/vicsesi/PyTorch-STN/blob/main/src/net.py
class SpatialTransformer(th.nn.Module):

    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = device

    def repeat(self, x, n_repeats):
        rep = th.ones((1, n_repeats)).long()
        x = th.matmul(x.view(-1, 1), rep)
        return x.view(-1)


    def forward(self, img, theta, shape):
        # sadly the "normal" implementation of the spatial transformer does not work here
        num_batch = img.shape[0]
        channels = img.shape[1]
        height = img.shape[2]
        width = img.shape[3]

        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, shape, align_corners=True).float()
        grid = grid.permute(0, 3, 1, 2).view(img.shape[0], 2, -1)

        x_s = grid[:, 0, :]
        y_s = grid[:, 1, :]
        x_s_flat = x_s.view(-1)
        y_s_flat = y_s.view(-1)

        x = x_s_flat.float()
        y = y_s_flat.float()

        out_height = shape[2]
        out_width = shape[3]
        max_y = img.shape[2] - 1
        max_x = img.shape[3] - 1

        x = (x + 1.0) * (width - 1.01) / 2.0
        y = (y + 1.0) * (height - 1.01) / 2.0

        x0 = th.floor(x).int()
        x1 = x0 + 1
        y0 = th.floor(y).int()
        y1 = y0 + 1

        x0 = th.clamp(x0, 0, max_x)
        x1 = th.clamp(x1, 0, max_x)
        y0 = th.clamp(y0, 0, max_y)
        y1 = th.clamp(y1, 0, max_y)

        dim2 = width
        dim1 = width * height

        arange = th.arange(start=0, end=num_batch).long()
        base = self.repeat(arange * dim1, out_height * out_width).long().to(self.device)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2

        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        img_flat = img.permute(0, 2, 3, 1).reshape((-1, channels)).float()
        idx_a = th.broadcast_to(idx_a.unsqueeze(1), (idx_a.shape[0], channels))
        img_a = th.gather(img_flat, dim=0, index=idx_a)
        idx_b = th.broadcast_to(idx_b.unsqueeze(1), (idx_b.shape[0], channels))
        img_b = th.gather(img_flat, dim=0, index=idx_b)
        idx_c = th.broadcast_to(idx_c.unsqueeze(1), (idx_c.shape[0], channels))
        img_c = th.gather(img_flat, dim=0, index=idx_c)
        idx_d = th.broadcast_to(idx_d.unsqueeze(1), (idx_d.shape[0], channels))
        img_d = th.gather(img_flat, dim=0, index=idx_d)

        x0f, x1f, y0f, y1f = x0.float(), x1.float(), y0.float(), y1.float()
        wa = ((x1f-x) * (y1f-y)).unsqueeze(1)
        wb = ((x1f-x) * (y-y0f)).unsqueeze(1)
        wc = ((x-x0f) * (y1f-y)).unsqueeze(1)
        wd = ((x-x0f) * (y-y0f)).unsqueeze(1)

        a = wa * img_a
        b = wb * img_b
        c = wc * img_c
        d = wd * img_d

        output = sum([a, b, c, d])
        output = output.view(num_batch, out_height, out_width, channels).permute(0, 3, 1, 2)
        return output

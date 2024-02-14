import torch as th
import torch.nn.functional as F

class PhysicsNet(th.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
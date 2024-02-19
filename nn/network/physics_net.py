import torch as th
import torch.nn.functional as F
from nn.network.blocks import ConvEncoder, VelocityEncoder, ConvSTDecoder, VariableFromNetwork
from nn.network.cells import SpringODECell


class TaskSetup(object):

    def __init__(self, n_objs, num_coords):
        self.n_objs = n_objs
        self.num_coords = num_coords


# total number of latent units for each datasets
# coord_units = num_objects*num_dimensions*2
TASK_SETUPS = {
    "bouncing_balls": TaskSetup(2, 2),
    "spring_color": TaskSetup(2, 2),
    "spring_color_half": TaskSetup(2, 2),
    "3bp_color": TaskSetup(3, 2),
    "mnist_spring_color": TaskSetup(2, 2)
}


class PhysicsNet(th.nn.Module):

    def __init__(self,
                 task,
                 input_shape,
                 seq_len=20,
                 input_steps=3,
                 pred_steps=5,
                 device='cpu',
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.task = task
        self.input_shape = input_shape  # for RGB 32x32 is: (3, 32, 32)
        self.seq_len = seq_len
        self.input_steps = input_steps
        self.pred_steps = pred_steps
        self.device = device

        self.task_setup = TASK_SETUPS[task]
        self.extrap_steps = self.seq_len - self.input_steps - self.pred_steps

        self.cell = SpringODECell()  # make configurable later
        self.conv_encoder = ConvEncoder(input_shape=input_shape,
                                        n_objs=self.task_setup.n_objs,
                                        num_outputs=self.task_setup.num_coords)
        self.velocity_encoder = VelocityEncoder(n_objs=self.task_setup.n_objs,
                                                input_steps=self.input_steps,
                                                num_outputs=self.task_setup.num_coords)
        self.conv_st_decoder = ConvSTDecoder(self.input_shape,
                                             n_objs=self.task_setup.n_objs,
                                             device=self.device)

        self.recons_out = None
        self.enc_pos = None

    def forward(self, x):
        h = th.reshape(x[:, :self.input_steps + self.pred_steps], (-1, *self.input_shape))
        enc_pos = self.conv_encoder(h)

        recons_out = self.conv_st_decoder(enc_pos)

        self.recons_out = th.reshape(recons_out, (x.shape[0], self.input_steps + self.pred_steps, *self.input_shape))
        self.enc_pos = th.reshape(enc_pos, (x.shape[0], self.input_steps + self.pred_steps,
                                            self.task_setup.n_objs * self.task_setup.num_coords))

        # only assume input_steps > 1 for now
        vel = self.velocity_encoder(self.enc_pos[:, :self.input_steps])
        pos = self.enc_pos[:, self.input_steps + 1]

        output_seq = []
        pos_vel_seq = [th.concat([pos, vel], dim=1)]
        for _ in range(self.pred_steps + self.extrap_steps):
            pos, vel = self.cell(pos, vel)
            out = self.conv_st_decoder(pos)

            pos_vel_seq.append(th.concat([pos, vel], dim=1))
            output_seq.append(out)

        output_seq = th.stack(output_seq)
        pos_vel_seq = th.stack(pos_vel_seq)
        output_seq = th.transpose(output_seq, 0, 1)
        pos_vel_seq = th.transpose(pos_vel_seq, 0, 1)
        return output_seq


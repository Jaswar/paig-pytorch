import numpy as np
# import tensorflow as tf
import torch as th
import torch.nn.functional as F


""" Useful subnetwork components """


# def unet(inp, base_channels, out_channels, upsamp=True):
#     h = inp
#     h = tf.layers.conv2d(h, base_channels, 3, activation=tf.nn.relu, padding="SAME")  #
#     h1 = tf.layers.conv2d(h, base_channels, 3, activation=tf.nn.relu, padding="SAME")  #
#     h = tf.layers.max_pooling2d(h1, 2, 2)  #
#     h = tf.layers.conv2d(h, base_channels*2, 3, activation=tf.nn.relu, padding="SAME")  #
#     h2 = tf.layers.conv2d(h, base_channels*2, 3, activation=tf.nn.relu, padding="SAME")  #
#     h = tf.layers.max_pooling2d(h2, 2, 2)  #
#     h = tf.layers.conv2d(h, base_channels*4, 3, activation=tf.nn.relu, padding="SAME")  #
#     h3 = tf.layers.conv2d(h, base_channels*4, 3, activation=tf.nn.relu, padding="SAME")  #
#     h = tf.layers.max_pooling2d(h3, 2, 2)  #
#     h = tf.layers.conv2d(h, base_channels*8, 3, activation=tf.nn.relu, padding="SAME")  #
#     h4 = tf.layers.conv2d(h, base_channels*8, 3, activation=tf.nn.relu, padding="SAME")  #
#     if upsamp:
#         h = tf.image.resize_bilinear(h4, h3.get_shape()[1:3])
#         h = tf.layers.conv2d(h, base_channels*2, 3, activation=None, padding="SAME")  #
#     else:
#         h = tf.layers.conv2d_transpose(h, base_channels*4, 3, 2, activation=None, padding="SAME")  #
#     h = tf.concat([h, h3], axis=-1)  #
#     h = tf.layers.conv2d(h, base_channels*4, 3, activation=tf.nn.relu, padding="SAME")  #
#     h = tf.layers.conv2d(h, base_channels*4, 3, activation=tf.nn.relu, padding="SAME")  #
#     if upsamp:
#         h = tf.image.resize_bilinear(h, h2.get_shape()[1:3])
#         h = tf.layers.conv2d(h, base_channels*2, 3, activation=None, padding="SAME")  #
#     else:
#         h = tf.layers.conv2d_transpose(h, base_channels*2, 3, 2, activation=None, padding="SAME")  #
#     h = tf.concat([h, h2], axis=-1)
#     h = tf.layers.conv2d(h, base_channels*2, 3, activation=tf.nn.relu, padding="SAME")  #
#     h = tf.layers.conv2d(h, base_channels*2, 3, activation=tf.nn.relu, padding="SAME")  #
#     if upsamp:
#         h = tf.image.resize_bilinear(h, h1.get_shape()[1:3])  #
#         h = tf.layers.conv2d(h, base_channels*2, 3, activation=None, padding="SAME")  #
#     else:
#         h = tf.layers.conv2d_transpose(h, base_channels, 3, 2, activation=None, padding="SAME")
#     h = tf.concat([h, h1], axis=-1)
#     h = tf.layers.conv2d(h, base_channels, 3, activation=tf.nn.relu, padding="SAME")
#     h = tf.layers.conv2d(h, base_channels, 3, activation=tf.nn.relu, padding="SAME")
#
#     h = tf.layers.conv2d(h, out_channels, 1, activation=None, padding="SAME")
#     return h

def pad(x1, x2):
    diff_0, diff_1 = x2.shape[2] - x1.shape[2], x2.shape[3] - x1.shape[3]
    return F.pad(x1, (diff_0 // 2, diff_0 - diff_0 // 2, diff_1 // 2, diff_1 - diff_1 // 2))


def initialize_convolutions(layers):
    for layer in layers:
        if layer.weight is not None:
            th.nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            th.nn.init.zeros_(layer.bias)


class UNet(th.nn.Module):

    def __init__(self, in_channels, base_channels, out_channels, upsample=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.out_channels = out_channels
        self.upsample = upsample

        self.conv0_0 = th.nn.Conv2d(in_channels, base_channels, kernel_size=3, padding='same')
        self.conv0_1 = th.nn.Conv2d(base_channels, base_channels, kernel_size=3, padding='same')
        self.mp0 = th.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1_0 = th.nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding='same')
        self.conv1_1 = th.nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding='same')
        self.mp1 = th.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_0 = th.nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding='same')
        self.conv2_1 = th.nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding='same')
        self.mp2 = th.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_0 = th.nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding='same')
        self.conv3_1 = th.nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, padding='same')

        if upsample:
            self.upsample0 = th.nn.Conv2d(base_channels * 8, base_channels * 2, kernel_size=3, padding='same')
            self.conv4_0 = th.nn.Conv2d(base_channels * (2 + 4), base_channels * 4, kernel_size=3, padding='same')
        else:
            self.upsample0 = th.nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=3, stride=2)
            self.conv4_0 = th.nn.Conv2d(base_channels * (4 + 4), base_channels * 4, kernel_size=3, padding='same')
        self.conv4_1 = th.nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding='same')

        if upsample:
            self.upsample1 = th.nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding='same')
            self.conv5_0 = th.nn.Conv2d(base_channels * (2 + 2), base_channels * 2, kernel_size=3, padding='same')
        else:
            self.upsample1 = th.nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=3, stride=2)
            self.conv5_0 = th.nn.Conv2d(base_channels * (2 + 2), base_channels * 2, kernel_size=3, padding='same')
        self.conv5_1 = th.nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding='same')

        if upsample:
            self.upsample2 = th.nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding='same')
            self.conv6_0 = th.nn.Conv2d(base_channels * (2 + 1), base_channels, kernel_size=3, padding='same')
        else:
            self.upsample2 = th.nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=3, stride=2)
            self.conv6_0 = th.nn.Conv2d(base_channels * (1 + 1), base_channels, kernel_size=3, padding='same')
        self.conv6_1 = th.nn.Conv2d(base_channels, base_channels, kernel_size=3, padding='same')

        self.conv7 = th.nn.Conv2d(base_channels, out_channels, kernel_size=1, padding='same')

    def forward(self, x):
        h = x
        h1 = F.relu(self.conv0_1(F.relu(self.conv0_0(h))))
        h = self.mp0(h1)
        h2 = F.relu(self.conv1_1(F.relu(self.conv1_0(h))))
        h = self.mp1(h2)
        h3 = F.relu(self.conv2_1(F.relu(self.conv2_0(h))))
        h = self.mp2(h3)

        h = F.relu(self.conv3_0(h))
        if self.upsample:
            h4 = F.relu(self.conv3_1(h))
            h = F.interpolate(h4, h3.shape[2:], mode='bilinear', align_corners=True)
            h = self.upsample0(h)
        else:
            h = self.upsample0(h)
            h = pad(h, h3)
        h = th.concat([h, h3], dim=1)

        h = F.relu(self.conv4_1(F.relu(self.conv4_0(h))))
        if self.upsample:
            h = F.interpolate(h, h2.shape[2:], mode='bilinear', align_corners=True)
            h = self.upsample1(h)
        else:
            h = self.upsample1(h)
            h = pad(h, h2)
        h = th.concat([h, h2], dim=1)

        h = F.relu(self.conv5_1(F.relu(self.conv5_0(h))))
        if self.upsample:
            h = F.interpolate(h, h1.shape[2:], mode='bilinear', align_corners=True)
            h = self.upsample2(h)
        else:
            h = self.upsample2(h)
            h = pad(h, h1)
        h = th.concat([h, h1], dim=1)

        h = F.relu(self.conv6_1(F.relu(self.conv6_0(h))))

        h = self.conv7(h)
        return h


# def shallow_unet(inp, base_channels, out_channels, upsamp=True):
#     h = inp
#     h = tf.layers.conv2d(h, base_channels, 3, activation=tf.nn.relu, padding="SAME")  #
#     h1 = tf.layers.conv2d(h, base_channels, 3, activation=tf.nn.relu, padding="SAME")  #
#     h = tf.layers.max_pooling2d(h1, 2, 2)  #
#     h = tf.layers.conv2d(h, base_channels*2, 3, activation=tf.nn.relu, padding="SAME")  #
#     h2 = tf.layers.conv2d(h, base_channels*2, 3, activation=tf.nn.relu, padding="SAME")  #
#     h = tf.layers.max_pooling2d(h2, 2, 2)  #
#     h = tf.layers.conv2d(h, base_channels*4, 3, activation=tf.nn.relu, padding="SAME")  #
#     h = tf.layers.conv2d(h, base_channels*4, 3, activation=tf.nn.relu, padding="SAME")  #
#     if upsamp:
#         h = tf.image.resize_bilinear(h, h2.get_shape()[1:3])
#         h = tf.layers.conv2d(h, base_channels*2, 3, activation=None, padding="SAME")
#     else:
#         h = tf.layers.conv2d_transpose(h, base_channels*2, 3, 2, activation=None, padding="SAME")
#     h = tf.concat([h, h2], axis=-1)
#     h = tf.layers.conv2d(h, base_channels*2, 3, activation=tf.nn.relu, padding="SAME")
#     h = tf.layers.conv2d(h, base_channels*2, 3, activation=tf.nn.relu, padding="SAME")
#     if upsamp:
#         h = tf.image.resize_bilinear(h, h1.get_shape()[1:3])
#         h = tf.layers.conv2d(h, base_channels*2, 3, activation=None, padding="SAME")
#     else:
#         h = tf.layers.conv2d_transpose(h, base_channels, 3, 2, activation=None, padding="SAME")
#     h = tf.concat([h, h1], axis=-1)
#     h = tf.layers.conv2d(h, base_channels, 3, activation=tf.nn.relu, padding="SAME")
#     h = tf.layers.conv2d(h, base_channels, 3, activation=tf.nn.relu, padding="SAME")
#
#     h = tf.layers.conv2d(h, out_channels, 1, activation=None, padding="SAME")
#     return h

class ShallowUNet(th.nn.Module):

    def __init__(self, in_channels, base_channels, out_channels, upsample=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.out_channels = out_channels
        self.upsample = upsample

        self.conv0_0 = th.nn.Conv2d(in_channels, base_channels, kernel_size=3, padding='same')
        self.conv0_1 = th.nn.Conv2d(base_channels, base_channels, kernel_size=3, padding='same')
        self.mp0 = th.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1_0 = th.nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding='same')
        self.conv1_1 = th.nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding='same')
        self.mp1 = th.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_0 = th.nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding='same')
        self.conv2_1 = th.nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding='same')

        if upsample:
            self.upsample0 = th.nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding='same')
            self.conv3_0 = th.nn.Conv2d(base_channels * (2 + 2), base_channels * 2, kernel_size=3, padding='same')
        else:
            self.upsample0 = th.nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=3, stride=2)
            self.conv3_0 = th.nn.Conv2d(base_channels * (2 + 2), base_channels * 2, kernel_size=3, padding='same')
        self.conv3_1 = th.nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding='same')

        if upsample:
            self.upsample1 = th.nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding='same')
            self.conv4_0 = th.nn.Conv2d(base_channels * (2 + 1), base_channels, kernel_size=3, padding='same')
        else:
            self.upsample1 = th.nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=3, stride=2)
            self.conv4_0 = th.nn.Conv2d(base_channels * (1 + 1), base_channels, kernel_size=3, padding='same')
        self.conv4_1 = th.nn.Conv2d(base_channels, base_channels, kernel_size=3, padding='same')

        self.conv5 = th.nn.Conv2d(base_channels, out_channels, kernel_size=1, padding='same')

        initialize_convolutions([self.conv0_0, self.conv0_1, self.conv1_0, self.conv1_1,
                                 self.conv2_0, self.conv2_1, self.conv3_0, self.conv3_1,
                                 self.conv4_0, self.conv4_1, self.conv5,
                                 self.upsample0, self.upsample1])

    def forward(self, x):
        h = x
        h1 = F.relu(self.conv0_1(F.relu(self.conv0_0(h))))
        h = self.mp0(h1)
        h2 = F.relu(self.conv1_1(F.relu(self.conv1_0(h))))
        h = self.mp1(h2)

        h = F.relu(self.conv2_1(F.relu(self.conv2_0(h))))
        if self.upsample:
            h = F.interpolate(h, h2.shape[2:], mode='bilinear', align_corners=True)
            h = self.upsample0(h)
        else:
            h = self.upsample0(h)
            h = pad(h, h2)
        h = th.concat([h, h2], dim=1)

        h = F.relu(self.conv3_1(F.relu(self.conv3_0(h))))
        if self.upsample:
            h = F.interpolate(h, h1.shape[2:], mode='bilinear', align_corners=True)
            h = self.upsample1(h)
        else:
            h = self.upsample1(h)
            h = pad(h, h1)
        h = th.concat([h, h1], dim=1)

        h = F.relu(self.conv4_1(F.relu(self.conv4_0(h))))

        h = self.conv5(h)
        return h



# def variable_from_network(shape):
#     # Produces a variable from a vector of 1's.
#     # Improves learning speed of contents and masks.
#     var = tf.ones([1,10])
#     var = tf.layers.dense(var, 200, activation=tf.tanh)
#     var = tf.layers.dense(var, np.prod(shape), activation=None)
#     var = tf.reshape(var, shape)
#     return var

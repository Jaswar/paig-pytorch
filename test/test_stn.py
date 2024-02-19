from nn.network.stn import SpatialTransformer
from test.tf_impl.stn import stn
import numpy as np
import torch as th
import tensorflow.compat.v1 as tf


def test_stn_model():
    torch_stn = SpatialTransformer()
    tf_stn = stn

    np.random.seed(42)

    shape = (5, 3, 32, 32)
    x = np.random.normal(size=(5, 3, 16, 16))
    theta = np.random.normal(size=(5, 6))

    torch_shape = shape
    torch_x = th.from_numpy(x).float()
    torch_theta = th.from_numpy(theta)

    tf_shape = (shape[2], shape[3])
    tf_x = tf.transpose(tf.convert_to_tensor(x), perm=[0, 2, 3, 1])
    tf_theta = tf.convert_to_tensor(theta)

    torch_out = torch_stn(torch_x, torch_theta, torch_shape).numpy()
    tf_out = tf.transpose(tf_stn(tf_x, tf_theta, tf_shape), perm=[0, 3, 1, 2]).numpy()

    # count = 0
    # for torch_el, tf_el in zip(torch_out.flatten(), tf_out.flatten()):
    #     if abs(torch_el - tf_el) > 0.0001:
    #         print(torch_el, tf_el)
    #         count += 1
    # print(count)
    print(np.allclose(torch_out, tf_out, atol=1e-4))

test_stn_model('cpu')


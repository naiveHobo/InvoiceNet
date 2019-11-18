from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib


def print_vars(vars):
    total = 0
    for var in vars:
        shape = tf.shape(var)
        print(var.name, shape)
        total += np.prod(shape)
    print(total)


def get_devices():
    gpus = [x.name for x in (device_lib.list_local_devices()) if x.device_type == 'GPU']
    if len(gpus) > 0:
        devices = gpus
    else:
        print("WARNING: No GPU's found. Using CPU")
        devices = ['cpu:0', 'cpu:0']

    print("Using devices: ", devices)
    return devices


def batch_parallel(map_fn, devices, **kwargs):
    """
    Parallelize map_fn across devices.

    :param map_fn: function that takes kwargs and returns a tuple of tensors
    :param devices: A list of devices to parallelize over
    :param kwargs: kwargs of input to map, will be split along axis 0.
    :return: The outputs of map_fn
    The inner list is the output from each device.
    """
    in_splits = {}
    for k, v in kwargs.items():
        in_splits[k] = tf.split(v, len(devices))

    out_splits = {}
    for i, device in enumerate(devices):
        with tf.device(device):
            with tf.variable_scope(tf.get_variable_scope(), reuse=i > 0):
                outs = map_fn(**{k: v[i] for k, v in in_splits.items()})
                for j, out in enumerate(outs):
                    if j not in out_splits:
                        out_splits[j] = []
                    out_splits[j].append(out)

    return [out_splits[i] for i in range(len(out_splits))]

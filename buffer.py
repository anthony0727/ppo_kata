from typing import OrderedDict

import numpy as np
import torch


def _infer_dtype(example):
    l = []
    for k, v in example.items():
        if hasattr(v, 'dtype'):
            _dtype = v.dtype
        else:
            _dtype = type(v)
            if _dtype == bool:
                _dtype = np.uint8

        if hasattr(v, 'shape'):
            _shape = v.shape
        else:
            # consider it's scalar, edge case?
            _shape = ()

        l.append((k, _dtype, _shape))

    return l


class Buffer(np.ndarray):
    # (s, r, d) -> env -> a pair
    default_keys = ['obs', 'action', 'reward', 'done']

    """
    implement circular queue
    pre-allocate memory
    overwrite from 0 index when curr_index hits maxsize
    """

    # if you change parameter order in __init__, then change here too
    # any better idea? guess new passes its arguments to init
    def __new__(
            cls,
            num_transitions: int,
            example: OrderedDict,
            extras: OrderedDict = {},
            seed=42,
            *args,
            **kwargs
    ):
        assert all([k in example.keys() for k in Buffer.default_keys])

        # type check too strong?
        # assert isinstance(example['obs'], np.ndarray)
        # or example['obs'].__class__.__module__ == 'builtins'

        dtype_list = _infer_dtype(example)
        extra_dtype_list = list(extras.items())

        dtype = np.dtype(dtype_list + extra_dtype_list)

        # add 1 for last value
        max_size = num_transitions + 1
        obj = super().__new__(cls, shape=max_size, dtype=dtype)
        obj.seed = seed
        obj.num_transitions = num_transitions

        return obj

    def __array_finalize__(self, obj):
        self.curr_idx = 0

    def reset(self):
        # reset will contain last transition from previous truncated episode
        self.curr_idx = 0
        self[0] = self[-1].copy()

    def is_full(self):
        return self.curr_idx == (len(self) - 1)

    def insert(self, idx, transition):
        for k, v in transition.items():
            self[k][idx] = v

    def append(self, **kwargs):
        self.insert(self.curr_idx, kwargs)
        self.curr_idx += 1

        if self.is_full():
            self.reset()

    @property
    def all_keys(self):
        return list(self.dtype.fields.keys())

    def _as_dict(self, cls_callback):
        d = {}
        for k in self.all_keys:
            d[k] = cls_callback(self[k])
        return d

    def as_dict(self):
        callback = np.array

        return self._as_dict(callback)

    def as_tensor_dict(self, device='cpu'):

        return self._as_dict(
            lambda x: torch.from_numpy(np.array(x)).to(device)
        )

    def get_tensor(self, key, device='cpu'):

        return torch.from_numpy(np.array(self[key])).to(device)
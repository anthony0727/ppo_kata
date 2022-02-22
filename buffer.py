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

        # add 1 for initial state
        max_size = num_transitions + 1
        obj = super().__new__(cls, shape=max_size, dtype=dtype)
        obj.seed = seed
        obj.num_transitions = num_transitions

        return obj

    def __array_finalize__(self, obj):
        self.curr_idx = -1

    def reset(self):
        self.curr_idx = -1

    def is_full(self):
        return self.curr_idx == (len(self) - 1)

    def append(self, **kwargs):
        if self.is_full():
            self.reset()

        self.curr_idx += 1
        for k, v in kwargs.items():
            self[k][self.curr_idx] = v

    @property
    def all_keys(self):
        return list(self.dtype.fields.keys())

    def as_tensor(self, device='cuda'):
        t = {}
        for k in self.all_keys:
            t[k] = torch.from_numpy(np.array(self[k])).to(device)

        return t

    def get_tensor(self, key, device='cuda'):
        return torch.from_numpy(np.array(self[key])).to(device)

    # def _sample_idxes(self, size):
    #     if not size:
    #         size = self.curr_idx
    #
    #     assert 0 <= self.curr_idx, 'buffer is empty'
    #     assert size <= self.curr_idx + 1, 'cannot sample larger than buffer ' \
    #                                       'size'
    #
    #     # range is exclusive
    #     idxes = np.arange(0, self.curr_idx)
    #
    #     return idxes

    # def sample(self, size=None):
    #     #         dtype = self._sample_dtype()
    #     #         data = np.array(data, dtype=dtype)
    #     idxes = self._sample_idxes(size)
    #
    #     t = {}
    #     t['next_obs'] = self[idxes + 1]['obs']
    #     for k in self.all_keys:
    #         t[k] = self[idxes][k]
    #
    #     # not contiguous
    #     return t
    #
    # def sample_as_tensor(self, size=None, device='cuda'):
    #     idxes = self._sample_idxes(size)
    #
    #     t = {}
    #     t['next_obs'] = torch.from_numpy(
    #         np.array(self[idxes + 1]['obs'])
    #     ).to(device)
    #     for k in self.all_keys:
    #         t[k] = torch.from_numpy(np.array(self[idxes][k])).to(device)
    #
    #     return t
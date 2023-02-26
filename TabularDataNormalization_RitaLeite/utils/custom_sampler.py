# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 17:02:30 2022

@author: ritas
"""

import math
import torch
from torch import Tensor
import random
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union


T_co = TypeVar('T_co', covariant=True)


class Sampler(Generic[T_co]):
    r"""Base class for all Samplers.
    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices of dataset elements, and a :meth:`__len__` method
    that returns the length of the returned iterators.
    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`~torch.utils.data.DataLoader`, but is expected in any
              calculation involving the length of a :class:`~torch.utils.data.DataLoader`.
    """

    def __init__(self, data_source: Optional[Sized]) -> None:
        pass

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError


class PersonalizedWeightedRandomSampler(Sampler[int]):
    r"""Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).
    Args:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
        generator (Generator): Generator used in sampling.
    Example:
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> list(WeightedRandomSampler([0.1, 0.9, 0.4, 0.7, 3.0, 0.6], 5, replacement=True))
        [4, 4, 1, 4, 5]
        >>> list(WeightedRandomSampler([0.9, 0.4, 0.05, 0.2, 0.3, 0.1], 5, replacement=False))
        [0, 1, 4, 3, 2]
    """
    weights: Tensor
    num_samples: int
    replacement: bool
    
    def __init__(self,labels:Sequence[float], weights: Sequence[float], batch_size: int,num_samples: int,
                 replacement: bool = True, generator=None) -> None:
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))

        weights_tensor = torch.as_tensor(weights, dtype=torch.double)
        if len(weights_tensor.shape) != 1:
            raise ValueError("weights should be a 1d sequence but given "
                             "weights have shape {}".format(tuple(weights_tensor.shape)))
        self.weights = weights_tensor
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator
        self.labels=labels
        self.batch_size=batch_size
    def __iter__(self) -> Iterator[int]:
        indexes=[]
        for i in range(0,math.ceil(self.num_samples/self.batch_size)):
            all_classes=False
            while all_classes==False:
                rand_tensor = torch.multinomial(self.weights, self.batch_size, self.replacement)
                if all(self.labels[rand_tensor.tolist()]==0) or all(self.labels[rand_tensor.tolist()]==1):
                    all_classes=False
                else:
                    
                    indexes=indexes+rand_tensor.tolist()
                    all_classes=True
        return iter(indexes)
    def __len__(self) -> int:
        return self.num_samples

"""
@Time    : 2021/10/14 11:51
@File    : transforms.py
@Software: PyCharm
@Desc    : 
"""
import abc
from typing import List, Union, Dict, Any

import numpy as np

from .functional import (
    random_jitter,
    random_flipping,
    random_negating,
    random_scaling,
    random_cropping,
    magnitude_warping,
    time_warping,
    channel_shuffling
)


class Transformation(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self, x: np.ndarray):
        """

        Parameters
        ----------
        x : (batch_size, *, channel, time_length)
        """
        pass


class Compose(Transformation):
    def __init__(self, trans_list: List[Transformation]):
        super(Transformation, self).__init__()

        self.trans_list = trans_list

    def apply(self, x: np.ndarray):
        pass

    def __call__(self, x: np.ndarray):
        out = x
        for transformation in self.trans_list:
            out = transformation(out)

        return out


class TwoCropsTransform(Transformation):
    def __init__(self, transform: Transformation):
        super(TwoCropsTransform, self).__init__()

        self.transform = transform

    def __call__(self, x: np.ndarray):
        return self.transform(x), self.transform(x)


class RandomJitter(Transformation):
    def __init__(self, loc: float = 0.0, sigma: float = 1.0):
        super(RandomJitter, self).__init__()

        self.loc = loc
        self.sigma = sigma

    def __call__(self, x: np.ndarray):
        return random_jitter(x, self.loc, self.sigma)


class RandomFlipping(Transformation):
    def __init__(self):
        super(RandomFlipping, self).__init__()

    def __call__(self, x: np.ndarray):
        return random_flipping(x)


class RandomNegating(Transformation):
    def __init__(self):
        super(RandomNegating, self).__init__()

    def __call__(self, x: np.ndarray):
        return random_negating(x)


class RandomScaling(Transformation):
    def __init__(self, loc: float = 2.0, sigma: float = 1.1):
        super(RandomScaling, self).__init__()

        self.loc = loc
        self.sigma = sigma

    def __call__(self, x: np.ndarray):
        return random_scaling(x, self.loc, self.sigma)


class RandomCropping(Transformation):
    def __init__(self, size: int):
        super(RandomCropping, self).__init__()

        self.size = size

    def __call__(self, x: np.ndarray):
        return random_cropping(x, self.size)


class MagnitudeWarping(Transformation):
    def __init__(self, sigma: float = 1.0, knots: int = 4):
        super(MagnitudeWarping, self).__init__()

        self.sigma = sigma
        self.knots = knots

    def __call__(self, x: np.ndarray):
        return magnitude_warping(x, self.sigma, self.knots)


class TimeWarping(Transformation):
    def __init__(self, sigma: float = 1.0, knots: int = 4):
        super(TimeWarping, self).__init__()

        self.sigma = sigma
        self.knots = knots

    def __call__(self, x: np.ndarray):
        return time_warping(x, self.sigma, self.knots)


class ChannelShuffling(Transformation):
    def __init__(self):
        super(ChannelShuffling, self).__init__()

    def __call__(self, x: np.ndarray):
        return channel_shuffling(x)


class Perturbation(Transformation):
    def __init__(self, min_perturbation, max_perturbation):
        super(Perturbation, self).__init__()

        self.min_perturbation = min_perturbation
        self.max_perturbation = max_perturbation

    def apply(self, x: np.ndarray):
        pass

    def __call__(self, x: Union[np.ndarray, Dict[str, Any]]):
        if isinstance(x, np.ndarray):
            raise ValueError('Only support for Dict input!')
        else:
            assert self.max_perturbation <= x['head'].shape[-1]
            num_perturbation = np.random.randint(self.min_perturbation, self.max_perturbation + 1)
            sign = np.random.choice([-1, 1])
            if sign == -1:
                out = np.concatenate([x['head'][:, -num_perturbation:], x['mid'][:, :-num_perturbation]], axis=-1)
            else:
                out = np.concatenate([x['mid'][:, num_perturbation:], x['tail'][:, :num_perturbation]], axis=-1)

            x['mid'] = out
            return x


class Subsequence(Transformation):
    def __init__(self, seq_len: int):
        super(Subsequence, self).__init__()

    def __call__(self, x: np.ndarray):
        pass

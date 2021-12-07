"""
@Time    : 2021/10/14 11:50
@File    : functional.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np
import torch
from scipy.interpolate import interp1d


def __random_curve(length, sigma=0.2, knots=4):
    xx = np.arange(0, length, (length - 1) / (knots + 1))
    yy = np.random.normal(loc=1.0, scale=sigma, size=knots + 2)
    x_range = np.arange(length)
    # interpolator = CubicSpline(xx, yy)
    interpolator = interp1d(xx, yy, kind='cubic')

    return interpolator(x_range)


def random_jitter(x: np.ndarray, loc: float, sigma: float):
    return x + sigma * np.random.randn(*x.shape) + loc


def random_flipping(x: np.ndarray):
    return np.flip(x, axis=-1) if np.random.randint(low=0, high=2) == 1 else x


def random_negating(x: np.ndarray):
    return -x if np.random.randint(low=0, high=2) == 1 else x


def random_scaling(x: np.ndarray, loc: float, sigma: float):
    factors = np.random.normal(loc=loc, scale=sigma, size=(*x[:-2].shape, 1, x.shape[-1]))

    return x * factors


def random_cropping(x: np.ndarray, size: int):
    start_idx = np.random.randint(low=0, high=x.shape[-1] - size)

    return x[..., start_idx: start_idx + size]


# def scaling(x, sigma=1.1):
#     factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
#     ai = []
#     for i in range(x.shape[1]):
#         xi = x[:, i, :]
#         ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
#     return np.concatenate((ai), axis=1)


def magnitude_warping(x: np.ndarray, sigma: float, knots: int):
    return x * __random_curve(x.shape[-1], sigma, knots)


def time_warping(x: np.ndarray, sigma: float, knots: int):
    length = x.shape[-1]
    out = []
    for i in range(x.shape[-2]):
        timestamps = __random_curve(length, sigma, knots)
        timestamps_cumsum = np.cumsum(timestamps)
        scale = (length - 1) / timestamps_cumsum[-1]
        timestamps_new = timestamps_cumsum * scale
        x_range = np.arange(length)
        out.append(np.interp(x_range, timestamps_new, x[i]))

    return np.stack(out)


def channel_shuffling(x: np.ndarray):
    shuffled_idx = np.arange(x.shape[-2])
    np.random.shuffle(shuffled_idx)

    return x[..., shuffled_idx, :]


def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0, warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)


def subsequence(x: np.ndarray, seq_len: int):
    pass


def binary_emotion(y: np.ndarray, label_dim: int, threshold: float):
    pass

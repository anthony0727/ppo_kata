import numpy as np
import scipy


def generator(buffer, num_batch=1):
    chunks = np.split(buffer, num_batch)

    for chunk in chunks:
        yield chunk


def standardize(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def discounted_cumsum(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and
    # advantage estimates
    return scipy.signal.lfilter(
        [1],
        [1, float(-discount)],
        x[::-1],
        axis=0
    )[::-1]
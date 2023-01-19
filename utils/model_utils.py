import torch
import numpy as np


def sampler(x, batch):
    return [sample[x] for sample in batch]


def pad(batch):
    """
        Pads to the longest sample
    :param batch:
    :return:
    """
    words = sampler(0, batch)
    is_heads = sampler(2, batch)
    tags = sampler(3, batch)
    sequence_length = sampler(-1, batch)

    maximum_length = np.array(sequence_length).max()

    f = lambda x, seq_length: [sample[x] + [0] * (seq_length - len(sample[x])) for sample in batch]  # 0:
    x = f(1, maximum_length)
    y = f(-2, maximum_length)

    f = torch.LongTensor

    return words, f(x), is_heads, tags, f(y), sequence_length

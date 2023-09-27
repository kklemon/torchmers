import torch
import numpy as np


def batched_bincount(x, dim, max_value):
    target = torch.zeros(x.shape[0], max_value, dtype=x.dtype, device=x.device)
    values = torch.ones_like(x)
    target.scatter_add_(dim, x.type(torch.int64), values)
    return target


def padding_mask_from_seq_lens(max_len, seq_lens):
    assert seq_lens.ndim == 1, f'Expected seq_lens to have shape (batch_size,) but found {seq_lens.shape}'
    return torch.arange(max_len, device=seq_lens.device).unsqueeze(0) >= seq_lens[:, None]


def prepare_inputs(x, k, base=4, seq_lens=None, padding_mask=None, device=None):
    assert seq_lens is None or padding_mask is None, \
        'Only one of seq_lens or padding_mask can be provided, not both.'
    
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    else:
        assert isinstance(x, torch.Tensor), \
            f'x must be a numpy array or a torch tensor, but found {type(x)}'

    if device is not None:
        x = x.to(device)
    
    x = x.type(torch.int64)

    if x.ndim == 1:
        was_batched = False
        x = x.unsqueeze(0)
    else:
        was_batched = True

    bz, sz = x.shape

    if seq_lens is not None:
        padding_mask = padding_mask_from_seq_lens(sz, seq_lens)
    
    # Set padded elements to the negative value of the largest k-mer + 1.
    # This allows identifying folds that overlap with paddedings.
    if padding_mask is not None:
        x[padding_mask] = -(1 + base ** k)
    
    max_value = x.max()
    assert max_value <= 3, \
        f'x must be a tensor of integers in the range [0, {base - 1}], ' \
        f'but found maximum value of {max_value}'
    
    return x, was_batched


def count_k_mers(x, *, k, base=4, seq_lens=None, padding_mask=None):
    x, was_batched = prepare_inputs(x, k, base, seq_lens, padding_mask)

    exponents = k - 1 - torch.arange(k, dtype=x.dtype, device=x.device)

    # Create windows of size k and multiply the i-th element of each fold
    # by base^i. The sum over each fold is a unique identifier for each k-mer.
    k_mers = x.unfold(-1, k, 1) * base ** exponents
    k_mers = k_mers.sum(-1).type(x.dtype)

    # Set negative values, i.e. ones that overlapped with padded tokens to the largest k-mer + 1
    # This value will simply be cut off after bincounting
    k_mers[k_mers < 0] = base ** k

    counts = batched_bincount(k_mers, -1, base ** k + 1)[:, :-1]

    if not was_batched:
        return counts.squeeze(0)

    return counts


def count_k_mers_python(x, *, k, base=4, seq_lens=None, padding_mask=None):
    x, was_batched = prepare_inputs(x, k, base, seq_lens, padding_mask)

    k_mers = base ** k

    counts = torch.zeros(x.shape[0], k_mers, dtype=torch.int64, device=x.device)

    for i, seq in enumerate(x):
        for j in range(len(seq) - k + 1):
            if seq[j + k - 1] < 0:
                continue

            k_mer = 0

            for l in range(k):
                k_mer += seq[j + l] * base ** (k - 1 - l)

            counts[i, k_mer] += 1
    
    if not was_batched:
        return counts.squeeze(0)
    
    return counts

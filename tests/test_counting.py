import pytest
import numpy as np
import torch
import re
from itertools import chain, product

from torchmers.tokenizers import Tokenizer
from torchmers.counting import count_k_mers_python, count_k_mers
from torchmers.constants import NUCLEOTIDES


test_ks = [1, 2, 3, 4]


def count_k_mers_naive(s, k):
    counts = np.zeros(4 ** k, dtype=np.int64)

    for i, k_mer in enumerate(product(NUCLEOTIDES, repeat=k)):
        counts[i] = len(re.findall(f'(?={"".join(k_mer)})', s))
    
    return counts


def generate_dna_sequences(k):
    tokenizer = Tokenizer.from_name('DNA')

    # Generate k-mer permutations of sequences from length 1 to 64
    sequence = ''.join(chain.from_iterable(
        product(NUCLEOTIDES, repeat=k)
    ))

    seq_lens = torch.tensor(list(range(k, 4 ** k + 1)))
    sequences = [sequence[:i] for i in seq_lens]
    tokens = torch.tensor(tokenizer.encode_batch(sequences))

    return sequences, seq_lens, tokens


@pytest.mark.parametrize('k', test_ks)
def test_count_k_mers_python(k):
    sequences, seq_lens, tokens = generate_dna_sequences(k)

    counts = count_k_mers_python(tokens, k=k, seq_lens=seq_lens).numpy()

    for count, seq in zip(counts, sequences):
        assert (count == count_k_mers_naive(seq, k)).all()


@pytest.mark.parametrize('k', test_ks)
def test_count_k_mers(k):
    sequences, seq_lens, tokens = generate_dna_sequences(k)

    counts = count_k_mers(tokens, k=k, seq_lens=seq_lens).numpy()

    for count, seq in zip(counts, sequences):
        assert (count == count_k_mers_naive(seq, k)).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU available")
@pytest.mark.parametrize('k', test_ks)
def test_count_k_mers_gpu(k):
    sequences, seq_lens, tokens = generate_dna_sequences(k)

    counts = count_k_mers(
        tokens.cuda(), k=k, seq_lens=seq_lens
    ).cpu().numpy()

    for count, seq in zip(counts, sequences):
        assert (count == count_k_mers_naive(seq, k)).all()
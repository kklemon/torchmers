import pytest
import torch

from torchmers.modules import KMerFrequencyEncoder
from torchmers.tokenizers import Tokenizer
from fixtures import random_dna_sequences


@pytest.mark.parametrize('k', [3, 5, 7])
@pytest.mark.parametrize('use_gpu', [False, True])
def test_k_mer_frequncy_encoder(random_dna_sequences, use_gpu, k):
    sequences, seq_lens = random_dna_sequences
    
    encoder = KMerFrequencyEncoder(k=k)

    tokenizer = Tokenizer.from_name('DNA')
    tokens = torch.tensor(tokenizer.encode_batch(sequences))
    seq_lens = torch.tensor(seq_lens)

    if use_gpu:
        tokens = tokens.cuda()
        seq_lens = seq_lens.cuda()

    counts = encoder(tokens, seq_lens=seq_lens)

    assert counts.shape == (len(sequences), 4 ** k)



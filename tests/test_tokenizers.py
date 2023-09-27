import random
import pytest

from fixtures import random_dna_sequences
from torchmers.tokenizers import Tokenizer


def test_encode_decode_single(random_dna_sequences):
    tokenizer = Tokenizer.from_name('DNA')

    for seq in random_dna_sequences[0]:
        tokens = tokenizer.encode(seq)
        assert len(tokens) == len(seq)

        decoded = tokenizer.decode(tokens)
        assert decoded == seq


def test_encode_decode_batch(random_dna_sequences):
    tokenizer = Tokenizer.from_name('DNA')

    tokens = tokenizer.encode_batch(random_dna_sequences[0])
    decoded = tokenizer.decode_batch(tokens)

    for seq, dec in zip(random_dna_sequences[0], decoded):
        assert dec == seq


def test_vocab_size_getter():
    tokenizer = Tokenizer.from_name('DNA')

    # DNA alphabet has 4 letters + 1 padding token
    assert tokenizer.vocab_size() == 5

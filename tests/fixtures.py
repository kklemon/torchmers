import random
import pytest

from torchmers.constants import NUCLEOTIDES


@pytest.fixture
def random_dna_sequences():
    lengths = [random.randint(1, 100) for _ in range(10)]
    sequences = [
        ''.join(random.choices(NUCLEOTIDES, k=length))
        for length in lengths
    ]
    return sequences, lengths

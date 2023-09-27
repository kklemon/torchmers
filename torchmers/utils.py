from itertools import product
from torchmers.constants import NUCLEOTIDES


def enumerate_k_mers(k: int):
    return list(map(''.join, product(NUCLEOTIDES, repeat=k)))


def k_mer_from_index(index: int, k: int):
    return ''.join(NUCLEOTIDES[(index // (4 ** i)) % 4] for i in range(k)[::-1])


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
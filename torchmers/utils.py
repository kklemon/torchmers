from itertools import product
import bioseq

from torchmers.constants import NUCLEOTIDES


def get_tokenizer(name: str):
    tokenizer_dict = bioseq.get_tokenizer_dict(
        False,  # include_bos
        False,  # include_eos
        True  # include_pad
    )
    return tokenizer_dict[name]


def enumerate_k_mers(k: int):
    return list(map(''.join, product(NUCLEOTIDES, repeat=k)))


def k_mer_from_index(index: int, k: int):
    return ''.join(NUCLEOTIDES[(index // (4 ** i)) % 4] for i in range(k)[::-1])


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
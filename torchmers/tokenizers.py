import bioseq
import numpy as np


class Tokenizer:
    def __init__(self, bioseq_tokenizer, n_threads: int = 8):
        self.tokenizer = bioseq_tokenizer
        self.n_threads = n_threads

    @classmethod
    def from_name(cls, name: str, **kwargs):
        tokenizer_dict = bioseq.get_tokenizer_dict(
            False,  # include_bos
            False,  # include_eos
            True  # include_pad
        )
        return cls(tokenizer_dict[name], **kwargs)
    
    def vocab_size(self):
        return self.tokenizer.alphabet_size()
    
    def encode(self, sequence: str):
        return self.tokenizer.batch_tokenize(
            [sequence],
            padlen=len(sequence),
            batch_first=True
        )[0]
    
    def encode_batch(self, sequences: list[str]):
        pad_len = max(len(s) for s in sequences)
        return self.tokenizer.batch_tokenize(
            sequences,
            padlen=pad_len,
            batch_first=True,
            nthreads=self.n_threads
        )
    
    def decode(self, x: np.ndarray):
        assert x.ndim == 1
        return self.tokenizer.decode_tokens(x[None, :])[0]
    
    def decode_batch(self, x: np.ndarray):
        assert x.ndim == 2

        seq_lengths = np.count_nonzero(x != self.vocab_size() - 1, axis=-1)

        decoded_padded = self.tokenizer.decode_tokens(x)
        decoded = [s[:n] for s, n in zip(decoded_padded, seq_lengths)]

        return decoded
import argparse
from dataclasses import dataclass
import time
import numpy as np
import pandas as pd

from tqdm import tqdm
from itertools import islice, product
from torch.utils.data import IterableDataset, DataLoader
from torchmers.constants import NUCLEOTIDES
from torchmers.counting import count_k_mers, count_k_mers_python
from torchmers.tokenizers import Tokenizer


class RandomDNADataset(IterableDataset):
    def __init__(self, min_seq_len, max_seq_len, batch_size):
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.tokenizer = Tokenizer.from_name('DNA')
    
    def __iter__(self):
        while True:
            tokens = np.random.randint(0, 4, (self.batch_size, self.max_seq_len))
            seq_lens = np.random.randint(self.min_seq_len, self.max_seq_len + 1, self.batch_size)

            tokens[np.arange(self.max_seq_len)[None, :] > seq_lens[:, None]] = 4

            yield tokens, seq_lens


@dataclass
class Implementation:
    name: str
    func: callable
    gpu: bool


implementations = [
    Implementation('python naive', count_k_mers_python, False),
    Implementation('torchmers cpu', count_k_mers, False),
    Implementation('torchmers gpu', count_k_mers, True),
]

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num_data_workers', type=int, default=24)
parser.add_argument('--max_batches', type=int, default=50)
parser.add_argument('--max_time', type=int, default=5 * 60)
parser.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 256, 1024])
parser.add_argument('--k_mer_lengths', type=int, nargs='+', default=[3, 5, 7, 9])
parser.add_argument('--implementations', nargs='+')

args = parser.parse_args()

if args.implementations is not None:
    implementations = [i for i in implementations if i.name in args.implementations]

measurements = []

for impl, batch_size in product(implementations, args.batch_sizes):
    dataset = RandomDNADataset(1000, 10_000, batch_size=batch_size)
    batches = DataLoader(dataset, batch_size=None, num_workers=args.num_data_workers)

    for k in args.k_mer_lengths:
        start = time.time()

        for tokens, seq_lens in tqdm(
            iterable=islice(batches, args.max_batches),
            desc=f'impl={impl.name}, batch_size={batch_size}, k={k}'
        ):
            if impl.gpu:
                tokens = tokens.cuda()
                seq_lens = seq_lens.cuda()

            nucleotides = seq_lens.sum()

            measure_start = time.time()

            counts = impl.func(tokens, k=k, seq_lens=seq_lens)

            measure_time = time.time() - measure_start

            measurements.append({
                'Implementation': impl.name,
                'Nucleotides per second': nucleotides.item() / measure_time,
                'k-mer length': k,
                'Batch size': batch_size,
            })

            if (time.time() - start) > args.max_time:
                break


df = pd.DataFrame(measurements)
df.to_csv('benchmark.csv', index=False)

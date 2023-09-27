from typing import Literal
import torch
import torch.nn as nn

from torchmers.counting import count_k_mers


class KMerFrequencyEncoder(nn.Module):
    def __init__(self, k: int, base: int = 4, binary: int = False, log_counts: bool = False):
        super().__init__()

        assert k > 0, f'k must be positive, but found {k}'
        assert not (binary and log_counts), \
            'binary and log_counts options are mutually exclusive.'

        self.k = k
        self.base = base
        self.binary = binary
        self.log_counts = log_counts

    def forward(self, input, mask=None, seq_lens=None):
        counts = count_k_mers(
            input,
            k=self.k,
            padding_mask=mask,
            seq_lens=seq_lens
        ).float()

        if self.log_counts:
            counts = torch.log(counts + 1)

        if self.binary:
            counts = (counts > 0).float()
        
        return counts


NormLayers = Literal['none', 'batch_norm', 'layer_norm']


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        bias: bool = True,
        norm_layer: NormLayers | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.bias = bias

        layers = []

        def make_norm_layer(num_out_features):
            if norm_layer == 'batch_norm':
                return nn.BatchNorm1d(num_out_features)
            elif norm_layer == 'layer_norm':
                return nn.LayerNorm(num_out_features)

        for i in range(num_layers):
            is_last = i + 1 == num_layers

            in_feat = hidden_dim if i else input_dim
            out_feat = output_dim if is_last else hidden_dim

            layers.append(nn.Linear(in_feat, out_feat, bias=bias))

            if not is_last:
                layers.append(nn.ReLU())

                if norm_layer not in ('none', None):
                    layers.append(make_norm_layer(out_feat))
                
                if dropout:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
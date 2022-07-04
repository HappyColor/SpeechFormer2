
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from module.speechformer_v2_layer import SpeechFormer_v2_Encoder
from module.utils import create_PositionalEncoding, statistical_information
from module.utils import _no_grad_trunc_normal_
from model.speechformer import make_layers
import math

class MergeBlock(nn.Module):
    ''' Merge features between tow phases.

        The number of tokens is decreased while the dimension of token is increased.
    '''
    def __init__(self, in_channels, merge_scale, num_wtok, expand=2):
        super().__init__()

        out_channels = in_channels * expand
        self.MS = merge_scale
        self.num_wtok = num_wtok
        self.pool = nn.AdaptiveAvgPool2d((1, in_channels))
        self.fc = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        
    def forward(self, x:torch.Tensor):
        x_wtok, x_fea = x[:, :self.num_wtok], x[:, self.num_wtok:]

        B, T, C = x_fea.shape
        ms = T if self.MS == -1 else self.MS

        need_pad = T % ms
        if need_pad:
            pad = ms - need_pad
            x_fea = F.pad(x_fea, (0, 0, 0, pad), mode='constant', value=0)
            T += pad

        x_fea = x_fea.view(B, T//ms, ms, C)
        x_fea = self.pool(x_fea).squeeze(dim=-2)

        x = torch.cat((x_wtok, x_fea), dim=1)
        x = self.norm(self.fc(x))

        return x

class SpeechFormer_v2_Blocks(nn.Module):
    def __init__(self, num_layers, embed_dim, ffn_embed_dim=2304, local_size=0, 
            num_heads=8, dropout=0.1, attention_dropout=0.1, activation='relu', 
            use_position=False, num_wtok=0):
        super().__init__()
        self.position = create_PositionalEncoding(embed_dim) if use_position else None
        self.input_norm = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList(
            [SpeechFormer_v2_Encoder(embed_dim, ffn_embed_dim, local_size, num_heads, dropout, 
                attention_dropout, activation, num_wtok=num_wtok) for _ in range(num_layers)])

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, kmeans_mask=None):
        output = self.input_norm(x)

        for layer in self.layers:
            output = layer(output, self.position, kmeans_mask)

        return output

class SpeechFormer_v2(nn.Module):
    def __init__(self, input_dim, ffn_embed_dim, num_layers, num_heads, hop, num_classes, 
            expand, dropout=0.1, attention_dropout=0.1, **kwargs):
        super().__init__()
        
        self.input_dim = input_dim // num_heads * num_heads
        Locals, Merge = statistical_information(hop)
        assert isinstance(num_layers, list)
        
        self.num_wtok = math.ceil(kwargs['length'] / Merge[-2])

        self.wtok = nn.Parameter(torch.empty(1, self.num_wtok, input_dim), requires_grad=True)
        _no_grad_trunc_normal_(self.wtok, std=0.02)

        Former_args = {'num_layers': None, 'embed_dim': self.input_dim, 'ffn_embed_dim': ffn_embed_dim, 'local_size': None, 
            'num_heads': num_heads, 'dropout': dropout, 'attention_dropout': attention_dropout, 'activation': 'relu', 'use_position': True, 'num_wtok': self.num_wtok}
        Merge_args = {'in_channels': self.input_dim, 'merge_scale': None, 'expand': None, 'num_wtok': self.num_wtok}

        self.layers = make_layers(Locals, Merge, expand, num_layers, SpeechFormer_v2_Blocks, MergeBlock, Former_args, Merge_args)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        dim_expand = abs(reduce(lambda x, y: x * y, expand))
        classifier_dim = self.input_dim * dim_expand
        self.classifier = nn.Sequential(
            nn.Linear(classifier_dim, classifier_dim//2),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(classifier_dim//2, classifier_dim//4),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(classifier_dim//4, num_classes),
        )

    def forward(self, x):
        if self.input_dim != x.shape[-1]:
            x = x[:, :, :self.input_dim]

        wtok = self.wtok.expand(x.shape[0], -1, -1)
        x = torch.cat((wtok, x), dim=1)

        x = self.layers(x)
        x = self.avgpool(x.transpose(-1, -2)).squeeze(dim=-1)

        pred = self.classifier(x)
        
        return pred

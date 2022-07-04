import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from module.speechformer_layer import SpeechFormerEncoder
from module.utils import create_PositionalEncoding, statistical_information

class MergeBlock(nn.Module):
    ''' Merge features between tow phases.

        The number of tokens is decreased while the dimension of token is increased.
    '''
    def __init__(self, in_channels, merge_scale, expand=2, **kw_args):
        super().__init__()

        out_channels = in_channels * expand
        self.MS = merge_scale
        self.pool = nn.AdaptiveAvgPool2d((1, in_channels))
        self.fc = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        
    def forward(self, x:torch.Tensor):
        B, T, C = x.shape
        ms = T if self.MS == -1 else self.MS

        need_pad = T % ms
        if need_pad:
            pad = ms - need_pad
            x = F.pad(x, (0, 0, 0, pad), mode='constant', value=0)
            T += pad

        x = x.view(B, T//ms, ms, C)
        x = self.pool(x).squeeze(dim=-2)
        x = self.norm(self.fc(x))

        return x

def make_layers(Locals: list, Merge: list, expand: list, num_layers: list, Former_blocks, Merge_blocks, Former_args: dict, Merge_args: dict):
    layers = []
    last_merge = 1
    while len(expand) < len(Merge):
        expand = expand + [-1]

    for l, ms, exp, num in zip(Locals, Merge, expand, num_layers):
        _l = l // last_merge if l != -1 else -1
        _ms = ms // last_merge if ms != -1 else -1

        Former_args['num_layers'] = num
        Former_args['local_size'] = _l
        module1 = Former_blocks(**Former_args)
        layers += [module1]

        if Merge_blocks is not None:
            if _ms != -1:
                Merge_args['merge_scale'] = _ms
                Merge_args['expand'] = exp
                module2 = Merge_blocks(**Merge_args)
                layers += [module2]

                Merge_args['in_channels'] *= exp
                Former_args['embed_dim'] *= exp
                Former_args['ffn_embed_dim'] *= exp
        
            last_merge = ms
        
        if Former_args['use_position']:
            Former_args['use_position'] = False   # only the first layer use positional embedding.
            
    return nn.Sequential(*layers)

class SpeechFormerBlocks(nn.Module):
    def __init__(self, num_layers, embed_dim, ffn_embed_dim=2304, local_size=0, num_heads=8, dropout=0.1, attention_dropout=0.1, activation='relu', use_position=False):
        super().__init__()
        self.position = create_PositionalEncoding(embed_dim) if use_position else None
        self.input_norm = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList([SpeechFormerEncoder(embed_dim, ffn_embed_dim, local_size, num_heads, dropout, attention_dropout, activation, overlap=True) for _ in range(num_layers)])

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        output = self.input_norm(x)

        for layer in self.layers:
            output = layer(output, self.position)

        return output

class SpeechFormer(nn.Module):
    def __init__(self, input_dim, ffn_embed_dim, num_layers, num_heads, hop, num_classes, expand, dropout=0.1, attention_dropout=0.1, device='cuda', **kwargs):
        super().__init__()
        
        self.input_dim = input_dim//num_heads * num_heads
        Locals, Merge = statistical_information(hop)
        assert isinstance(num_layers, list)

        # None -> modify in the func: make_layers
        Former_args = {'num_layers': None, 'embed_dim': self.input_dim, 'ffn_embed_dim': ffn_embed_dim, 'local_size': None, 
            'num_heads': num_heads, 'dropout': dropout, 'attention_dropout': attention_dropout, 'activation': 'relu', 'use_position': True}
        Merge_args = {'in_channels': self.input_dim, 'merge_scale': None, 'expand': None}

        self.layers = make_layers(Locals, Merge, expand, num_layers, SpeechFormerBlocks, MergeBlock, Former_args, Merge_args)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        dim_expand = reduce(lambda x, y: x * y, expand)
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

        x = self.layers(x).squeeze(dim=1)
        x = self.avgpool(x.transpose(-1, -2)).squeeze(dim=-1)
        pred = self.classifier(x)
        
        return pred



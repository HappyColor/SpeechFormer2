
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from module.utils import _get_activation_fn, add_position, get_overlap_segments

class Speech_MSA_v2(nn.Module):
    ''' The Multi-Head Self-Attention in SpeechFormer++.
    '''
    def __init__(self, embed_dim, num_heads, local_size, dropout=0., bias=True, num_wtok=0):
        '''
        Args: 
            num_wtok: the number of learnable word tokens.
        '''
        super(Speech_MSA_v2, self).__init__()
        self.qdim = embed_dim
        self.kdim = embed_dim
        self.vdim = embed_dim
        self.local_size = local_size

        assert num_wtok > 0
        self.num_wtok = num_wtok

        self.project_qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.project_out = nn.Linear(embed_dim, embed_dim, bias=bias)

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.scaling = float(self.head_dim) ** -0.5

    def forward(self, x):
        '''
        Args:
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. This is an binary mask. When the value is True,
                the corresponding value on the attention layer will be filled with -inf.
            attn_mask: mask that prevents attention to certain positions. This is an additive mask
                (i.e. the values will be added to the attention layer).
        Shape:
            Inputs:
            - x: :math:`(B, T, E)` where T is the target sequence length, B is the batch size, E is
              the embedding dimension.
            - key_padding_mask: :math:`(B, S)`, ByteTensor, where B is the batch size, S is the source sequence length.
              3-D key_padding_mask with math:`(B, T, S)` is supported now, where T is the target sequence length.
            - attn_mask: :math:`(T, S)` where T is the target sequence length, S is the source sequence length.
        '''
        b, t = x.shape[:2]
        global_attn = self.local_size == -1

        Q, K, V = self.project_qkv(x).chunk(3, dim=-1)
        Q = Q * self.scaling
        Q = Q.transpose(0, 1).reshape(t, b * self.num_heads, self.head_dim).transpose(0, 1)
        K = K.transpose(0, 1).reshape(t, b * self.num_heads, self.head_dim).transpose(0, 1)
        V = V.transpose(0, 1).reshape(t, b * self.num_heads, self.head_dim).transpose(0, 1)

        if not global_attn:
            Q_wtok, Q = Q[:, :self.num_wtok], Q[:, self.num_wtok:]
            K_wtok, K = K[:, :self.num_wtok], K[:, self.num_wtok:]
            V_wtok, V = V[:, :self.num_wtok], V[:, self.num_wtok:]
            
            t_fea = Q.shape[1]

            # Compute the learnable word tokens first, then the feature tokens.
            pad_fea = (t_fea % self.num_wtok)
            if pad_fea:
                pad_len = self.num_wtok - pad_fea
                K_pad = F.pad(K, (0, 0, 0, pad_len), value=0)
                V_pad = F.pad(V, (0, 0, 0, pad_len), value=0)
            else:
                pad_len = 0
                K_pad = K
                V_pad = V
            
            K_pad = K_pad.reshape(b * self.num_heads, self.num_wtok, -1, self.head_dim)
            V_pad = V_pad.reshape(b * self.num_heads, self.num_wtok, -1, self.head_dim)

            attn_weights_wtok = torch.matmul(Q_wtok.unsqueeze(dim=2), torch.cat((K_wtok.unsqueeze(dim=2), K_pad), dim=2).transpose(-1, -2))
            attn_weights_wtok = F.softmax(attn_weights_wtok, dim=-1)
            attn_weights_wtok = F.dropout(attn_weights_wtok, p=self.dropout, training=self.training)

            output_wtok = torch.matmul(attn_weights_wtok, torch.cat((V_wtok.unsqueeze(dim=2), V_pad), dim=2)).squeeze(dim=2)

            expand = math.ceil(t_fea / self.num_wtok)
            output_wtok_expa = output_wtok[:, :, None, :].expand(-1, -1, expand, -1).reshape(b * self.num_heads, -1, self.head_dim)[:, :t_fea]
            
            K = get_overlap_segments(K, window_size=self.local_size)
            V = get_overlap_segments(V, window_size=self.local_size)

            attn_weights_fea = torch.matmul(Q.unsqueeze(dim=2), torch.cat((output_wtok_expa.unsqueeze(dim=2), K), dim=-2).transpose(-1, -2))

            # Separate softmax operations
            weights_wtok, weights_fea = attn_weights_fea[:, :, :, :1], attn_weights_fea[:, :, :, 1:]
            weights_wtok = weights_wtok.reshape(b * self.num_heads, t_fea)
            if pad_len:
                weights_wtok = F.pad(weights_wtok, (0, pad_len), value=float('-inf'))
            weights_wtok = weights_wtok.reshape(b * self.num_heads, self.num_wtok, -1)
            weights_wtok = F.softmax(weights_wtok, dim=-1).reshape(b * self.num_heads, -1)[:, :t_fea, None, None]
            weights_fea = F.softmax(weights_fea, dim=-1)
            attn_weights_fea = torch.cat((weights_wtok, weights_fea), dim=-1)
            
            attn_weights_fea = F.dropout(attn_weights_fea, p=self.dropout, training=self.training)

            output_fea = torch.matmul(attn_weights_fea, torch.cat((output_wtok_expa.unsqueeze(dim=2), V), dim=-2)).squeeze(dim=2)

            out = torch.cat([output_wtok, output_fea], dim=1)
        else:
            attn_weights = torch.matmul(Q, K.transpose(-1, -2))
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

            out = torch.matmul(attn_weights, V)

        out = out.transpose(0, 1).reshape(t, b, self.embed_dim).transpose(0, 1)
        out = self.project_out(out)

        return out

class SpeechFormer_v2_Encoder(nn.Module):
    """Implementation for SpeechFormer++.

    Args:
        embed_dim: input and output feature dimension.
        ffn_embed_dim: internal feature dimension in FFN.
        local_size: the attetion scope used when computing Speech-MSA.
        num_heads: the number of heads used in attention.
        dropout: dropout probability used in FFN.
        attention_dropout: dropout probability for attention matrix.
        activation: activation function used in FFN.
        nstok: number of special tokens used.
    """
    def __init__(self, embed_dim=1024, ffn_embed_dim=2304, local_size=0, num_heads=8, 
            dropout=0.1, attention_dropout=0.1, activation='relu', overlap=True, num_wtok=0):
        super().__init__()
        self.dropout = dropout
        self.activation_fn = _get_activation_fn(activation)

        self.attention = Speech_MSA_v2(embed_dim, num_heads, local_size, attention_dropout, num_wtok=num_wtok)

        self.attention_layer_norm = nn.LayerNorm(embed_dim)
        
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, x_position=None, kmeans_mask=None):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x
        x = add_position(x, x_position)

        x = self.attention(x)  # kmeans_mask=kmeans_mask
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.attention_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return x

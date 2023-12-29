import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, sizes, cat_pe=False):
        super().__init__()

        self.cat_pe = cat_pe
        pe = self.gen_pe(d_model, sizes)
        self.register_buffer('pe', pe)

    @staticmethod
    def gen_pe(d_model, sizes):
        pe = torch.zeros(sizes, d_model)
        position = torch.arange(sizes, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, x):
        if self.cat_pe:
            return torch.cat((x, self.pe[:, :x.size(1)].expand(x.size(0), -1, -1)), dim=2)
        else:
            return x + self.pe[:, :x.size(1)]


class Tfmer(nn.Module):
    def __init__(self, d_model, n_head, d_fefo, dropout, n_lyr):
        super().__init__()
        self.tgt = nn.Parameter(torch.randn((1, 1, d_model)))
        self.lyrs = nn.ModuleList()
        for _ in range(n_lyr):
            self.lyrs.append(nn.ModuleDict({
                'cross': CrossAttention(d_model, n_head, dropout),
                'fefo': FefoAttention(d_model, d_fefo, dropout),
            }))

    def forward(self, src):
        tgt = self.tgt.expand(src.size(0), -1, -1)
        for lyr in self.lyrs:
            tgt = lyr['cross'](tgt, src)
            tgt = lyr['fefo'](tgt)
        return tgt[:, 0]
        # tgtsrc = torch.cat((tgt, src), dim=1)
        #     tgtsrc = lyr['self'](tgtsrc)
        #     tgtsrc = lyr['fefo'](tgtsrc)
        # return tgtsrc[:, 0]


class SelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout, layer_norm_eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=0, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        identity = tgt
        tgt = self.norm(tgt)
        tgt = self.attn(tgt, tgt, tgt,
                        attn_mask=tgt_mask,
                        key_padding_mask=tgt_key_padding_mask,
                        need_weights=False)[0]
        tgt = self.drop(tgt)
        return tgt + identity


class CrossAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout, layer_norm_eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=0, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, tgt, mem, mem_mask=None, mem_key_padding_mask=None):
        # identity = tgt
        tgt_ = self.norm(tgt)
        tgt_ = self.attn(tgt_, mem, mem,
                         attn_mask=mem_mask,
                         key_padding_mask=mem_key_padding_mask,
                         need_weights=False)[0]
        tgt_ = self.drop(tgt_)
        return tgt + tgt_


class FefoAttention(nn.Module):
    def __init__(self, d_model, d_fefo, dropout, layer_norm_eps=1e-5):
        super().__init__()
        self.fefo = nn.Sequential(
            nn.LayerNorm(d_model, eps=layer_norm_eps),
            # nn.Linear(d_model, d_fefo * 2),
            # GEGLU(),
            nn.Linear(d_model, d_fefo),
            nn.Mish(),
            nn.Linear(d_fefo, d_model),
            nn.Dropout(dropout))

    def forward(self, x):
        return x + self.fefo(x)


class GEGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class TfmerFeaEx(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super().__init__(observation_space, features_dim)
        n_t, n_s, n_c, n_h, n_w = observation_space.shape
        d_input = n_c * 1
        self.cnn = nn.Sequential(
            nn.Conv2d(d_input * 1, d_input * 2, kernel_size=3, bias=False),  # 8x6 -> 6x4
            nn.BatchNorm2d(d_input * 2),
            nn.ReLU(),
            nn.Conv2d(d_input * 2, d_input * 4, kernel_size=3, bias=False),  # 6x4 -> 4x2
            nn.BatchNorm2d(d_input * 4),
            nn.ReLU(),
            nn.Conv2d(d_input * 4, d_input * 8, kernel_size=(3, 2), bias=False),  # 4x2 -> 2x1
            nn.BatchNorm2d(d_input * 8),
            nn.ReLU(),
            nn.Flatten(),  # 2x1 -> 2
            nn.Linear(d_input * 8 * 2, features_dim // n_s),
        )
        self.pos_enc = PositionalEncoding(d_model=features_dim, sizes=n_t, cat_pe=False)
        self.tfmer = Tfmer(d_model=features_dim, n_head=2, d_fefo=features_dim * 4,
                           dropout=0, n_lyr=8)

    def forward(self, x):
        b, t, s, c, h, w = x.shape
        x = rearrange(x, 'b t s c h w -> (b s t) c h w', b=b, t=t, s=s, c=c, h=h, w=w)
        x = self.cnn(x)
        x = rearrange(x, '(b s t) d -> b t (s d)', b=b, s=s, t=t)
        x = self.pos_enc(x)
        x = self.tfmer(x)
        return x


class CnnFeaEx(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super().__init__(observation_space, features_dim)
        n_t, n_s, n_c, n_h, n_w = observation_space.shape
        d_input = n_c * n_t
        self.model = nn.Sequential(
            nn.Conv2d(d_input * 1, d_input * 2, kernel_size=3, bias=False),  # 8x6 -> 6x4
            nn.BatchNorm2d(d_input * 2),
            nn.ReLU(),
            nn.Conv2d(d_input * 2, d_input * 3, kernel_size=3, bias=False),  # 6x4 -> 4x2
            nn.BatchNorm2d(d_input * 3),
            nn.ReLU(),
            nn.Conv2d(d_input * 3, d_input * 4, kernel_size=(3, 2), bias=False),  # 4x2 -> 2x1
            nn.BatchNorm2d(d_input * 4),
            nn.ReLU(),
            nn.Flatten(),  # 2x1 -> 2
            nn.Linear(d_input * 4 * 2, features_dim // n_s),
        )

    def forward(self, x):
        b, t, s, c, h, w = x.shape
        x = rearrange(x, 'b t s c h w -> (b s) (t c) h w', b=b, t=t, s=s, c=c, h=h, w=w)
        x = self.model(x)
        x = rearrange(x, '(b s) d -> b (s d)', b=b, s=s)
        return x

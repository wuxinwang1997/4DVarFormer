#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:38:05 2020

@author: rfablet
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_


class PreNorm(nn.Module):
    """
        PreNorm for Attention
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    """
        MLP FeedForward
    """
    def __init__(self, dim_in, hidden_dim, dim_out, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim_out),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    """
        Linear Attention
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class CorssAttention(nn.Module):
    """
        Linear Cross-Attention
    """
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()

        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qk = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, grad):
        qk = self.to_qk(x).chunk(2, dim=-1)
        q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qk)
        v = rearrange(self.to_v(grad), 'b n (h d) -> b h n d', h = self.heads)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Decoder(nn.Module):
    """
        encoder to extract features from background fields
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dim, dropout))
            ]))

    def forward(self, x):
        for attn, ffn in self.layers:
            x = attn(x) + x
            x = ffn(x) + x
        return x

# Gradient-based minimization using a LSTM using a (sub)gradient as inputs
class ViT(torch.nn.Module):
    def __init__(self,
                 image_size=[160, 160],
                 patch_size=[16, 16],
                 dim=768,
                 depth=12,
                 heads=12,
                 dim_head=64,
                 in_chans = 48,
                 out_chans = 24,
                 dropout = 0.,
                 mlp_ratio = 4,
                 emb_dropout = 0.):
        super(ViT, self).__init__()

        self.image_height, self.image_width = tuple(image_size)
        self.patch_height, self.patch_width = tuple(patch_size)
        self.out_chans = out_chans

        assert self.image_height % self.patch_height == 0 and self.image_width % self.patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (self.image_height // self.patch_height) * (self.image_width // self.patch_width)

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_chans, dim, kernel_size=self.patch_height, stride=self.patch_width, padding=0),
            Rearrange('b c h w -> b (h w) c')
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.decoder = Decoder(dim, depth, heads, dim_head, mlp_ratio * dim, dropout)

        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange(
                "b (h w) (p1 p2 c_out) -> b c_out (h p1) (w p2)",
                p1=self.patch_height,
                p2=self.patch_width,
                h=self.image_height // self.patch_height,
                w=self.image_width // self.patch_width,
            ),
            nn.Conv2d(dim // (self.patch_height * self.patch_width), self.out_chans, kernel_size=1, stride=1)
        )

        trunc_normal_(self.pos_embedding, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, xb, obs, mask):
        # compute increment
        x = torch.concat([xb, obs * mask], dim=1)

        x = self.to_patch_embedding(x)
        x = x + self.pos_embedding
        x = self.decoder(x)

        return self.head(x) + xb
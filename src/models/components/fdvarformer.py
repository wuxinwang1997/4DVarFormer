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
from visualizer import get_local

# Modules for the definition of the norms for
# the observation and prior model
class Model_WeightedL2Norm(torch.nn.Module):
    def __init__(self, num_vars, obserr):
        super(Model_WeightedL2Norm, self).__init__()
        obserr_ = np.ones((1, num_vars))
        for i in range(num_vars):
            obserr_[:, i] = obserr[i] * obserr_[:, i]
        obserr = obserr_ ** 2
        R_inv = np.where(obserr == 0, 0, 1 / obserr)
        self.R_inv = torch.nn.Parameter(torch.Tensor(R_inv), requires_grad=False)

    def forward(self, x, std):
        var = (torch.unsqueeze(std, dim=0) ** 2).to(x.device, dtype=x.dtype)  # (24)
        loss = torch.nansum(x ** 2, dim=(-2, -1))  # (B, 4, 24)
        loss = torch.nanmean(loss, dim=1)  # (B, 24)
        loss = loss * self.R_inv * var  # (B, 24)
        loss_z = torch.nansum(loss[:, :4]) / x.shape[0] # (B)
        loss_r = torch.nansum(loss[:, 4:8]) / x.shape[0]  # (B)
        loss_t = torch.nansum(loss[:, 8:12]) / x.shape[0]  # (B)
        loss_uv = torch.nansum(loss[:, 12:20]) / x.shape[0]  # (B)
        loss_uv10 = torch.nansum(loss[:, 20:22]) / x.shape[0]  # (B)
        loss_t2m = torch.nansum(loss[:, 22:23]) / x.shape[0]  # (B)
        loss_msl = torch.nansum(loss[:, 23:24]) / x.shape[0]  # (B)
        loss = torch.nansum(loss) / x.shape[0]  # (B)
        return loss, [loss_z, loss_r, loss_t, loss_uv, loss_uv10, loss_t2m, loss_msl]

# Gradient-based minimization using a LSTM using a (sub)gradient as inputs
class Pre_CorssAttn_Norm(nn.Module):
    """
        PreNorm for Cross-Attention
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, y, **kwargs):
        return self.fn(self.norm1(x), self.norm2(y), **kwargs)

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

    @get_local('attn')
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

        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    @get_local('cross_attn')
    def forward(self, x, grad):
        q = rearrange(self.to_q(x), 'b n (h d) -> b h n d', h=self.heads)
        kv = self.to_kv(grad).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), kv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        cross_attn = self.attend(dots)
        cross_attn = self.dropout(cross_attn)

        out = torch.matmul(cross_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# Gradient-based minimization using a LSTM using a (sub)gradient as inputs
class FDVarFormer(torch.nn.Module):
    def __init__(self,
                 image_size=[160, 160],
                 patch_size=[8, 8],
                 dim=768,
                 depth=3,
                 heads=12,
                 dim_head=64,
                 in_chans = 24,
                 out_chans = 24,
                 dropout = 0.,
                 mlp_ratio = 4,
                 emb_dropout = 0.):
        super(FDVarFormer, self).__init__()

        image_height, image_width = tuple(image_size)
        patch_height, patch_width = tuple(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)

        self.to_patch_embedding_x = nn.Sequential(
            nn.Conv2d(in_chans, dim, kernel_size=patch_height, stride=patch_width, padding=0),
            Rearrange('b c h w -> b (h w) c'),
        )

        self.to_patch_embedding_grad = nn.Sequential(
            nn.Conv2d(in_chans, dim, kernel_size=patch_height, stride=patch_width, padding=0),
            Rearrange('b c h w -> b (h w) c'),
        )

        self.pos_embedding_x = nn.Parameter(torch.zeros(1, num_patches, dim))
        self.pos_embedding_grad = nn.Parameter(torch.zeros(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Pre_CorssAttn_Norm(dim, CorssAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_ratio * dim, dim, dropout)),
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_ratio * dim, dim, dropout)),
            ]))

        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_chans * patch_height * patch_width, bias=False),
            Rearrange(
                "b (h w) (p1 p2 c_out) -> b c_out (h p1) (w p2)",
                p1=patch_height,
                p2=patch_width,
                h=image_height // patch_height,
                w=image_width // patch_width,
            ),
        )

        trunc_normal_(self.pos_embedding_x, std=.02)
        trunc_normal_(self.pos_embedding_grad, std=.02)
        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'to_patch_embedding_x', 'to_patch_embedding_grad'}

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

    def forward(self, xb, grad):
        # compute increment
        xb = self.to_patch_embedding_x(xb)
        xb = xb + self.pos_embedding_x
        grad = self.to_patch_embedding_grad(grad)
        grad = grad + self.pos_embedding_grad

        for attn1, ffn1, attn2, ffn2 in self.layers:
            grad = attn1(xb, grad) + grad
            grad = ffn1(grad) + grad
            grad = attn2(grad) + grad
            grad = ffn2(grad) + grad

        out = self.head(grad)

        return out

# New module for the definition/computation of the variational cost
class Model_Var_Cost(nn.Module):
    def __init__(self ,m_NormObs):
        super(Model_Var_Cost, self).__init__()
        # parameters for variational cost
        self.normObs   = m_NormObs

    def forward(self, dy, std):
        loss =  self.normObs(dy, std)

        return loss

class Model_H(torch.nn.Module):
    def __init__(self, shape_data):
        super(Model_H, self).__init__()
        self.dim_obs = 1
        self.dim_obs_channel = np.array(shape_data)

    def forward(self, x, y, mask):
        dyout = (x - y) * mask

        return dyout

# 4DVarNN Solver class using automatic differentiation for the computation of gradient of the variational cost
# input modules: operator phi_r, gradient-based update model m_Grad
# modules for the definition of the norm of the observation and prior terms given as input parameters
# (default norm (None) refers to the L2 norm)
# updated inner modles to account for the variational model module
class Solver(nn.Module):
    def __init__(self ,phi_r,mod_H, m_Grad, num_vars, obserr, shape_data, n_iter):
        super(Solver, self).__init__()
        self.phi_r = phi_r
        m_NormObs =  Model_WeightedL2Norm(num_vars, obserr)
        self.shape_data = shape_data
        self.model_H = mod_H
        self.model_Grad = m_Grad
        self.model_VarCost = Model_Var_Cost(m_NormObs)
        self.preds = []
        with torch.no_grad():
            self.n_step = int(n_iter)

    def forward(self, x, yobs, mask, mult):
        return self.solve(x, yobs, mask, mult)

    def solve(self, x_0, obs, mask, mult):
        x_k = torch.mul(x_0,1.)
        x_k_plus_1 = None
        for _ in range(self.n_step):
            self.preds = []
            x_k_plus_1 = self.solver_step(x_k, obs, mask, mult)

            x_k = torch.mul(x_k_plus_1,1.)

        return x_k_plus_1

    def solver_step(self, x_k, obs, mask, mult):
        _, var_cost_grad = self.var_cost(x_k, obs, mask, mult)
        normgrad = torch.sqrt(torch.mean(var_cost_grad**2, dim=(1,2,3), keepdim=True))
        normgrad = torch.where(normgrad == 0, 1, normgrad)
        delta_x = self.model_Grad(x_k, var_cost_grad / normgrad)
        x_k_plus_1 = x_k + delta_x
        return x_k_plus_1

    def var_cost(self, xb, yobs, mask, std):
        self.preds.append(xb)
        for i in np.arange(1, yobs.shape[1]):
            self.preds.append(self.phi_r(self.preds[i-1]))
        preds = torch.stack(self.preds, dim=1)
        dy = self.model_H(preds, yobs, mask)

        loss, losses = self.model_VarCost(dy, std)
        var_cost_grad = torch.zeros_like(xb).to(xb.device, dtype=xb.dtype)
        num_nonzero = 0
        for l in losses:
            if torch.sum(l).item() != 0:
                num_nonzero += 1
                grad = torch.autograd.grad(l, xb, grad_outputs=torch.ones_like(l), retain_graph=True)[0]
                gradnorm = torch.sqrt(torch.mean(grad ** 2 + 0., dim=(1, 2, 3), keepdim=True))
                var_cost_grad += grad / gradnorm
        if num_nonzero > 0:
            var_cost_grad /= num_nonzero
        var_cost_grad = torch.where(torch.isnan(var_cost_grad), 0, var_cost_grad)
        var_cost_grad = torch.where(torch.isinf(var_cost_grad), 0, var_cost_grad)
        return loss, var_cost_grad
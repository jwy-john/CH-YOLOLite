import numpy as np
import torch
from torch import nn
from torch.nn import init

#   CBAM注意力
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out
    
    
    

# Biformer
"""
Core of BiFormer, Bi-Level Routing Attention.
To be refactored.
author: ZHU Lei
github: https://github.com/rayleizhu
email: ray.leizhu@outlook.com
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


class TopkRouting(nn.Module):
    """
    differentiable topk routing with scaling
    Args:
        qk_dim: int, feature dimension of query and key
        topk: int, the 'topk'
        qk_scale: int or None, temperature (multiply) of softmax activation
        with_param: bool, wether inorporate learnable params in routing unit
        diff_routing: bool, wether make routing differentiable
        soft_routing: bool, wether make output value multiplied by routing weights
    """
    def __init__(self, qk_dim, topk=4, qk_scale=None, param_routing=False, diff_routing=False):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5
        self.diff_routing = diff_routing
        # TODO: norm layer before/after linear?
        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()
        # routing activation
        self.routing_act = nn.Softmax(dim=-1)
    
    def forward(self, query:Tensor, key:Tensor)->Tuple[Tensor]:
        """
        Args:
            q, k: (n, p^2, c) tensor
        Return:
            r_weight, topk_index: (n, p^2, topk) tensor
        """
        if not self.diff_routing:
            query, key = query.detach(), key.detach()
        query_hat, key_hat = self.emb(query), self.emb(key) # per-window pooling -> (n, p^2, c) 
        attn_logit = (query_hat*self.scale) @ key_hat.transpose(-2, -1) # (n, p^2, p^2)
        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1) # (n, p^2, k), (n, p^2, k)
        r_weight = self.routing_act(topk_attn_logit) # (n, p^2, k)
        
        return r_weight, topk_index
        

class KVGather(nn.Module):
    def __init__(self, mul_weight='none'):
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight

    def forward(self, r_idx:Tensor, r_weight:Tensor, kv:Tensor):
        """
        r_idx: (n, p^2, topk) tensor
        r_weight: (n, p^2, topk) tensor
        kv: (n, p^2, w^2, c_kq+c_v)
        Return:
            (n, p^2, topk, w^2, c_kq+c_v) tensor
        """
        # select kv according to routing index
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)
        # print(r_idx.size(), r_weight.size())
        # FIXME: gather consumes much memory (topk times redundancy), write cuda kernel? 
        topk_kv = torch.gather(kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1), # (n, p^2, p^2, w^2, c_kv) without mem cpy
                                dim=2,
                                index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv) # (n, p^2, k, w^2, c_kv)
                               )

        if self.mul_weight == 'soft':
            topk_kv = r_weight.view(n, p2, topk, 1, 1) * topk_kv # (n, p^2, k, w^2, c_kv)
        elif self.mul_weight == 'hard':
            raise NotImplementedError('differentiable hard routing TBA')
        # else: #'none'
        #     topk_kv = topk_kv # do nothing

        return topk_kv

class QKVLinear(nn.Module):
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias)
    
    def forward(self, x):
        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim+self.dim], dim=-1)
        return q, kv
        # q, k, v = self.qkv(x).split([self.qk_dim, self.qk_dim, self.dim], dim=-1)
        # return q, k, v

class BiLevelRoutingAttention(nn.Module):
    """
    n_win: number of windows in one side (so the actual number of windows is n_win*n_win)
    kv_per_win: for kv_downsample_mode='ada_xxxpool' only, number of key/values per window. Similar to n_win, the actual number is kv_per_win*kv_per_win.
    topk: topk for window filtering
    param_attention: 'qkvo'-linear for q,k,v and o, 'none': param free attention
    param_routing: extra linear for routing
    diff_routing: wether to set routing differentiable
    soft_routing: wether to multiply soft routing weights 
    """
    def __init__(self, dim, n_win=7, num_heads=8, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='identity',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False, side_dwconv=3,
                 auto_pad=True):
        super().__init__()
        # local attention setting
        self.dim = dim
        self.n_win = n_win  # Wh, Ww
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        assert self.qk_dim % num_heads == 0 and self.dim % num_heads==0, 'qk_dim and dim must be divisible by num_heads!'
        self.scale = qk_scale or self.qk_dim ** -0.5


        ################side_dwconv (i.e. LCE in ShuntedTransformer)###########
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv//2, groups=dim) if side_dwconv > 0 else \
                    lambda x: torch.zeros_like(x)
        
        ################ global routing setting #################
        self.topk = topk
        self.param_routing = param_routing
        self.diff_routing = diff_routing
        self.soft_routing = soft_routing
        # router
        assert not (self.param_routing and not self.diff_routing) # cannot be with_param=True and diff_routing=False
        self.router = TopkRouting(qk_dim=self.qk_dim,
                                  qk_scale=self.scale,
                                  topk=self.topk,
                                  diff_routing=self.diff_routing,
                                  param_routing=self.param_routing)
        if self.soft_routing: # soft routing, always diffrentiable (if no detach)
            mul_weight = 'soft'
        elif self.diff_routing: # hard differentiable routing
            mul_weight = 'hard'
        else:  # hard non-differentiable routing
            mul_weight = 'none'
        self.kv_gather = KVGather(mul_weight=mul_weight)

        # qkv mapping (shared by both global routing and local attention)
        self.param_attention = param_attention
        if self.param_attention == 'qkvo':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Linear(dim, dim)
        elif self.param_attention == 'qkv':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Identity()
        else:
            raise ValueError(f'param_attention mode {self.param_attention} is not surpported!')
        
        self.kv_downsample_mode = kv_downsample_mode
        self.kv_per_win = kv_per_win
        self.kv_downsample_ratio = kv_downsample_ratio
        self.kv_downsample_kenel = kv_downsample_kernel
        if self.kv_downsample_mode == 'ada_avgpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'ada_maxpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveMaxPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'maxpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.MaxPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'avgpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.AvgPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'identity': # no kv downsampling
            self.kv_down = nn.Identity()
        elif self.kv_downsample_mode == 'fracpool':
            # assert self.kv_downsample_ratio is not None
            # assert self.kv_downsample_kenel is not None
            # TODO: fracpool
            # 1. kernel size should be input size dependent
            # 2. there is a random factor, need to avoid independent sampling for k and v 
            raise NotImplementedError('fracpool policy is not implemented yet!')
        elif kv_downsample_mode == 'conv':
            # TODO: need to consider the case where k != v so that need two downsample modules
            raise NotImplementedError('conv policy is not implemented yet!')
        else:
            raise ValueError(f'kv_down_sample_mode {self.kv_downsaple_mode} is not surpported!')

        # softmax for local attention
        self.attn_act = nn.Softmax(dim=-1)

        self.auto_pad=auto_pad

    def forward(self, x, ret_attn_mask=False):
        """
        x: NHWC tensor
        Return:
            NHWC tensor
        """
        x = rearrange(x, "n c h w -> n h w c")
         # NOTE: use padding for semantic segmentation
        ###################################################
        if self.auto_pad:
            N, H_in, W_in, C = x.size()

            pad_l = pad_t = 0
            pad_r = (self.n_win - W_in % self.n_win) % self.n_win
            pad_b = (self.n_win - H_in % self.n_win) % self.n_win
            x = F.pad(x, (0, 0, # dim=-1
                          pad_l, pad_r, # dim=-2
                          pad_t, pad_b)) # dim=-3
            _, H, W, _ = x.size() # padded size
        else:
            N, H, W, C = x.size()
            assert H%self.n_win == 0 and W%self.n_win == 0 #
        ###################################################


        # patchify, (n, p^2, w, w, c), keep 2d window as we need 2d pooling to reduce kv size
        x = rearrange(x, "n (j h) (i w) c -> n (j i) h w c", j=self.n_win, i=self.n_win)

        #################qkv projection###################
        # q: (n, p^2, w, w, c_qk)
        # kv: (n, p^2, w, w, c_qk+c_v)
        # NOTE: separte kv if there were memory leak issue caused by gather
        q, kv = self.qkv(x) 

        # pixel-wise qkv
        # q_pix: (n, p^2, w^2, c_qk)
        # kv_pix: (n, p^2, h_kv*w_kv, c_qk+c_v)
        q_pix = rearrange(q, 'n p2 h w c -> n p2 (h w) c')
        kv_pix = self.kv_down(rearrange(kv, 'n p2 h w c -> (n p2) c h w'))
        kv_pix = rearrange(kv_pix, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win)

        q_win, k_win = q.mean([2, 3]), kv[..., 0:self.qk_dim].mean([2, 3]) # window-wise qk, (n, p^2, c_qk), (n, p^2, c_qk)

        ##################side_dwconv(lepe)##################
        # NOTE: call contiguous to avoid gradient warning when using ddp
        lepe = self.lepe(rearrange(kv[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', j=self.n_win, i=self.n_win).contiguous())
        lepe = rearrange(lepe, 'n c (j h) (i w) -> n (j h) (i w) c', j=self.n_win, i=self.n_win)

        ############ gather q dependent k/v #################

        r_weight, r_idx = self.router(q_win, k_win) # both are (n, p^2, topk) tensors

        kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix) #(n, p^2, topk, h_kv*w_kv, c_qk+c_v)
        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], dim=-1)
        # kv_pix_sel: (n, p^2, topk, h_kv*w_kv, c_qk)
        # v_pix_sel: (n, p^2, topk, h_kv*w_kv, c_v)
        
        ######### do attention as normal ####################
        k_pix_sel = rearrange(k_pix_sel, 'n p2 k w2 (m c) -> (n p2) m c (k w2)', m=self.num_heads) # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_kq//m) transpose here?
        v_pix_sel = rearrange(v_pix_sel, 'n p2 k w2 (m c) -> (n p2) m (k w2) c', m=self.num_heads) # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_v//m)
        q_pix = rearrange(q_pix, 'n p2 w2 (m c) -> (n p2) m w2 c', m=self.num_heads) # to BMLC tensor (n*p^2, m, w^2, c_qk//m)

        # param-free multihead attention
        attn_weight = (q_pix * self.scale) @ k_pix_sel # (n*p^2, m, w^2, c) @ (n*p^2, m, c, topk*h_kv*w_kv) -> (n*p^2, m, w^2, topk*h_kv*w_kv)
        attn_weight = self.attn_act(attn_weight)
        out = attn_weight @ v_pix_sel # (n*p^2, m, w^2, topk*h_kv*w_kv) @ (n*p^2, m, topk*h_kv*w_kv, c) -> (n*p^2, m, w^2, c)
        out = rearrange(out, '(n j i) m (h w) c -> n (j h) (i w) (m c)', j=self.n_win, i=self.n_win,
                        h=H//self.n_win, w=W//self.n_win)

        out = out + lepe
        # output linear
        out = self.wo(out)

        # NOTE: use padding for semantic segmentation
        # crop padded region
        if self.auto_pad and (pad_r > 0 or pad_b > 0):
            out = out[:, :H_in, :W_in, :].contiguous()

        if ret_attn_mask:
            return out, r_weight, r_idx, attn_weight
        else:
            return rearrange(out, "n h w c -> n c h w")

class Attention(nn.Module):
    """
    vanilla attention
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        args:
            x: NCHW tensor
        return:
            NCHW tensor
        """
        _, _, H, W = x.size()
        x = rearrange(x, 'n c h w -> n (h w) c')
        
        #######################################
        B, N, C = x.shape        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        #######################################

        x = rearrange(x, 'n (h w) c -> n c h w', h=H, w=W)
        return x

class AttentionLePE(nn.Module):
    """
    vanilla attention
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., side_dwconv=5):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv//2, groups=dim) if side_dwconv > 0 else \
                    lambda x: torch.zeros_like(x)

    def forward(self, x):
        """
        args:
            x: NCHW tensor
        return:
            NCHW tensor
        """
        _, _, H, W = x.size()
        x = rearrange(x, 'n c h w -> n (h w) c')
        
        #######################################
        B, N, C = x.shape        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        lepe = self.lepe(rearrange(x, 'n (h w) c -> n c h w', h=H, w=W))
        lepe = rearrange(lepe, 'n c h w -> n (h w) c')

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = x + lepe

        x = self.proj(x)
        x = self.proj_drop(x)
        #######################################

        x = rearrange(x, 'n (h w) c -> n c h w', h=H, w=W)
        return 
    
    

# DYhead 基于注意力的检测头
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmcv.cnn import build_activation_layer, build_norm_layer
# from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d
# from mmengine.model import constant_init, normal_init

# 这段代码的作用是将一个值调整为最接近的可被指定除数整除的整数
# 同时满足约束条件，以便在某些计算或优化过程中使用。
# v: 需要调整的数值。  divisor: 除数，用于指定新的数值必须是其整数倍。 min_value: 最小值，如果调整后的数值小于该值，则取该值
# def _make_divisible(v, divisor, min_value=None):
#     if min_value is None:
#         min_value = divisor
#     new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
#     # Make sure that round down does not go down by more than 10%.
#     if new_v < 0.9 * v:
#         new_v += divisor
#     return new_v


# class swish(nn.Module):
#     def forward(self, x):
#         return x * torch.sigmoid(x)


# class h_swish(nn.Module):
#     def __init__(self, inplace=False):
#         super(h_swish, self).__init__()
#         self.inplace = inplace

#     def forward(self, x):
#         return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


# class h_sigmoid(nn.Module):
#     def __init__(self, inplace=True, h_max=1):
#         super(h_sigmoid, self).__init__()
#         self.relu = nn.ReLU6(inplace=inplace)
#         self.h_max = h_max

#     def forward(self, x):
#         return self.relu(x + 3) * self.h_max / 6

'''
    inp是输入通道数（input channels）。
    reduction是用于决定调整参数的降维因子（reduction factor）。
    lambda_a是激活函数参数a的缩放因子。
    K2是一个布尔值，表示是否使用K2参数化方法。
    use_bias是一个布尔值，表示是否在激活函数中使用偏置。
    use_spatial是一个布尔值，表示是否使用空间注意力机制。
    init_a是初始化参数a的列表，长度为2。
    init_b是初始化参数b的列表，长度为2。
'''
# class DyReLU(nn.Module):

#     def __init__(self, inp, reduction=4, lambda_a=1.0, K2=True, use_bias=True, use_spatial=False,
#                  init_a=[1.0, 0.0], init_b=[0.0, 0.0]):
#         super(DyReLU, self).__init__()
#         self.oup = inp
#         self.lambda_a = lambda_a * 2
#         self.K2 = K2
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)

#         self.use_bias = use_bias
#         if K2:
#             self.exp = 4 if use_bias else 2    #   K2=True , use_bias=True  ---> exp = 4   use_bias=False  ---> exp = 2
#         else:
#             self.exp = 2 if use_bias else 1    #   K2=False , use_bias=True  ---> exp = 2   use_bias=False  ---> exp = 1
#         self.init_a = init_a
#         self.init_b = init_b

#         # determine squeeze    用于全连接层降维
#         if reduction == 4:      
#             squeeze = inp // reduction
#         else:
#             squeeze = _make_divisible(inp // reduction, 4)
#         # print('reduction: {}, squeeze: {}/{}'.format(reduction, inp, squeeze))
#         # print('init_a: {}, init_b: {}'.format(self.init_a, self.init_b))

#         self.fc = nn.Sequential(
#             nn.Linear(inp, squeeze),
#             nn.ReLU(inplace=True),
#             nn.Linear(squeeze, self.oup * self.exp),
#             h_sigmoid()
#         )
#         if use_spatial:
#             self.spa = nn.Sequential(
#                 nn.Conv2d(inp, 1, kernel_size=1),
#                 nn.BatchNorm2d(1),
#             )
#         else:
#             self.spa = None

#     def forward(self, x):
#         if isinstance(x, list):
#             x_in = x[0]
#             x_out = x[1]
#         else:
#             x_in = x
#             x_out = x
#         b, c, h, w = x_in.size()
#         y = self.avg_pool(x_in).view(b, c)
#         y = self.fc(y).view(b, self.oup * self.exp, 1, 1)
#         if self.exp == 4:
#             a1, b1, a2, b2 = torch.split(y, self.oup, dim=1)
#             a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
#             a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]

#             b1 = b1 - 0.5 + self.init_b[0]
#             b2 = b2 - 0.5 + self.init_b[1]
#             out = torch.max(x_out * a1 + b1, x_out * a2 + b2)
#         elif self.exp == 2:
#             if self.use_bias:  # bias but not PL
#                 a1, b1 = torch.split(y, self.oup, dim=1)
#                 a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
#                 b1 = b1 - 0.5 + self.init_b[0]
#                 out = x_out * a1 + b1

#             else:
#                 a1, a2 = torch.split(y, self.oup, dim=1)
#                 a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
#                 a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]
#                 out = torch.max(x_out * a1, x_out * a2)

#         elif self.exp == 1:
#             a1 = y
#             a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
#             out = x_out * a1

#         if self.spa:
#             ys = self.spa(x_in).view(b, -1)
#             ys = F.softmax(ys, dim=1).view(b, 1, h, w) * h * w
#             ys = F.hardtanh(ys, 0, 3, inplace=True)/3
#             out = out * ys

#         return out

# class DyDCNv2(nn.Module):
#     """ModulatedDeformConv2d with normalization layer used in DyHead.
#     This module cannot be configured with `conv_cfg=dict(type='DCNv2')`
#     because DyHead calculates offset and mask from middle-level feature.
#     Args:
#         in_channels (int): Number of input channels.
#         out_channels (int): Number of output channels.
#         stride (int | tuple[int], optional): Stride of the convolution.
#             Default: 1.
#         norm_cfg (dict, optional): Config dict for normalization layer.
#             Default: dict(type='GN', num_groups=16, requires_grad=True).
#     """

#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  stride=1,
#                  norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)):
#         super().__init__()
#         self.with_norm = norm_cfg is not None
#         bias = not self.with_norm
#         self.conv = ModulatedDeformConv2d(
#             in_channels, out_channels, 3, stride=stride, padding=1, bias=bias)
#         if self.with_norm:
#             self.norm = build_norm_layer(norm_cfg, out_channels)[1]

#     def forward(self, x, offset, mask):
#         """Forward function."""
#         x = self.conv(x.contiguous(), offset, mask)
#         if self.with_norm:
#             x = self.norm(x)
#         return x


# class DyHeadBlock(nn.Module):
#     """DyHead Block with three types of attention.
#     HSigmoid arguments in default act_cfg follow official code, not paper.
#     https://github.com/microsoft/DynamicHead/blob/master/dyhead/dyrelu.py
#     """

#     def __init__(self,
#                  in_channels,    # 输入特征的通道数
#                  norm_type='GN',  # 归一化的类型，默认为'GN'（组归一化）。
#                  zero_init_offset=True,  # 是否将偏移量初始化为零，默认为True。
#                  act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0)):  # 激活函数的配置，默认为dict(type='HSigmoid', bias=3.0, divisor=6.0)
#         super().__init__()
#         self.zero_init_offset = zero_init_offset
#         # (offset_x, offset_y, mask) * kernel_size_y * kernel_size_x
#         self.offset_and_mask_dim = 3 * 3 * 3   # 偏移量和掩码的总维度 
#         self.offset_dim = 2 * 3 * 3  # 偏移量的维度

#         if norm_type == 'GN':
#             norm_dict = dict(type='GN', num_groups=16, requires_grad=True)
#         elif norm_type == 'BN':
#             norm_dict = dict(type='BN', requires_grad=True)
        
#         # 使用DyDCNv2模块实例化的可变形卷积层，用于处理中间级别的特征。
#         # 这些可变形卷积层根据偏移量和掩码进行卷积操作，生成中间级别特征的输出。
#         self.spatial_conv_high = DyDCNv2(in_channels, in_channels, norm_cfg=norm_dict)
#         self.spatial_conv_mid = DyDCNv2(in_channels, in_channels)
#         self.spatial_conv_low = DyDCNv2(in_channels, in_channels, stride=2)
        
#         # 普通的卷积层，用于生成可变形卷积中的偏移量和掩码。它的输入是中间级别的特征，输出维度为3x3x3，
#         # 其中前半部分表示偏移量，后半部分表示掩码。
#         self.spatial_conv_offset = nn.Conv2d(
#             in_channels, self.offset_and_mask_dim, 3, padding=1)
        
#         # 尺度注意力模块，用于对中间级别的特征进行缩放。它首先通过自适应平均池化层将特征降维到1x1大小
#         # 然后通过一个卷积层和激活函数对特征进行处理，最后得到缩放后的特征。
#         self.scale_attn_module = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, 1, 1),
#             nn.ReLU(inplace=True), build_activation_layer(act_cfg))
        
#         # 动态ReLU模块，用于对特征进行动态调整。它根据输入特征生成动态参数
#         # 然后根据这些参数对输入特征进行加权或选择操作。
#         self.task_attn_module = DyReLU(in_channels)
#         self._init_weights()

#     # 初始化模块中的卷积层参数和偏移量。
#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 normal_init(m, 0, 0.01)
#         if self.zero_init_offset:
#             constant_init(self.spatial_conv_offset, 0)

#     def forward(self, x):
#         """Forward function."""
#         outs = []
        
#         for level in range(len(x)):
#             # 首先，对每个级别的特征计算出偏移量和掩码。
#             # 然后，分别对当前级别的特征、上一级别的特征和下一级别的特征应用可变形卷积层
#             # 并根据缩放注意力模块对卷积结果进行缩放和加权。
#             # calculate offset and mask of DCNv2 from middle-level feature
#             offset_and_mask = self.spatial_conv_offset(x[level])
#             offset = offset_and_mask[:, :self.offset_dim, :, :]
#             mask = offset_and_mask[:, self.offset_dim:, :, :].sigmoid()

#             mid_feat = self.spatial_conv_mid(x[level], offset, mask)
#             sum_feat = mid_feat * self.scale_attn_module(mid_feat)
#             summed_levels = 1
#             if level > 0:
#                 low_feat = self.spatial_conv_low(x[level - 1], offset, mask)
#                 sum_feat += low_feat * self.scale_attn_module(low_feat)
#                 summed_levels += 1
#             if level < len(x) - 1:
#                 # this upsample order is weird, but faster than natural order
#                 # https://github.com/microsoft/DynamicHead/issues/25
#                 high_feat = F.interpolate(
#                     self.spatial_conv_high(x[level + 1], offset, mask),
#                     size=x[level].shape[-2:],
#                     mode='bilinear',
#                     align_corners=True)
#                 sum_feat += high_feat * self.scale_attn_module(high_feat)
#                 summed_levels += 1
#             # 最后，将动态ReLU模块应用于缩放后的特征，得到最终的输出。
#             outs.append(self.task_attn_module(sum_feat / summed_levels))

#         return outs
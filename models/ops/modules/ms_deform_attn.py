# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MSDeformAttnFunction


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        '''
        生成一個[0,1,...,n_heads-1]的序列，然後乘上2π/n_heads
        生成 (n_heads, 2) 的張量，每一行代表一個頭在單位圓上的二維座標 [x, y]
        每個座標除以其最大絕對值，使其值保持在 [-1, 1] 的範圍內，這個操作確保了這些座標點在單位圓上
        '''
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        
        ''' 
        grid_init: (n_heads, n_levels, n_points, 2) 
        '''
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        
        ''' 
        為每個 sampling point按比例放大偏移量
        '''
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1

        '''
        將計算好的採樣偏移量 grid_init 作為 sampling_offsets 線性層的偏置進行初始化
        '''
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        '''
        input_flatten: (N, Len_in, C)
        value: (N, Len_in, C)
        '''
        value = self.value_proj(input_flatten)

        '''
        input_padding_mask[..., None] 將 input_padding_mask 從形狀 (N, Len_in) 擴展為 (N, Len_in, 1)，以便與 value 的形狀 (N, Len_in, C) 兼容
        將符合條件的 value 中的元素填充為 0
        '''
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        '''
        value: (N, Len_in, C) -> (N, Len_in, n_heads, C // n_heads)
        '''
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)

        '''
        sampling_offsets(query): (N, Len_q, n_heads * n_levels * n_points * 2) -> (N, Len_q, n_heads, n_levels, n_points, 2)
        attention_weights(query): (N, Len_q, n_heads * n_levels * n_points) -> (N, Len_q, n_heads, n_levels*n_points)
        接著將 attention_weights 進行 softmax，且是在所有採樣點的注意力權重之間進行歸一化，而不是在每個層次的採樣點之間分別歸一化
        attention_weights: (N, Len_q, n_heads, n_levels, n_points)
        '''
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        
        '''
        1.
            reference_points: (N, Len_q, n_levels, 2) (x,y) 用來指示這個查詢點在特徵圖中的位置
            input_spatial_shapes: (n_levels, 2) 表示每個特徵層次的空間維度 (height, width)
            offset_normalizer: 將 input_spatial_shapes 的每個特徵層次的 width 和 height 堆疊起來，生成一個新的張量
            reference_points[:, :, None, :, None, :]：將 reference_points 的形狀擴展為 (N, Len_q, 1, n_levels, 1, 2)，以便與 sampling_offsets 進行操作
            offset_normalizer[None, None, None, :, None, :]: (n_levels, 2) -> (1, 1, 1, n_levels, 1, 2)
            sampling_offsets: (N, Len_q, n_heads, n_levels, n_points, 2)
            將 sampling_offsets 除以 offset_normalizer，將偏移量正規化為特徵圖的比例偏移值
            再將正規化的 sampling_offsets 加到 reference_points 上，生成最終的採樣位置 sampling_locations，其形狀為 (N, Len_q, n_heads, n_levels, n_points, 2)
        2.
            reference_points: (N, Len_q, n_levels, 4) (x,y,w,h)
            reference_points[:, :, None, :, None, :2]: 提取 reference_points 的前兩個維度 (x, y)，並將形狀擴展為 (N, Len_q, 1, n_levels, 1, 2)，以便與 sampling_offsets 進行操作
            sampling_offsets: (N, Len_q, n_heads, n_levels, n_points, 2)
            sampling_offsets / self.n_points : 正規化
            偏移量還要乘以參考框的寬度和高度的一半，即 (w/2, h/2)，以適應參考框的大小
            將計算得到的偏移量加到 (x, y) 坐標上，生成最終的採樣位置 sampling_locations，其形狀為 (N, Len_q, n_heads, n_levels, n_points, 2)
        '''
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        

        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        
        output = self.output_proj(output)
        return output

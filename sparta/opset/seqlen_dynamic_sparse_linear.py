# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import copy
import tempfile
from cmath import inf
import torch
import types
import logging

from .sparse_opbase import SparseOPBase
from sparta.common.utils import *

import seqlen_dynamic_sparse_linear_cpp
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class SeqlenDynamicSparseLinearFunction(torch.autograd.Function):
    """
    支持序列长度动态变化的稀疏线性层的自动求导函数
    实现了前向和反向传播的计算逻辑
    """

    @staticmethod
    def forward(
        ctx,
        activation,    # 输入激活张量 [batch_size, seq_len, in_features]
        weight,        # 稀疏权重矩阵 [out_features, in_features]
        bias,          # 偏置向量 [out_features] (可为None)
        seqlens        # 序列长度张量 [batch_size]，指定每个样本的有效长度
    ):
        # 保存输入张量用于反向传播
        ctx.save_for_backward(
            activation,
            weight,
            bias,
            seqlens
        )
        # 打印activation信息
        print("\n[activation] Shape:", activation.shape)
        print("          Dtype:", activation.dtype)
        print("          Device:", activation.device)
        print("          First 5 values:", activation.flatten()[:5].tolist())
        
        # 打印weight信息
        print("\n[weight] Shape:", weight.shape)
        print("        Dtype:", weight.dtype)
        print("        Device:", weight.device)
        print("        First 5 values:", weight.flatten()[:5].tolist())
        
        # 打印bias信息（注意处理None）
        if bias is not None:
            print("\n[bias] Shape:", bias.shape)
            print("      Dtype:", bias.dtype)
            print("      Device:", bias.device)
            print("      First 5 values:", bias.flatten()[:5].tolist())
        else:
            print("\n[bias] None")
            
        # 打印seqlens信息
        print("\n[seqlens] Shape:", seqlens.shape)
        print("         Dtype:", seqlens.dtype)
        print("         Device:", seqlens.device)
        print("         Values:", seqlens.tolist())
        print("="*50 + "\n")
        
        # 根据是否有偏置调用不同的C++扩展函数
        if bias is not None:
            return seqlen_dynamic_sparse_linear_cpp.forward(
                activation,
                weight,
                bias,
                seqlens
            )
        else:
            return seqlen_dynamic_sparse_linear_cpp.forward2(
                activation,
                weight,
                seqlens
            )

    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        反向传播计算梯度
        注意：当前未实现偏置和序列长度的梯度计算
        """
        # 从上下文中获取保存的张量
        activation, weight, bias, seqlens = ctx.saved_tensors
        
        # 调用C++扩展计算输入激活和权重的梯度
        a_grad, w_grad = seqlen_dynamic_sparse_linear_cpp.backward(
            activation, weight, seqlens, grad_outputs[0]
        )
        
        # 返回对应输入的梯度（偏置和seqlens的梯度设为None）
        return a_grad, w_grad, None, None


class SeqlenDynamicSparseLinear(SparseOPBase):
    """
    支持动态稀疏模式的线性层模块
    能够根据序列长度动态调整稀疏计算模式
    """
    # 全局序列长度设置（用于全局模式，所有实例共享）
    global_seqlen = None

    @staticmethod
    def set_global_seqlens(seqlens):
        """
        设置全局序列长度
        seqlens: 一维张量 [batch_size]，每个元素表示对应样本的有效序列长度
        """
        assert isinstance(seqlens, torch.Tensor), "seqlens必须是PyTorch张量"
        assert seqlens.is_cuda, "seqlens必须在GPU上"
        assert seqlens.dtype == torch.int32, "seqlens必须是int32类型"
        SeqlenDynamicSparseLinear.global_seqlen = seqlens

    def __init__(self, ori_linear, global_mode=True):
        """
        初始化动态稀疏线性层
        
        参数:
        ori_linear: 原始的PyTorch线性层，用于继承权重和偏置
        global_mode: 是否使用全局模式（所有实例共享相同稀疏模式）
        """
        super(SeqlenDynamicSparseLinear, self).__init__()
        self.global_mode = global_mode
        
        # 中间结果缓存（当前未使用）
        self.inter_result = None
        
        # 继承原始线性层的权重、偏置和引用
        self.weight = ori_linear.weight
        self.bias = ori_linear.bias
        self.ori_linear = ori_linear

    def forward(self, activation, seqlens=None):
        """
        前向传播计算
        
        参数:
        activation: 输入激活张量
        seqlens: 序列长度张量（仅在local模式下需要）
        """
        # 确保输入张量是连续的（内存布局优化）
        if not activation.is_contiguous():
            activation = activation.contiguous()
            
        # 根据模式获取序列长度
        if not self.global_mode:
            # 局部模式：必须提供seqlens且batch_size匹配
            assert isinstance(seqlens, torch.Tensor), "局部模式下必须提供seqlens"
            assert seqlens.size(0) == activation.size(0), "seqlens的batch_size不匹配"
        else:
            # 全局模式：使用预设置的全局seqlens并移至相同设备
            seqlens = SeqlenDynamicSparseLinear.global_seqlen.to(activation.device)
            
        # 调用自定义autograd函数执行稀疏矩阵乘法
        result = SeqlenDynamicSparseLinearFunction.apply(
            activation, self.weight, self.bias, seqlens
        )
        return result

    def reference_forward(self, activation):
        """
        使用原始线性层计算参考结果
        用于验证稀疏计算的正确性
        """
        ref_out = self.ori_linear(activation)
        return ref_out
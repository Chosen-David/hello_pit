import torch
import torch.nn as nn
# from sparta.tuning.sparse_linear import SeqlenDynamicSparseLinear
# from ../opset/seqlen_dynamic_sparse_linear.py import SeqlenDynamicSparseLinear

import os
import sys
# 获取项目根目录（SparTA）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
print(f"Project root: {project_root}")
sys.path.append(project_root)
from sparta.opset.seqlen_dynamic_sparse_linear import SeqlenDynamicSparseLinear
def test_seqlen_dynamic_sparse_linear(
    batch_size=32,
    seq_len=128,
    in_features=768,
    out_features=3072,
    test_times=10,
    use_global_mode=True,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    测试 SeqlenDynamicSparseLinear 的前向传播功能

    参数:
    - batch_size: 输入 batch 大小
    - seq_len: 序列长度
    - in_features: 输入维度
    - out_features: 输出维度
    - test_times: 测试次数
    - use_global_mode: 是否使用全局模式
    - device: 使用设备（"cuda" 或 "cpu"）
    """
    # 确保设备可用
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA 不可用，自动切换到 CPU")

    # 创建原始线性层
    linear = nn.Linear(in_features, out_features).to(device)

    # 替换为稀疏线性层
    sparse_linear = SeqlenDynamicSparseLinear(linear, global_mode=use_global_mode).to(device)

    # 生成随机序列长度（每个样本的有效长度）
    seqlens = torch.randint(1, seq_len + 1, (batch_size,)).int().to(device)

    # 测试 10 次
    for i in range(test_times):
        # 构造随机输入激活 [batch_size, seq_len, in_features]
        activation = torch.randn(batch_size, seq_len, in_features, device=device)

        # 设置全局序列长度（如果使用 global_mode）
        if use_global_mode:
            SeqlenDynamicSparseLinear.set_global_seqlens(seqlens)

        # 前向传播
        with torch.no_grad():
            if use_global_mode:
                output = sparse_linear(activation)
            else:
                output = sparse_linear(activation, seqlens)

        # 与原始线性层结果对比
        with torch.no_grad():
            ref_output = linear(activation)

        # 验证输出形状
        assert output.shape == ref_output.shape, f"输出形状不匹配: {output.shape} vs {ref_output.shape}"

        # 计算误差
        diff = (output - ref_output).abs().mean().item()

        # 打印信息
        print(f"Test {i+1}: Output shape = {output.shape}")
        print(f"  Output min/max: {output.min().item():.4f} / {output.max().item():.4f}")
        print(f"  Reference min/max: {ref_output.min().item():.4f} / {ref_output.max().item():.4f}")
        print(f"  Mean absolute diff: {diff:.6f}\n")

    print("✅ 测试完成")


# 如果是主程序，运行测试
if __name__ == "__main__":
    test_seqlen_dynamic_sparse_linear()
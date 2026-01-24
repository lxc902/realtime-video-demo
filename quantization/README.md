# 量化模块显存观察记录

## 测试环境

- **Ada48**: 48GB VRAM (RTX 6000 Ada, NVIDIA RTX 6000 Ada Generation, 49140 MiB)
- **Pro96**: 96GB VRAM (RTX Pro 6000, NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97887 MiB)

## 显存使用观察

### INT8 量化 (torchao)

| 阶段 | Ada48 | Pro96 |
|------|-------|-------|
| 加载模型 | 40GB | 40GB |
| 量化完成后 | 26GB | 26GB |
| 每帧推理增量 | +2GB | +9GB |

### FP8 量化 (ComfyUI 实现)

| 阶段 | Ada48 | Pro96 |
|------|-------|-------|
| 加载完成 | 25GB | 28GB |
| 每帧推理增量 | +2GB | +9GB |

## 问题分析：推理内存爆炸

### 1. torch.compile / torch._dynamo 缓存累积

每次推理时，`torch._dynamo` 会缓存编译的计算图。虽然 `cleanup_inference()` 中调用了 `torch._dynamo.reset()`，但可能有缓存没有被完全释放。

### 2. FP8 forward 中重复创建张量

`fp8.py` 中的 `fp8_linear_forward` 每次调用都会创建新的 scale 张量：

```python
scale_input = torch.ones((), device=input.device, dtype=torch.float32)
```

这应该改为在 `convert_fp8_linear` 时预创建并复用。

### 3. 中间张量副本

`.contiguous()` 调用会创建新的张量副本：

```python
inn = input.reshape(-1, input_shape[2]).to(torch.float8_e4m3fn).contiguous()
```

### 4. flex_attention block_mask 缓存

Transformer 使用的 flex_attention 会缓存 block_mask，这些缓存在连续推理时会累积。

### 5. Ada vs Blackwell 差异解释

Pro96 每帧消耗 9GB vs Ada48 的 2GB，可能原因：

1. **架构差异**：RTX 6000 Ada 是 Ada Lovelace (CC 8.9)，RTX Pro 6000 是 Blackwell (CC 10.0+)
2. **不同的 kernel 路径**：`torch._scaled_mm` 在 Blackwell 上可能使用不同的实现
3. **更大的并行度**：Blackwell 有更多 CUDA 核心，需要更大的中间激活批次
4. **新架构驱动优化**：Blackwell 驱动可能更激进地预分配 memory pool

## 已实施的优化

1. ✅ **FP8 scale_input 复用** (`fp8.py`)：预创建 `_scale_input_cache` 并复用，避免每次 forward 创建新张量
2. ✅ **每帧清理显存** (`local_inference.py`)：`generate_next_block` 结束时调用 `torch.cuda.empty_cache()`
3. ✅ **使用 inference_mode** (`local_inference.py`)：用 `torch.inference_mode()` 包装推理，比 `no_grad` 更激进
4. ✅ **显式删除临时变量** (`local_inference.py`)：推理后 `del kwargs`
5. ✅ **INT8 加载后清理** (`int8.py`)：量化完成后 `gc.collect()` + `torch.cuda.synchronize()`

## 待深入研究

1. 调查 `torch._scaled_mm` 的内存行为
2. 检查 flex_attention 缓存是否有官方的清理方法
3. 考虑使用 `torch.cuda.memory_stats()` 进行详细的内存分析
4. Blackwell 架构上 PyTorch 的 FP8 实现可能需要特殊优化

## 相关代码文件

- `fp8.py` - FP8 量化实现
- `int8.py` - INT8 量化实现
- `local_inference.py` - 推理逻辑和缓存清理

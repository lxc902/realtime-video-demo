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

### 根本原因：diffusers ModularPipeline 的 deepcopy

**这是最主要的原因！** 每次调用 `pipe()` 时，diffusers 的 `ModularPipeline.__call__` 会对 `state` 进行 `deepcopy`：

```python
# diffusers/modular_pipelines/modular_pipeline.py:2559
state = deepcopy(state)
```

`state.values` 中累积了生成的视频帧张量（GPU 上），deepcopy 会尝试克隆所有张量，随着推理次数增加，所需内存不断翻倍直到 OOM。

**解决方案**：在调用 `pipe()` 之前清理 state 中的大张量（见 `_cleanup_state_tensors` 方法）。

### 其他次要原因

1. **torch.compile / torch._dynamo 缓存累积**：编译的计算图会缓存

2. **FP8 forward 中重复创建张量**：每次 forward 创建 `scale_input`（已修复：使用 `_scale_input_cache`）

3. **flex_attention block_mask 缓存**：Transformer 使用的 flex_attention 会缓存 block_mask

### Ada vs Blackwell 差异解释

Pro96 每帧消耗 9GB vs Ada48 的 2GB，可能原因：

1. **架构差异**：RTX 6000 Ada 是 Ada Lovelace (CC 8.9)，RTX Pro 6000 是 Blackwell (CC 10.0+)
2. **不同的 kernel 路径**：`torch._scaled_mm` 在 Blackwell 上可能使用不同的实现
3. **更大的并行度**：Blackwell 有更多 CUDA 核心，需要更大的中间激活批次
4. **新架构驱动优化**：Blackwell 驱动可能更激进地预分配 memory pool

## 已实施的优化

1. ✅ **清理 state 中的大张量** (`local_inference.py`)：`_cleanup_state_tensors()` 在调用 pipe 前清理 videos, decoder_cache, video_stream, kv_cache, crossattn_cache 等大张量，防止内存累积
2. ✅ **FP8 scale_input 复用** (`fp8.py`)：预创建 `_scale_input_cache` 并复用，避免每次 forward 创建新张量
3. ✅ **每帧清理显存** (`local_inference.py`)：`generate_next_block` 结束时调用 `gc.collect()` + `torch.cuda.empty_cache()` + `torch.cuda.synchronize()`
4. ✅ **使用 inference_mode** (`local_inference.py`)：用 `torch.inference_mode()` 包装推理，比 `no_grad` 更激进
5. ✅ **显式删除临时变量** (`local_inference.py`)：推理后 `del kwargs`
6. ✅ **INT8 加载后清理** (`int8.py`)：量化完成后 `gc.collect()` + `torch.cuda.synchronize()`
7. ❌ ~~每帧重置 dynamo 缓存~~：已移除，因为重复编译比缓存复用更耗内存
8. ✅ **诊断信息增强** (`local_inference.py`)：打印 state.values 中的 key，便于调试内存问题

## 重要警告

⚠️ **无量化 BF16 模式需要 54GB+ 显存**，即使在 96GB GPU 上也很紧张。强烈建议使用量化：

```bash
# 推荐用法
bash run.sh --fp8   # FP8 量化，~25GB 显存
bash run.sh --int8  # INT8 量化，~26GB 显存
```

## 待深入研究

1. 调查 `torch._scaled_mm` 的内存行为
2. 检查 flex_attention 缓存是否有官方的清理方法
3. 考虑使用 `torch.cuda.memory_stats()` 进行详细的内存分析
4. Blackwell 架构上 PyTorch 的 FP8 实现可能需要特殊优化

## 相关代码文件

- `fp8.py` - FP8 量化实现
- `int8.py` - INT8 量化实现
- `local_inference.py` - 推理逻辑和缓存清理

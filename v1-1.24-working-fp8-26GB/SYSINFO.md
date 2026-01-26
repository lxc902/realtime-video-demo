## Krea Realtime 14B 视频生成模型 GPU 性能对比报告（扩展版）

以下表格基于 **NVIDIA B200 GPU 运行 Krea Realtime 14B 模型（4步推理）的官方性能（11 FPS）** 为基准，通过理论算力比例与架构代际关系综合推算各 GPU 的**大致推理速度**。  
已**补充所有理论或实际性能明确高于 RTX 6000 Ada 的 NVIDIA GPU**，用于完整选型参考。

| GPU | 显存 | 主要算力指标（参考） | 预估 FPS（基于 B200 = 11 FPS） | 关键说明 |
| :--- | :--- | :--- | :--- | :--- |
| **NVIDIA B200** | **192GB HBM3e** | Blackwell / FP8 极强 | **11.0** | 官方实测基准 |
| **NVIDIA H200 SXM** | **141GB HBM3e** | Hopper / 超高带宽 | **≈10.0** | HBM3e 对视频模型极其友好 |
| **RTX Pro 6000 (Blackwell)** | **96GB GDDR7** | ~125 TFLOPS FP32 | **≈9.5** | 单卡显存最大的专业卡 |
| **NVIDIA H100 SXM** | 80GB HBM3 | ~60 TFLOPS FP32 / FP8 强 | **≈8.8** | Tensor Core 极强，FP8 友好 |
| **NVIDIA H800 SXM** | 80GB HBM2e | ~60 TFLOPS FP32 | **≈8.0** | 中国特供版 H100 |
| **RTX 5090** | 32GB GDDR7 | ~105 TFLOPS FP32 | **≈8.2** | 消费级最强，显存受限 |
| **RTX 4090** | 24GB GDDR6X | 82.6 TFLOPS FP32 | **≈7.0** | 性价比高，但显存偏小 |
| **RTX 8000 Ada** | **48GB GDDR6** | ~91 TFLOPS FP32 | **≈6.8–7.2** | 明显强于 RTX 6000 Ada |
| **RTX 6000 Ada** | 48GB GDDR6 | 91.1 TFLOPS FP32 | **≈3.6** | 本表性能分界线 |
| **L40S** | 48GB GDDR6 | FP8 Tensor 强 | **≈3.6** | 云端推理卡 |
| **L40** | 48GB GDDR6 | FP8 Tensor 强 | **≈3.6** | 与 L40S 接近 |
| **A100 80G** | 80GB HBM2e | 19.5 TFLOPS FP32 | **≈2.0** | 老一代训练卡 |
| **A800 80G** | 80GB HBM2e | 19.5 TFLOPS FP32 | **≈0.8** | 不支持 FP8 |
| **H20** | 96GB HBM | FP8 推理取向 | **≈0.7** | 合规推理卡 |
| **L20** | 48GB GDDR6 | 中端推理 | **≈0.6** | 性能有限 |
| **华为昇腾 910B** | 64GB HBM | ~320 TFLOPS FP16 | **≈0.8** | 非 CUDA 生态 |

> 注：
> - “预估 FPS”用于**同模型、同推理步数（4-step）、同分辨率**下的横向比较  
> - 实际性能受显存带宽、FP8 支持、Kernel 优化、驱动与框架影响显著  
> - **RTX 6000 Ada 以上（表格上半部分）均属于可跑 Krea Realtime 14B 的“性能安全区”**



(main) root@C.30438521:/workspace/realtime-video-demo$ ./tmp/venv/bin/python -V
Python 3.12.12
(main) root@C.30438521:/workspace/realtime-video-demo$ lsb_release  -a
No LSB modules are available.
Distributor ID: Ubuntu
Description:    Ubuntu 24.04.3 LTS
Release:        24.04
Codename:       noble
(main) root@C.30438521:/workspace/realtime-video-demo$ uname -a
Linux 62275f3dbcdf 5.15.0-161-generic #171-Ubuntu SMP Sat Oct 11 08:17:01 UTC 2025 x86_64 x86_64 x86_64 GNU/Linux
(main) root@C.30438521:/workspace/realtime-video-demo$ ./tmp/venv/bin/python -m pip -V
pip 25.0.1 from /workspace/realtime-video-demo/tmp/venv/lib/python3.12/site-packages/pip (python 3.12)
(main) root@C.30438521:/workspace/realtime-video-demo$ 


(main) root@C.30438521:/workspace/realtime-video-demo$ nvidia-smi
Sun Jan 25 17:03:20 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.95.05              Driver Version: 580.95.05      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX PRO 6000 Blac...    On  |   00000000:01:00.0 Off |                  Off |
| 30%   26C    P8             16W /  600W |       2MiB /  97887MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

(main) root@C.30438521:/workspace/realtime-video-demo$ cat /proc/driver/nvidia/version
NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  580.95.05  Release Build  (dvs-builder@U22-I3-B17-02-5)  Tue Sep 23 09:55:41 UTC 2025
GCC version:  gcc version 11.4.0 (Ubuntu 11.4.0-1ubuntu1~22.04.2) 




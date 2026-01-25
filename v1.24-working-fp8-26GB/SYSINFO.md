## Krea Realtime 14B 视频生成模型 GPU 性能对比报告

以下表格基于 **NVIDIA B200 GPU 运行 Krea Realtime 14B 模型（4步推理）的官方性能（11 FPS）** 为基准，通过理论算力比例推算出各GPU的大致推理速度，并汇总了关键的显存信息，供您选型参考。

| GPU | 显存 | FP32 算力 (参考) | 预估 FPS (基于 B200 11 FPS 推算) | 关键说明 |
| :--- | :--- | :--- | :--- | :--- |
| **NVIDIA B200** | **192GB HBM3e** | - | **11** (官方基准) | 官方基准，所有推算的源头。 |
| **RTX Pro 6000 (Blackwell)** | **96GB GDDR7** | **125 TFLOPS** | **≈ 9.5** | 新一代专业旗舰，拥有最大的单卡显存和顶级算力。 |
| **RTX 5090** | 32GB GDDR7 | ~105 TFLOPS | **≈8.2** | 新一代消费级旗舰，性能强劲。 |
| **RTX 4090** | **24GB GDDR6X** | **82.58 TFLOPS** | **≈7.0** | 上一代消费级旗舰，性能依然强大，但显存是主要瓶颈。 |
| **H100 / H800** | 80GB HBM2e | 67 TFLOPS | **≈4.8** | 数据中心主力卡，但FP32算力已不及新一代卡。 |
| **RTX 6000 Ada** | 48GB GDDR6 | 91.1 TFLOPS | **≈3.6** | 上一代专业卡，作为性能参照基线。 |
| **L40S** | 48GB GDDR6 | ~1,466 TFLOPS (FP8) | **≈3.6** | 云服务器推理卡。 |
| **L40** | 48GB GDDR6 | ~1,466 TFLOPS (FP8) | **≈3.6** | 性能与L40S相近。 |
| **A800 (80G)** | 80GB HBM2e | 19.5 TFLOPS (FP32) | **≈0.8** | **不支持FP8加速**，架构较老，推理速度慢。 |
| **H20** | 96GB HBM2e | 296 TFLOPS (FP8) | **≈0.7** | 专为推理设计，但算力较低。 |
| **L20** | 48GB GDDR6 | 239 TFLOPS (INT8/FP8) | **≈0.6** | 中端推理卡。 |
| **华为昇腾 910B** | 64GB HBM | 320 TFLOPS (FP16) | **≈0.8** | **非NVIDIA平台**，需专门优化，预估仅供参考。 |

> **注：** 表中“预估 FPS”主要依据各GPU与B200的理论算力比例进行推算，实际性能会受显存带宽、驱动优化、模型配置（如分辨率、步数）等因素影响，此数据适用于初步比较和选型参考。



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




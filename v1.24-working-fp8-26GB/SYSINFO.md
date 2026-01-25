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


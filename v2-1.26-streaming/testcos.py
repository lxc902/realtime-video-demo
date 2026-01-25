from qcloud_cos import CosConfig, CosS3Client
import os

# ========= 配置 =========
SECRET_ID = ""
SECRET_KEY = ""
REGION = "ap-nanjing"
BUCKET = "rtcos-1394285684"

LOCAL_FILE = "./a.txt"
COS_KEY = "data/a.txt"
# =======================

# 创建本地文件
with open(LOCAL_FILE, "w") as f:
    f.write("123")

# COS 客户端
config = CosConfig(
    Region=REGION,
    SecretId=SECRET_ID,
    SecretKey=SECRET_KEY,
    Scheme="https"
)
client = CosS3Client(config)

# 上传
resp = client.upload_file(
    Bucket=BUCKET,
    LocalFilePath=LOCAL_FILE,
    Key=COS_KEY
)

print("Upload OK, ETag =", resp.get("ETag"))
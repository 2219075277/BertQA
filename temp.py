import os
import requests

# 定义要下载的文件列表
files_to_download = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "spiece.model",
    "pytorch_model.bin"
]

# 定义本地保存目录
local_dir = "t5-small"
os.makedirs(local_dir, exist_ok=True)

# 定义镜像源
mirror_url = "https://hf-mirror.com/t5-small/resolve/main/"

for file in files_to_download:
    file_url = mirror_url + file
    local_path = os.path.join(local_dir, file)
    # 检查文件是否已经存在
    if os.path.exists(local_path):
        print(f"{file} 已存在，跳过下载。")
        continue
    try:
        print(f"开始下载 {file}...")
        response = requests.get(file_url, stream=True)
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"{file} 下载完成。")
    except requests.RequestException as e:
        print(f"下载 {file} 时出错: {e}")

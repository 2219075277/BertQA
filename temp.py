import os
import requests

# 定义要下载的模型及其对应的文件列表
models = {
    "hfl/chinese-bert-wwm-ext": [
        "config.json",
        "pytorch_model.bin",
        "vocab.txt",
        "tokenizer_config.json"
    ],
    "uer/t5-small-chinese-cluecorpussmall": [
        "config.json",
        "pytorch_model.bin",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "spiece.model"
    ]
}

# 镜像源地址
mirror_url_base = "https://hf-mirror.com/"

for model_name, files in models.items():
    # 创建本地保存目录
    local_dir = model_name.replace("/", "-")
    os.makedirs(local_dir, exist_ok=True)


    # 构建模型的镜像源 URL
    mirror_url = mirror_url_base + model_name + "/resolve/main/"

    for file in files:
        file_url = mirror_url + file
        local_path = os.path.join(local_dir, file)
        if os.path.exists(local_path):
            print(f"{file} 在 {model_name} 中已存在，跳过下载。")
            continue
        try:
            print(f"开始下载 {model_name} 的 {file}...")
            response = requests.get(file_url, stream=True)
            response.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"{model_name} 的 {file} 下载完成。")
        except requests.RequestException as e:
            print(f"下载 {model_name} 的 {file} 时出错: {e}")



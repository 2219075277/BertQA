"""
# 推理入口，命令行输入问题生成答案

"""
# inference.py
import warnings
# 忽略 flash attention 相关的 UserWarning
warnings.filterwarnings(
    "ignore",
    message=".*flash attention.*",
    category=UserWarning,
)
import logging
logging.getLogger("transformers.modeling_bert").setLevel(logging.ERROR)

from models.fusion_model import FusionQAModel
import torch

question = "你叫什么名字?"
passages = ["这是一个关于人的问题。", "回答这个问题非常简单。"]

model = FusionQAModel(device='cuda')
# model.load_state_dict(torch.load("saved_model/best.pth"))
model.eval()
# … 先加载/训练模型权重 …
answer = model.generate_answer(
    question=question, passages=passages,max_length=64
)
print('问题：',question)
print("生成的答案：", answer)
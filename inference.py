"""
# 推理入口，命令行输入问题生成答案

"""
# inference.py
import json
from models.fusion_model import FusionQAModel

qa_model = FusionQAModel()

question = "What is the capital of France?"
passages = [
    "France is a country in Europe. It has a population of over 60 million.",
    "Paris is the capital and largest city of France.",
    "The Eiffel Tower is located in the capital city.",
]

answer = qa_model.generate_answer(question, passages)
print(f"Q: {question}\nA: {answer}")

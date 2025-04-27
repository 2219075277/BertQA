"""
# 推理入口，命令行输入问题生成答案

"""
# inference.py
import json
from models.fusion_model import FusionQAModel

qa_model = FusionQAModel()

question = "法国的首都是什么？"
passages = [
"法国是欧洲的一个国家。它有超过 6000 万人口。",
"巴黎是法国的首都，也是法国最大的城市。",
"埃菲尔铁塔位于这座首都城市。"
]

answer = qa_model.generate_answer(question, passages)
print(f"Q: {question}\nA: {answer}")

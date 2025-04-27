"""
启动API服务
"""
# api/app.py
from fastapi import FastAPI
from pydantic import BaseModel
from models.fusion_model import FusionQAModel
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有域名，如果想更安全可以改成你的前端域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法，包括OPTIONS
    allow_headers=["*"],  # 允许所有请求头
)
qa_model = FusionQAModel()
# qa_model.load_pretrained('你的模型路径')    # 预训练权重

class QuestionRequest(BaseModel):
    question: str
    passages: list

@app.post("/qa")
def generate_answer(request: QuestionRequest):
    answer = qa_model.generate_answer(request.question, request.passages)
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
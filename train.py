"""
训练窗口
"""

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
from models.fusion_model import FusionQAModel

# 设置训练数据集
def train(model, dataset, batch_size=16, num_epochs=3, lr=1e-5, save_dir="saved_model"):
    # 定义数据加载器
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 设置优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model.train()
    min_loss = float("inf")
    for epoch in tqdm(range(num_epochs)):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)

            # 前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            print(f"Loss: {loss.item()}")

        # 每个epoch结束后保存模型权重
        if loss.item() < min_loss:
            min_loss = loss.item()

            model.save_pretrained(save_dir)
            print(f"模型已保存到 {save_dir} 目录。")

# 加载中文SQuAD数据集（你可以使用其他中文数据集）
dataset = load_dataset("squad_zh")['train']  # 示例使用中文SQuAD数据集

# 创建模型并训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
qa_model = FusionQAModel(device=device)

# 开始训练
train(qa_model, dataset, save_dir="saved_model")

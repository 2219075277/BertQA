from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, BertTokenizerFast
from torch.optim import AdamW
from tqdm import tqdm
import torch
from transformers import BertTokenizer

from models.fusion_model import FusionQAModel
import warnings
import logging
from utils.logger import save_answer
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)

# 加载 BERT tokenizer
bert_tokenizer = BertTokenizer.from_pretrained("./hfl-chinese-bert-wwm-ext")
t5_tokenizer  = bert_tokenizer
# ------------------ 预处理 ------------------

def preprocess_dataset(raw_dataset, bert_tokenizer, t5_tokenizer):
    examples = []
    for item in raw_dataset["data"]:
        for para in item["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                question = qa["question"]
                # 取第一个答案
                answer_text = qa["answers"][0]["text"]

                # 编码 question + context
                enc = bert_tokenizer(
                    question,
                    context,
                    padding="max_length",
                    truncation="only_second",
                    max_length=512,
                )
                # 编码 answer 作为 labels
                ans = t5_tokenizer(
                    answer_text,
                    padding="max_length",
                    truncation=True,
                    max_length=64,
                )

                examples.append({
                    "input_ids":     torch.tensor(enc["input_ids"],     dtype=torch.long),
                    "attention_mask":torch.tensor(enc["attention_mask"],dtype=torch.long),
                    "labels":        torch.tensor(ans["input_ids"],     dtype=torch.long),
                })
    return examples



# ------------------ collate_fn ------------------

def collate_fn(batch):
    input_ids      = torch.stack([x["input_ids"]      for x in batch])
    attention_mask = torch.stack([x["attention_mask"] for x in batch])
    labels         = torch.stack([x["labels"]         for x in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }



# ------------------ train函数 ------------------

def train(model, dataset, batch_size=8, num_epochs=50, lr=1e-5):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optim = AdamW(model.parameters(), lr=lr)
    total_steps = len(dataloader) * num_epochs
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=total_steps)
    min_loss = float("inf")

    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for i, batch in tqdm(enumerate(dataloader)):
            for k,v in batch.items():
                batch[k] = v.to(model.device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            sched.step()
            if i % 100 == 0:
                print(f"loss:{loss}")
                question = "你是谁"
                passages = ["这是一个关于人的问题。", "回答这个问题非常简单。"]
                answer = model.generate_answer(
                    question=question, passages=passages, max_length=64
                )
                save_answer(question, answer, epoch, i)

        if loss.item() < min_loss:
            min_loss = loss.item()
            torch.save(model.state_dict(), "saved_model/best.pth")
        print(f"  Loss: {loss.item():.4f}")



# ------------------ 主流程 ------------------

if __name__ == "__main__":
    raw = load_dataset("json", data_files="./data/cmrc2018_train.json", split="train")
    dataset = preprocess_dataset(raw, bert_tokenizer, t5_tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qa_model = FusionQAModel(device=device)
    train(qa_model, dataset)
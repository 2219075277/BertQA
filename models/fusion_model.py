# models/fusion_model.py
"""
# BERT 编码器 + T5 解码器

"""
import torch
from transformers import BertModel, BertTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer

class FusionQAModel:
    def __init__(self, device='cuda'):
        self.device = device
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)

        self.t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

    def encode_passages(self, question, passages):
        """分别将每段passage与question编码，并返回embedding序列"""
        all_embeddings = []
        for passage in passages:
            text = f"[CLS] {question} [SEP] {passage} [SEP]"
            inputs = self.bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=384).to(self.device)
            outputs = self.bert_model(**inputs)
            all_embeddings.append(outputs.last_hidden_state)
        return all_embeddings

    def generate_answer(self, question, passages):
        with torch.no_grad():
            encodings = self.encode_passages(question, passages)
            # 将多个段落编码拼接为 decoder 输入
            stacked = torch.cat(encodings, dim=1)  # shape: (1, total_len, hidden_size)

            # 为 decoder 构造 dummy input（prompt）
            t5_input = self.t5_tokenizer("answer:", return_tensors="pt").input_ids.to(self.device)

            output = self.t5_model.generate(
                encoder_outputs=(stacked,),
                decoder_input_ids=t5_input,
                max_length=64,
                num_beams=4,
                early_stopping=True
            )

            return self.t5_tokenizer.decode(output[0], skip_special_tokens=True)

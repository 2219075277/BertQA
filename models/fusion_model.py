# models/fusion_model.py
"""
# BERT 编码器 + T5 解码器

"""
import torch
from transformers import BertModel, BertTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput
from torch import nn
from models.joint import Joint
# 设置镜像源
import os
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1" # 忽略警告
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.tuna.tsinghua.edu.cn'
class FusionQAModel:
    def __init__(self, device='cuda'):
        self.device = device

        print('🔄 Downloading BERT...')
        self.bert_tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext") # 	哈工大版，Whole Word Masking，理解力更强一点
        self.bert_model = BertModel.from_pretrained("hfl/chinese-bert-wwm-ext").to(device)
        print('✅ BERT loaded.')

        print('🔄 Loading T5...')
        self.t5_tokenizer = BertTokenizer.from_pretrained(r"uer/t5-small-chinese-cluecorpussmall")
        self.t5_model = T5ForConditionalGeneration.from_pretrained(r"uer/t5-small-chinese-cluecorpussmall").to(device)
        print('✅ T5 loaded.')

        # 添加 BERT → T5 的线性映射
        self.projection = Joint(in_channels=768, out_channels=512, hidden_channels=568)  # BERT输出768 → T5期望512

    def encode_passages(self, question, passages):
        """分别将每段passage与question编码，并返回embedding序列"""
        all_embeddings = []
        for passage in passages:
            text = f"[CLS] {question} [SEP] {passage} [SEP]"
            inputs = self.bert_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=384,
                padding='max_length'
            ).to(self.device)
            outputs = self.bert_model(**inputs)
            # bert_hidden = outputs.last_hidden_state  # (1, seq_len, 768)
            # projected = self.projection(bert_hidden)  # (1, seq_len, 512)
            all_embeddings.append(self.projection(outputs.last_hidden_state))  # shape: (1, seq_len, hidden_size)
        return all_embeddings

    def generate_answer(self, question, passages):
        with torch.no_grad():
            # 编码所有段落
            encodings = self.encode_passages(question, passages)

            # 限制拼接总长度，避免爆显存
            stacked = torch.cat(encodings, dim=1)  # shape: (1, total_len, hidden_size)
            if stacked.size(1) > 512:
                print("⚠️ Total token length too long, truncating to 512 tokens.")
                stacked = stacked[:, :512, :]

            encoder_outputs = BaseModelOutput(last_hidden_state=stacked)

            decoder_input = self.t5_tokenizer("answer:", return_tensors="pt").input_ids.to(self.device)

            output = self.t5_model.generate(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input,
                max_length=64,
                num_beams=4,
                early_stopping=True
            )

            return self.t5_tokenizer.decode(output[0], skip_special_tokens=True)
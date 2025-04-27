# models/fusion_model.py
"""
BERT 编码器 + T5 解码器
"""
import torch
import os
from torch import nn
from transformers import BertModel, BertTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput
from models.joint import Joint

# 设置镜像源
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.tuna.tsinghua.edu.cn'

class FusionQAModel(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device

        # BERT tokenizer 和模型
        self.bert_tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
        self.bert_model = BertModel.from_pretrained("hfl/chinese-bert-wwm-ext").to(device)

        # T5 tokenizer 和模型
        self.t5_tokenizer = BertTokenizer.from_pretrained("uer/t5-small-chinese-cluecorpussmall")
        self.t5_model = T5ForConditionalGeneration.from_pretrained("uer/t5-small-chinese-cluecorpussmall").to(device)

        # BERT到T5的映射（如果需要的话）
        self.projection = Joint(768, 512).to(device)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        # 使用 BERT 提取特征
        encoder_outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        bert_hidden_states = encoder_outputs.last_hidden_state

        # 如果需要映射到 T5 的隐藏层大小
        projected_hidden_states = self.projection(bert_hidden_states)
        encoder_outputs = BaseModelOutput(last_hidden_state=projected_hidden_states)

        # 使用 T5 的 decoder 来生成答案
        outputs = self.t5_model(
            input_ids=input_ids,  # BERT 编码后的 input_ids
            attention_mask=attention_mask,
            decoder_input_ids=None,  # Decoder 输入在这种生成任务下可能为空
            encoder_outputs=encoder_outputs,  # 提供给 T5 编码器的输入
            labels=labels  # 如果是训练的话，传入 labels，生成任务的目标答案
        )

        return outputs

    def generate_answer(self, question, passages, max_length=64, num_beams=4):
        """推理：给定问题和多个段落，生成答案文本"""
        self.eval()
        with torch.no_grad():
            all_embeddings = []
            for passage in passages:
                enc = self.bert_tokenizer(
                    question,
                    passage,
                    padding="max_length",
                    truncation="only_second",
                    max_length=512,
                    return_tensors="pt"
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                bert_out = self.bert_model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    token_type_ids=enc.get("token_type_ids", None)
                )
                proj = self.projection(bert_out.last_hidden_state)
                all_embeddings.append(proj)

            # 拼接
            encoder_hidden = torch.cat(all_embeddings, dim=1)
            if encoder_hidden.size(1) > 512:
                encoder_hidden = encoder_hidden[:, :512, :]

            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)

            generated_ids = self.t5_model.generate(
                encoder_outputs=encoder_outputs,
                decoder_start_token_id=self.t5_tokenizer.cls_token_id,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )
            return self.t5_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

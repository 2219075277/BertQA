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
        projected_hidden_states = self.projection(bert_hidden_states, mask=attention_mask)
        encoder_outputs = BaseModelOutput(last_hidden_state=projected_hidden_states)
        # print(projected_hidden_states.abs().mean())

        # 使用 T5 的 decoder 来生成答案
        outputs = self.t5_model(
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

                # BERT编码
                bert_out = self.bert_model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    token_type_ids=enc.get("token_type_ids", None)
                )

                # 对BERT输出进行归一化
                hidden_states = bert_out.last_hidden_state
                hidden_norm = torch.norm(hidden_states, p=2, dim=-1, keepdim=True)
                hidden_states = hidden_states / (hidden_norm + 1e-7)

                # 投影并添加到列表
                proj = self.projection(hidden_states, mask=enc["attention_mask"])
                all_embeddings.append(proj)

            # 智能拼接，保留最相关的部分
            if len(all_embeddings) > 1:
                # 计算每个embedding的重要性分数
                importance_scores = []
                for emb in all_embeddings:
                    # 使用L2范数作为重要性度量
                    score = torch.norm(emb, p=2, dim=(1, 2)).mean().item()
                    importance_scores.append(score)

                # 根据分数对embeddings进行排序
                sorted_pairs = sorted(enumerate(importance_scores), key=lambda x: x[1], reverse=True)
                sorted_indices = [idx for idx, _ in sorted_pairs]

                # 按重要性顺序重排embeddings
                sorted_embeddings = [all_embeddings[i] for i in sorted_indices]

                # 拼接并确保不超过最大长度
                encoder_hidden = torch.cat(sorted_embeddings, dim=1)
                if encoder_hidden.size(1) > 512:
                    encoder_hidden = encoder_hidden[:, :512, :]
            else:
                encoder_hidden = all_embeddings[0]

            # 对最终的hidden states再次归一化
            hidden_norm = torch.norm(encoder_hidden, p=2, dim=-1, keepdim=True)
            encoder_hidden = encoder_hidden / (hidden_norm + 1e-7)

            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)

            # 生成答案
            generated_ids = self.t5_model.generate(
                encoder_outputs=encoder_outputs,
                decoder_start_token_id=self.t5_model.config.decoder_start_token_id,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                eos_token_id=self.t5_model.config.eos_token_id,
                no_repeat_ngram_size=2,  # 避免重复生成
                length_penalty=1.0,  # 长度惩罚
                temperature=0.7  # 生成的多样性
            )

            return self.t5_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

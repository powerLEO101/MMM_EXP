import argparse
import pandas as pd
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from time import time
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)


# --- Dataset ---

def create_query_text(subject_name, construct_name, question_text, correct_answer, incorrect_answer) -> str:
    return f'Instruct: Given a math question and a misconcepte incorrect answer, please retrieve the most accurate reason for the misconception.\nQuery: \n### Question ###:\n{subject_name}, {construct_name}\n{question_text}\n### Correct Answer ###:\n{correct_answer}\n### Incorrect Answer ###:\n{incorrect_answer}'

class MyDataLoader:
    def __init__(self, train_df_path, misconceptions_path, batch_size, model_name, supplemental_batch_size=None, seed=42):
        train_df = pd.read_csv(train_df_path)
        misconceptions = pd.read_csv(misconceptions_path)
        train_df['correct_answer_text'] = train_df.apply(lambda x : x[f"Answer{x['CorrectAnswer']}Text"], axis=1)
        self.data = []
        for idx, row in train_df.iterrows():
            for choice in ['A', 'B', 'C', 'D']:
                if choice == row['CorrectAnswer']: continue
                if np.isnan(row[f'Misconception{choice}Id']): continue
                query_text = create_query_text(row['SubjectName'], row['ConstructName'], row['QuestionText'], row['correct_answer_text'], row[f'Answer{choice}Text'])
                query_target = misconceptions.iloc[int(row[f'Misconception{choice}Id'])]['MisconceptionName']
                self.data.append([idx, choice, query_text, query_target])
        self.data = pd.DataFrame(self.data, columns=['id', 'choice', 'text', 'target'])

        self.batch_size = batch_size
        self.supplemental_batch_size = int(self.batch_size / 2) if supplemental_batch_size is None else supplemental_batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.misconceptions = misconceptions
        self.random = np.random.RandomState(seed)

    def tokenize_everything(self, *args):
        # no max_legnth is given, only pad to the longest sequence in the batch. the longest input text should be around 512 tokens
        result = []
        for arg in args:
            result.append(self.tokenizer(arg, padding=True, return_tensors='pt'))
        return result

    def next_batch(self):
        batch_text, batch_mis = [], []
        for _ in range(self.batch_size):
            idx = self.random.randint(len(self.data))
            batch_text.append(self.data.iloc[idx]['text'])
            batch_mis.append(self.data.iloc[idx]['target'])
        
        # should not sample distractions in batch because some misconception never appear in training data
        supplemental_misconceptions = self.misconceptions.sample(self.supplemental_batch_size)['MisconceptionName'].values.tolist()
        batch_mis.extend(supplemental_misconceptions)
        batch_text, batch_mis = self.tokenize_everything(batch_text, batch_mis)
        
        return batch_text, batch_mis

# --- Model ---
class MyEmbeddingModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.embed_model = AutoModel.from_pretrained(model_name)
        config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
            lora_dropout=0.05,  # Conventional
            task_type="FEATURE_EXTRACTION", 
        )
        self.embed_model = get_peft_model(self.embed_model, config)
        self.embed_model.print_trainable_parameters()

    def last_token_pool(self, last_hidden_states: Tensor,
                        attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def encode(self, x):
        x1 = self.embed_model(input_ids=x['input_ids'], attention_mask=x['attention_mask'], return_dict=True)
        x = self.last_token_pool(x1.last_hidden_state, x['attention_mask'])
        return x.contiguous()

    def compute_similarity(self, a, b, eps=1e-8):
        # https://stackoverflow.com/questions/50411191/how-to-compute-the-cosine-similarity-in-pytorch-for-all-rows-in-a-matrix-with-re
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt

    def forward(self, batch_text, batch_mis=None):
        if not self.training:
            batch_text = self.encode(batch_text)
            return batch_text

        batch_text = self.encode(batch_text)
        batch_mis = self.encode(batch_mis)
        # sims = F.cosine_similarity(batch_text, batch_mis, dim=-1)
        sims = self.compute_similarity(batch_text, batch_mis) # batch_size, mis_size
        print(sims.shape)

        label = torch.arange(sims.shape[0], dtype=torch.long, device=sims.device)
        # technically the sims here are not logits, they cannot go lower than 0, but 
        # let's just follow the original paper TODO can be a improvement?
        loss = F.cross_entropy(sims, label)
        return loss

# --- Training Loop ---

def move_to_device(x, device):
    if isinstance(x, dict):
        return {k : v.to(device) for k, v in x.items()}
    elif isinstance(x, list):
        return [v.to(device) for v in x]
    else:
        return x.to(device)

def train_loop(model, dataloader, optimizer, total_steps):
    model.train()
    for step in range(total_steps):
        time_start = time()

        optimizer.zero_grad()
        batch_text, batch_mis = dataloader.next_batch()
        batch_text = move_to_device(batch_text)
        batch_mis = move_to_device(batch_mis)

        loss = model(batch_text, batch_mis)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        print(f'step:\t{step}/{total_steps} | loss:\t{loss.detach().item()} | grad_norm:\t{grad_norm} | time:\t{time() - time_start}')
    return model

# --- Main ---

def get_optimizer_grouped_parameters(
        model,
        weight_decay,
        lora_lr=5e-4,
        no_decay_name_list=["bias", "LayerNorm.weight"],
        lora_name_list=["lora_right_weight", "lora_left_weight"],
        ):
    optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if (not any(nd in n for nd in no_decay_name_list)
                        and p.requires_grad and not any(nd in n for nd in lora_name_list))
                        ],
                "weight_decay": weight_decay,
                },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if (not any(nd in n for nd in no_decay_name_list)
                        and p.requires_grad and any(nd in n for nd in lora_name_list))
                        ],
                "weight_decay": weight_decay,
                # "lr": lora_lr
                # we set learning rate later?
                },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if (any(nd in n for nd in no_decay_name_list) and p.requires_grad)
                    ],
                "weight_decay": 0.0,
                },
            ]
    if not optimizer_grouped_parameters[1]["params"]:
        optimizer_grouped_parameters.pop(1)
    return optimizer_grouped_parameters

def parse_args():
    parser = argparse.ArgumentParser( description='Script with base path and model name arguments')
    parser.add_argument('--base_path', type=str, help='Base path for the operation', default='/media/workspace/DATA_WAREHOUSE/MMM_INPUT')
    parser.add_argument('--model_name', type=str, help='Name of the model to use', default='Salesforce/SFR-Embedding-Mistral')
    parser.add_argument('--batch_size', type=int, help='Batch size', default=16)
    parser.add_argument('--lr', type=float, help='Batch size', default=3e-4)
    return parser.parse_args()

def main():
    model = MyEmbeddingModel(args.model_name)
    model = model.to(device)
    dataloader = MyDataLoader(train_df_path=f'{args.base_path}/train.csv',
                              misconceptions_path=f'{args.base_path}/misconception_mapping.csv',
                              batch_size=args.batch_size,
                              model_name=args.model_name,
                              )
    optim_groups = get_optimizer_grouped_parameters(model, 0.01)
    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.99), eps=1e-8, fused=True)
    train_loop(model, dataloader, optimizer, 10)

if __name__ == '__main__':
    device='cuda:0'
    args = parse_args()
    main()

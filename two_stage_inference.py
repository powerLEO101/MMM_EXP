# --- Setting ---

import os
import sys
import math
import argparse
import pandas as pd
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from time import time
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
import bitsandbytes as bnb
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import argparse
def parse_args():
    parser = argparse.ArgumentParser( description='Script with base path and model name arguments')
    parser.add_argument('--base_path', type=str, required=True)
    parser.add_argument('--adapter_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    return parser.parse_args()

# --- Stage 1 ---

import sys
sys.path.append('/media/workspace/MMM_EXP')
from hard_negative_mining_retriever import MyEmbeddingModel

model = MyEmbeddingModel(args.model_name, 1, False)
# model.embed_model.load_adapter(args.adapter_path)
model.eval()

def create_query_text(subject_name, construct_name, question_text, correct_answer, incorrect_answer) -> str:
    return f'Instruct: Given a math question and an incorrect answer, please retrieve the most accurate reason for the misconception.\nQuery: \n### Question ###:\n{subject_name}, {construct_name}\n{question_text}\n### Correct Answer ###:\n{correct_answer}\n### Incorrect Answer ###:\n{incorrect_answer}'

class MyDataLoader:
    def __init__(self, train_df_path, misconceptions_path, batch_size, model_name, rank, folds, supplemental_batch_size=None, seed=42):
        train_df = pd.read_csv(train_df_path) # can also be eval, but naming doesnt matter
        train_df['fold'] = train_df['QuestionId'].apply(lambda x : x % 5)
        train_df = train_df[train_df['fold'].isin(folds)]
        misconceptions = pd.read_csv(misconceptions_path)
        train_df['correct_answer_text'] = train_df.apply(lambda x : x[f"Answer{x['CorrectAnswer']}Text"], axis=1)
        self.data = []
        for idx, row in train_df.iterrows():
            for choice in ['A', 'B', 'C', 'D']:
                if choice == row['CorrectAnswer']: continue
                query_text = create_query_text(row['SubjectName'], row['ConstructName'], row['QuestionText'], row['correct_answer_text'], row[f'Answer{choice}Text'])
                self.data.append([idx, choice, query_text])
        self.data = pd.DataFrame(self.data, columns=['id', 'choice', 'text'])

        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.misconceptions = misconceptions
        self.rank = rank
        self.random = np.random.RandomState(seed)

    def set_hard_examples(self, hard_examples):
        self.hard_examples = hard_examples

    def tokenize_everything(self, *args):
        # no max_legnth is given, only pad to the longest sequence in the batch. the longest input text should be around 512 tokens
        result = []
        for arg in args:
            if len(arg) == 0:
                result.append(torch.tensor([], dtype=torch.long))
                continue
            result.append(self.tokenizer(arg, padding=True, return_tensors='pt'))
        return result

    def next_batch(self):
        assert hasattr(self, 'hard_examples'), 'Run set hard examples before next batch'
        batch_text, batch_mis, batch_index = [], [], []
        for _ in range(self.batch_size):
            idx = self.random.randint(len(self.data))
            batch_text.append(self.data.iloc[idx]['text'])
            batch_mis.append(self.data.iloc[idx]['target'])
            batch_index.append(idx)
        
        # handle ddp
        actual_batch_size = int(self.batch_size / ddp_world_size)
        batch_text = batch_text[self.rank * actual_batch_size : (self.rank + 1) * actual_batch_size]
        batch_mis = batch_mis[self.rank * actual_batch_size : (self.rank + 1) * actual_batch_size]
        batch_index = batch_index[self.rank * actual_batch_size : (self.rank + 1) * actual_batch_size]

        # should not sample distractions in batch because some misconception never appear in training data
        supplemental_misconceptions = []
        for one_index in batch_index:
            # use randint 1e6 here to keep the random seed the same
            hard_example_indices = self.random.randint(0, 1e6, self.supplemental_batch_size) % len(self.hard_examples[one_index])
            hard_example_indices = [self.hard_examples[one_index][x] for x in hard_example_indices]
            supplemental_misconceptions.extend(self.misconceptions.iloc[hard_example_indices]['MisconceptionName'].values.tolist())
        batch_mis.extend(supplemental_misconceptions)
        batch_text, batch_mis = self.tokenize_everything(batch_text, batch_mis)
        
        return batch_text, batch_mis
    
    def pad_list_to_batch_size(self, x, fill):
        assert isinstance(x, list)
        actual_batch_size = int(self.batch_size / ddp_world_size)
        if len(x) < actual_batch_size:
            x = x + [fill] * (actual_batch_size - len(x))
        return x

    def all_text(self):
        all_text = self.data['text'].values.tolist()
        # all_target = self.data['target_id'].values.tolist()
        for i in range(0, len(all_text), self.batch_size):
            batch_text = all_text[i : min(len(all_text), i + self.batch_size)]
            # batch_target = all_target[i : min(len(all_text), i + self.batch_size)]

            actual_batch_size = int(self.batch_size / ddp_world_size)
            batch_text = self.pad_list_to_batch_size(batch_text[self.rank * actual_batch_size : (self.rank + 1) * actual_batch_size], '')
            # batch_target = self.pad_list_to_batch_size(batch_target[self.rank * actual_batch_size : (self.rank + 1) * actual_batch_size], 0)

            batch_text = self.tokenize_everything(batch_text)[0]
            # batch_target = torch.tensor(batch_target, dtype=torch.long)
            yield batch_text

    def all_misconceptions(self):
        all_mis = self.misconceptions['MisconceptionName'].values.tolist()
        for i in range(0, len(all_mis), self.batch_size):
            batch_mis = all_mis[i : min(len(all_mis), i + self.batch_size)]
            actual_batch_size = int(self.batch_size / ddp_world_size)
            batch_mis = self.pad_list_to_batch_size(batch_mis[self.rank * actual_batch_size : (self.rank + 1) * actual_batch_size], '')
            batch_mis = self.tokenize_everything(batch_mis)[0]
            yield batch_mis

# --- 

def get_all_embeddings(model, dataloader):
    if master_process: print('--- Get all embeddings ---')
    time_start = time()
    text_embeddingst = []
    for batch_text, batch_label in dataloader.all_text():
        batch_text = move_to_device(batch_text, device)
        text_embedding = model(batch_text)

        text_embedding = ddp_sync_concat_tensor(text_embedding).cpu()
        batch_label = ddp_sync_concat_tensor(batch_label).cpu()
        text_embeddings.append(text_embedding)
    text_embeddings = torch.cat(text_embeddings, dim=0)
    if master_process: print(f'Embedding text took {time() - time_start} s')
    time_start = time()

    mis_embeddings = []
    for batch_mis in dataloader.all_misconceptions():
        batch_mis = move_to_device(batch_mis, device)
        mis_embedding = model(batch_mis)
        mis_embedding = ddp_sync_concat_tensor(mis_embedding).cpu()
        mis_embeddings.append(mis_embedding)
    mis_embeddings = torch.cat(mis_embeddings, dim=0)
    if master_process: print(f'Embedding misconceptions took {time() - time_start} s')

    text_embeddings = text_embeddings[ : len(dataloader.data)]
    all_targets = all_targets[ : len(dataloader.data)]
    mis_embeddings = mis_embeddings[ : len(dataloader.misconceptions)]

# --- Stage 2 ---
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(model="meta-llama/Llama-2-7b-hf", enable_lora=True)

# --- End ---

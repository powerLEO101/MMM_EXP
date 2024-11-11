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
import metrics
import utils
import visualize

class MyEmbeddingModel(nn.Module):
    def __init__(self, model_name, temperature, init_lora=True):
        super().__init__()
        bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
                )
        self.embed_model = AutoModel.from_pretrained(model_name, quantization_config=bnb_config)
        if init_lora:
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_r * 1.4,
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
        # self.embed_model.gradient_checkpointing_enable()

        self.temperature = temperature

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

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def forward(self, batch_text, batch_mis=None):
        if not self.training:
            batch_text = F.normalize(self.encode(batch_text), p=2, dim=-1)
            return batch_text

        if master_process and args.visualize != 0:
            visualize.visualize(batch_text['input_ids'].cpu())
            visualize.visualize(batch_mis['input_ids'].cpu())
        batch_text = F.normalize(self.encode(batch_text), p=2, dim=-1)
        batch_mis = F.normalize(self.encode(batch_mis), p=2, dim=-1)
        if ddp:
            batch_text = ddp_sync_concat_tensor(batch_text)
            batch_mis = ddp_sync_concat_tensor(batch_mis)
        # sims = F.cosine_similarity(batch_text, batch_mis, dim=-1)
        sims = self.compute_similarity(batch_text, batch_mis) # batch_size, mis_size
        sims = sims / self.temperature # to increase the difference in probability, sims is capped at (0, 1)

        if ddp:
            mis_batch_size = int(batch_mis.shape[0] / ddp_world_size)
            actual_batch_size = int(batch_text.shape[0] / ddp_world_size)
            label = [list(range(mis_batch_size * x, mis_batch_size * x + actual_batch_size)) for x in range(ddp_world_size)]
            label = torch.tensor(label, dtype=torch.long, device=sims.device).flatten()
        else:
            label = torch.arange(sims.shape[0], dtype=torch.long, device=sims.device)
        # technically the sims here are not logits, they cannot go lower than 0, but 
        # let's just follow the original paper TODO can be a improvement?
        loss = F.cross_entropy(sims, label)
        return loss

    def save_pretrained(self, path):
        self.embed_model.save_pretrained(path)

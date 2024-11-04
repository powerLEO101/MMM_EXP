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

ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# --- Dataset ---

def create_query_text(subject_name, construct_name, question_text, correct_answer, incorrect_answer) -> str:
    return f'Instruct: Given a math question and a misconcepte incorrect answer, please retrieve the most accurate reason for the misconception.\nQuery: \n### Question ###:\n{subject_name}, {construct_name}\n{question_text}\n### Correct Answer ###:\n{correct_answer}\n### Incorrect Answer ###:\n{incorrect_answer}'

class MyDataLoader:
    def __init__(self, train_df_path, misconceptions_path, batch_size, model_name, rank, supplemental_batch_size=None, seed=42):
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
                self.data.append([idx, choice, query_text, query_target, int(row[f'Misconception{choice}Id'])])
        self.data = pd.DataFrame(self.data, columns=['id', 'choice', 'text', 'target', 'target_id'])

        self.batch_size = batch_size
        self.supplemental_batch_size = int(self.batch_size / 2) if supplemental_batch_size is None else supplemental_batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.misconceptions = misconceptions
        self.rank = rank
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
        
        # handle ddp
        actual_batch_size = int(self.batch_size / ddp_world_size)
        batch_text = batch_text[self.rank * actual_batch_size : (self.rank + 1) * actual_batch_size]
        batch_mis = batch_mis[self.rank * actual_batch_size : (self.rank + 1) * actual_batch_size]

        # should not sample distractions in batch because some misconception never appear in training data
        supplemental_misconceptions = self.misconceptions.sample(self.supplemental_batch_size)['MisconceptionName'].values.tolist()
        batch_mis.extend(supplemental_misconceptions)
        batch_text, batch_mis = self.tokenize_everything(batch_text, batch_mis)
        
        return batch_text, batch_mis
    
    def all_text(self):
        all_text = self.data['text'].values.tolist()
        all_target = self.data['target_id'].values.tolist()
        for i in range(0, len(batch_text), self.batch_size):
            batch_text = all_text[i : min(len(batch_text), i + self.batch_size)]
            batch_target = all_target[i : min(len(batch_text), i + self.batch_size)]

            actual_batch_size = int(self.batch_size / ddp_world_size)
            batch_text = batch_text[self.rank * actual_batch_size : (self.rank + 1) * actual_batch_size]
            batch_target = batch_target[self.rank * actual_batch_size : (self.rank + 1) * actual_batch_size]

            batch_text = self.tokenize_everything(batch_text)[0]
            batch_target = torch.tensor(batch_target, dtype=torch.long)
            yield batch_text, batch_target

    def all_misconceptions(self):
        batch_mis = self.misconceptions['MisconceptionName'].values.tolist()
        for i in range(0, len(batch_mis), self.batch_size):
            batch_mis = all_text[i : min(len(batch_mis), i + self.batch_size)]
            actual_batch_size = int(self.batch_size / ddp_world_size)
            batch_mis = batch_mis[self.rank * actual_batch_size : (self.rank + 1) * actual_batch_size]
            batch_mis = self.tokenize_everything(batch_mis)[0]
            yield batch_mis


# --- Model ---
class MyEmbeddingModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
                )
        self.embed_model = AutoModel.from_pretrained(model_name, quantization_config=bnb_config)
        config = LoraConfig(
            r=32,
            lora_alpha=64,
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

        label = torch.arange(sims.shape[0], dtype=torch.long, device=sims.device)
        # technically the sims here are not logits, they cannot go lower than 0, but 
        # let's just follow the original paper TODO can be a improvement?
        loss = F.cross_entropy(sims, label)
        return loss

    def save_pretrained(self, path):
        self.embed_model.save_pretrained(path)

# --- Training Loop ---

def get_lr(it):
    # https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_steps:
        return args.max_lr * (it+1) / args.warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > args.total_step:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - args.warmup_steps) / (args.total_step - args.warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return args.min_lr + coeff * (args.max_lr - args.min_lr)

def move_to_device(x, device):
    if isinstance(x, dict):
        return {k : v.to(device) for k, v in x.items()}
    elif isinstance(x, list):
        return [v.to(device) for v in x]
    else:
        return x.to(device)

def ddp_sync_concat_tensor(x, dim=0):
    tensor_list = [torch.zeros_like(x, device=device) for _ in range(ddp_world_size)]
    dist.all_gather(tensor_list, x)
    tensor_list = torch.cat(tensor_list, dim=dim)
    return tensor_list

@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    if master_process:
        print('--- Evaluate ---')
        print('--- Embedding text ---')
        
    text_embeddings, all_targets = [], []
    for batch_text, batch_label in dataloder.all_text():
        move_to_device(batch_text, device)
        text_embedding = model(batch_text)
        text_embedding = ddp_sync_concat_tensor(text_embedding).cpu()
        batch_label = ddp_sync_concat_tensor(batch_label)
        all_targets.append(batch_label)
        text_embeddings.append(text_embedding)
    text_embeddings = torch.cat(text_embeddings, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    if master_process:
        print('--- Embedding misconceptions ---')
    mis_embeddings = []
    for batch_mis in dataloader.all_misconceptions():
        move_to_device(batch_mis, device)
        mis_embedding = model(batch_mis)
        mis_embedding = ddp_sync_concat_tensor(mis_embedding).cpu()
        mis_embeddings.append(mis_embedding)
    mis_embeddings = torch.cat(mis_embeddings, dim=0)
    
    scores = model.compute_similarity(text_embeddings, mis_embeddings) # all_text, all_mis
    top_scores = torch.argsort(scores, dim=-1, descending=True) # all_text, all_mis in id
    top25_ids = top_scores[:, : 25]

    map25_score = metrics.mapk(actual=[[x] for x in all_targets.tolist()],
                               predicted=top25_ids.tolist())
    top25_hitrate = sum([(top_scores[i] == all_targets[i]).any() for i in range(len(all_targets))]) / len(all_targets)
    print(f'map@25:\t{map25_score : .3f} | top@25 hitrate:\t{top25_hitrate : .3f}')


class MyLogger:
    def __init__(self, cumulative, average, literal, log_step_total, log_interval=5):
        '''
        cumulative: names for the logging variables that are cumulative, such as time
        average: names for the logging variables that are average, such as loss
        '''
        self.cumulative = cumulative
        self.average = average
        self.literal = literal
        self.data = {x : [] for x in cumulative + average + literal}
        self.log_step = 0
        self.log_step_total = log_step_total
        self.log_interval = log_interval

    def log(self, **arg_dict):
        for k, v in arg_dict.items():
            self.data[k].append(v)
        self.log_step += 1
        if (self.log_step + 1) % self.log_interval == 0:
            self.print_log()

    def print_log(self):
        text = []
        for c in self.cumulative:
            text.append(f'{c}:\t{sum(self.data[c][-self.log_interval : ]) : .3f}')
        for a in self.average:
            text.append(f'{a}:\t{sum(self.data[a][-self.log_interval : ]) / self.log_interval : .3f}')
        for l in self.literal:
            text.append(f'{l}:\t{self.data[l][-1]}')
        text = f'{self.log_step}/{self.log_step_total} || ' + ' | '.join(text)
        print(text)

    def to_csv(self, path):
        pd.DataFrame.from_dict(self.data).to_csv(path)


def train_loop(model, dataloader, optimizer, total_steps):
    logger = MyLogger(cumulative=['time'],
                      average=['lr', 'loss_accum', 'grad_norm'],
                      literal=['step'],
                      log_interval=args.log_interval,
                      log_step_total=total_steps)
    for step in range(total_steps):
        time_start = time()

        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(args.grad_accum):
            batch_text, batch_mis = dataloader.next_batch()
            batch_text = move_to_device(batch_text, device)
            batch_mis = move_to_device(batch_mis, device)

            if ddp: # actually, I don't understand why can't I just do a full backward at the end
                    # but lets just follow the nano gpt 
                model.require_backward_grad_sync = (micro_step == args.grad_accum - 1)
            loss = model(batch_text, batch_mis) / args.grad_accum
            loss.backward()
            loss_accum += loss.detach().cpu().item()

        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0))
        # gradient is synced here, so no need to all reduce
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            time_elapsed = time() - time_start
            logger.log(step=step, lr=lr, loss_accum=loss_accum, grad_norm=grad_norm, time=time_elapsed)
            if (step + 1) % args.ckpt_interval == 0:
                model.save_pretrained(f'{save_path}/step{step : 05d}_checkpoint')
    return model, logger

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
                # set learning rate later
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
    parser.add_argument('--batch_size', type=int, help='Batch size', default=4)
    parser.add_argument('--lr', type=float, help='Learning rate is not used in this code', default=3e-4)
    parser.add_argument('--max_lr', type=float, help='Max learning rate', default=5e-4)
    parser.add_argument('--min_lr', type=float, help='Min learning rate', default=5e-5)
    parser.add_argument('--grad_accum', type=int, help='Gradient accumulation', default=16)
    parser.add_argument('--warmup_steps', type=int, help='Warmup steps', default=10)
    parser.add_argument('--total_step', type=int, help='Total step', default=100)
    parser.add_argument('--ckpt_interval', type=int, help='Ckpt interval', default=40)
    parser.add_argument('--log_interval', type=int, help='Ckpt interval', default=1)
    parser.add_argument('--exp_id', type=str, required=True, help='Experiment id')
    return parser.parse_args()

def print_args():
    print('------ Printing args ------')
    for k, v in args.__dict__.items():
        print(k,':', v)
    print('Actual batch size', ':', args.batch_size * args.grad_accum)
    import csv
    data = args.__dict__
    with open(f"{save_path}/args.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data.keys())
        writer.writerow(data.values())
    print(f'------ Args are saved to {save_path}/args.csv ------')

def main():
    model = MyEmbeddingModel(args.model_name)
    model = model.to(device)
    assert args.batch_size % ddp_world_size == 0
    dataloader = MyDataLoader(train_df_path=f'{args.base_path}/train.csv',
                              misconceptions_path=f'{args.base_path}/misconception_mapping.csv',
                              batch_size=args.batch_size,
                              model_name=args.model_name,
                              supplemental_batch_size=int(16 - (args.batch_size / ddp_world_size)),
                              rank=ddp_rank,
                              seed=42,
                              )
    optim_groups = get_optimizer_grouped_parameters(model, 0.01)
    optimizer = bnb.optim.Adam8bit(optim_groups, lr=args.lr, betas=(0.9, 0.99), eps=1e-8)
    model, logger = train_loop(model, dataloader, optimizer, args.total_step)
    if master_process:
        print(f'------ Experiment finished, saving checkpoint to {save_path} ------')
        model.save_pretrained(f'{save_path}/final_checkpoint')
        logger.to_csv(f'{save_path}/df_log.csv')

if __name__ == '__main__':
    args = parse_args()
    save_path = f'/media/workspace/MMM_SAVE/{args.exp_id}'
    if master_process:
        print(f'------ Executing experiment {args.exp_id} ------')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print(f'------ New experiment, creating save_path {save_path} ------')
        else:
            import click
            if click.confirm('Save path has already been created, abort?', default=False):
                if ddp:
                    destroy_process_group()
                sys.exit(0)
            print(f'------ Rerunning existing experiment, overwriting {save_path} ------')
        print_args()
    dist.barrier()
    main()

if ddp:
    destroy_process_group()

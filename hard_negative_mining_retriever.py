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
    return f'Instruct: Given a math question and an incorrect answer, please retrieve the most accurate misconception for the incorrect answer.\nQuery: \n### Question ###:\n{subject_name}, {construct_name}\n{question_text}\n### Correct Answer ###:\n{correct_answer}\n### Incorrect Answer ###:\n{incorrect_answer}'

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
        all_target = self.data['target_id'].values.tolist()
        for i in range(0, len(all_text), self.batch_size):
            batch_text = all_text[i : min(len(all_text), i + self.batch_size)]
            batch_target = all_target[i : min(len(all_text), i + self.batch_size)]

            actual_batch_size = int(self.batch_size / ddp_world_size)
            batch_text = self.pad_list_to_batch_size(batch_text[self.rank * actual_batch_size : (self.rank + 1) * actual_batch_size], '')
            batch_target = self.pad_list_to_batch_size(batch_target[self.rank * actual_batch_size : (self.rank + 1) * actual_batch_size], 0)

            batch_text = self.tokenize_everything(batch_text)[0]
            batch_target = torch.tensor(batch_target, dtype=torch.long)
            yield batch_text, batch_target

    def all_misconceptions(self):
        all_mis = self.misconceptions['MisconceptionName'].values.tolist()
        for i in range(0, len(all_mis), self.batch_size):
            batch_mis = all_mis[i : min(len(all_mis), i + self.batch_size)]
            actual_batch_size = int(self.batch_size / ddp_world_size)
            batch_mis = self.pad_list_to_batch_size(batch_mis[self.rank * actual_batch_size : (self.rank + 1) * actual_batch_size], '')
            batch_mis = self.tokenize_everything(batch_mis)[0]
            yield batch_mis


# --- Model ---

from embedding_model import MyEmbeddingModel

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
    tensor_list[ddp_rank] = x # !! replace the current rank tensor with the given one so it has grad
    tensor_list = torch.cat(tensor_list, dim=dim)
    return tensor_list

@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    if master_process:
        print('--- Evaluate ---')
        
    time_start = time()
    text_embeddings, all_targets = [], []
    for batch_text, batch_label in dataloader.all_text():
        batch_text = move_to_device(batch_text, device)
        batch_label = move_to_device(batch_label, device) # even though we don't actually need it on gpu
                                                          # we need to move it because we want to all gather
        text_embedding = model(batch_text)

        text_embedding = ddp_sync_concat_tensor(text_embedding).cpu()
        batch_label = ddp_sync_concat_tensor(batch_label).cpu()
        text_embeddings.append(text_embedding)
        all_targets.append(batch_label)

    text_embeddings = torch.cat(text_embeddings, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

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

    # we padded the input, not truncate the unwanted part
    text_embeddings = text_embeddings[ : len(dataloader.data)]
    all_targets = all_targets[ : len(dataloader.data)]
    mis_embeddings = mis_embeddings[ : len(dataloader.misconceptions)]
    scores = model.compute_similarity(text_embeddings, mis_embeddings) # all_text, all_mis
    top_scores = torch.argsort(scores, dim=-1, descending=True) # all_text, all_mis in id
    top25_ids = top_scores[:, : 25]

    map25_score = metrics.mapk(actual=[[x] for x in all_targets.tolist()],
                               predicted=top25_ids.tolist())
    top25_hitrate = sum([(top25_ids[i] == all_targets[i]).any() for i in range(len(all_targets))]) / len(all_targets)
    model.train()

    if master_process: print(f'map@25:\t{map25_score : .3f} | top@25 hitrate:\t{top25_hitrate : .3f}')
    return float(map25_score), float(top25_hitrate)

@torch.no_grad()
def get_hard_negative_samples(model, dataloader):
    if args.hard_example_path is not None:
        if master_process: print(f'--- Using hard examples from {args.hard_example_path} ---')
        result = torch.load(args.hard_example_path, weights_only=False)
        args.hard_example_path = None
        return result
    original_batch_size = dataloader.batch_size
    dataloader.batch_size *= 4 # we can temporarily increase train dataloader's batch size because of no grad
    model.eval()

    if master_process: print('--- Getting hard negatvie samples ---')
    time_start = time()
    text_embeddings, all_targets = [], []
    for batch_text, batch_label in dataloader.all_text():
        batch_text = move_to_device(batch_text, device)
        batch_label = move_to_device(batch_label, device) # even though we don't actually need it on gpu
                                                          # we need to move it because we want to all gather
        text_embedding = model(batch_text)

        text_embedding = ddp_sync_concat_tensor(text_embedding).cpu()
        batch_label = ddp_sync_concat_tensor(batch_label).cpu()
        text_embeddings.append(text_embedding)
        all_targets.append(batch_label)
    text_embeddings = torch.cat(text_embeddings, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
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
    scores = model.compute_similarity(text_embeddings, mis_embeddings) # all_text, all_mis
    top_scores = torch.argsort(scores, dim=-1, descending=True) # all_text, all_mis in id
    target_indices = [(top_scores[i] == all_targets[i]).nonzero()[0][0] for i in range(len(top_scores))] # target has to be in top scores
    hard_examples = [top_scores[i, : max(32, x)].tolist() for i, x in enumerate(target_indices)]
    for i in range(len(hard_examples)):
        if target_indices[i] < len(hard_examples[i]):
            del hard_examples[i][target_indices[i]]

    model.train()
    dataloader.batch_size = original_batch_size
    return hard_examples

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
        if (self.log_step + 1) % self.log_interval == 0:
            self.print_log()
        self.log_step += 1

    def print_log(self):
        text = []
        for c in self.cumulative:
            text.append(f'{c}:\t{sum(self.data[c][-self.log_interval : ]) : .3e}')
        for a in self.average:
            text.append(f'{a}:\t{sum(self.data[a][-self.log_interval : ]) / self.log_interval : .3e}')
        for l in self.literal:
            text.append(f'{l}:\t{self.data[l][-1]}')
        text = f'{self.log_step}/{self.log_step_total} || ' + ' | '.join(text)
        print(text)

    def to_csv(self, path):
        pd.DataFrame.from_dict(self.data).to_csv(path)


def train_loop(model, dataloader, eval_dataloader, optimizer, total_steps):
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model
    logger = MyLogger(cumulative=['time'],
                      average=['lr', 'loss_accum', 'grad_norm'],
                      literal=['step'],
                      log_interval=args.log_interval,
                      log_step_total=total_steps)
    eval_logger = MyLogger([], [], ['step', 'map25_score', 'top25_hitrate'], 
                           log_interval=100000,
                           log_step_total=100000)
    for step in range(total_steps):
        if step % args.reset_hard_examples_interval == 0:
            hard_examples = get_hard_negative_samples(raw_model, dataloader)
            torch.save(hard_examples, f'{save_path}/hard_examples_{step}.pt')
            dataloader.set_hard_examples(hard_examples)

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
            loss_accum += loss.detach()

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
            logger.log(step=step, lr=lr, loss_accum=loss_accum.cpu().item(), grad_norm=grad_norm, time=time_elapsed)
            if (step + 1) % args.ckpt_interval == 0:
                raw_model.save_pretrained(f'{save_path}/step{step : 05d}_checkpoint')

        if (step + 1) % args.eval_interval == 0:
            map25_score, top25_hitrate = evaluate(raw_model, eval_dataloader)
            eval_logger.log(map25_score=map25_score, top25_hitrate=top25_hitrate, step=step)

    return raw_model, logger, eval_logger

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
    parser.add_argument('--save_path', type=str, help='Save path', default='/media/workspace/MMM_SAVE')
    parser.add_argument('--model_name', type=str, help='Name of the model to use', default='Salesforce/SFR-Embedding-Mistral')
    parser.add_argument('--batch_size', type=int, help='Batch size', default=4)
    parser.add_argument('--lr', type=float, help='Learning rate is not used in this code', default=3e-4)
    parser.add_argument('--max_lr', type=float, help='Max learning rate', default=1e-4)
    parser.add_argument('--min_lr', type=float, help='Min learning rate', default=1e-5)
    parser.add_argument('--temperature', type=float, help='Softmax temperature', default=0.05)
    parser.add_argument('--grad_accum', type=int, help='Gradient accumulation', default=16)
    parser.add_argument('--warmup_steps', type=int, help='Warmup steps', default=10)
    parser.add_argument('--total_step', type=int, help='Total step', default=100)
    parser.add_argument('--supplemental_batch_size', type=int, help='supplemental_batch_size', default=3)
    parser.add_argument('--ckpt_interval', type=int, help='Ckpt interval', default=40)
    parser.add_argument('--log_interval', type=int, help='Log interval', default=1)
    parser.add_argument('--eval_interval', type=int, help='Eval interval', default=100)
    parser.add_argument('--reset_hard_examples_interval', type=int, help='Reset hard examples', default=200)
    parser.add_argument('--hard_example_path', type=str, help='Hard example path', default=None)
    parser.add_argument('--lora_r', type=int, help='Lora rank', default=32)
    parser.add_argument('--visualize', type=int, help='visualze', default=0)
    parser.add_argument('--exp_id', type=str, required=True, help='Experiment id')
    return parser.parse_args()

def print_args():
    print('------ Printing args ------')
    for k, v in args.__dict__.items():
        print(k,':', v)
    print('Actual batch size', ':', args.batch_size * args.grad_accum)
    print('Batch size per device', ':', args.batch_size / ddp_world_size)
    import csv
    data = args.__dict__
    with open(f"{save_path}/args.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data.keys())
        writer.writerow(data.values())
    print(f'------ Args are saved to {save_path}/args.csv ------')

def main():
    if args.visualize: visualize.set_tokenizer(args.model_name)
    model = MyEmbeddingModel(args.model_name, args.temperature)
    model = model.to(device)
    assert args.batch_size % ddp_world_size == 0
    dataloader = MyDataLoader(train_df_path=f'{args.base_path}/train.csv',
                              misconceptions_path=f'{args.base_path}/misconception_mapping.csv',
                              batch_size=args.batch_size,
                              model_name=args.model_name,
                              supplemental_batch_size=args.supplemental_batch_size,
                              rank=ddp_rank,
                              seed=42,
                              folds=[0, 1, 2, 3],
                              )
    eval_dataloader = MyDataLoader(train_df_path=f'{args.base_path}/train.csv',
                                   misconceptions_path=f'{args.base_path}/misconception_mapping.csv',
                                   batch_size=args.batch_size * 4,
                                   model_name=args.model_name,
                                   supplemental_batch_size=args.supplemental_batch_size,
                                   rank=ddp_rank,
                                   seed=42,
                                   folds=[4],
                                   )
    optim_groups = get_optimizer_grouped_parameters(model, 0.01)
    optimizer = bnb.optim.Adam8bit(optim_groups, lr=args.lr, betas=(0.9, 0.99), eps=1e-8)
    model, logger, eval_logger = train_loop(model, dataloader, eval_dataloader, optimizer, args.total_step)
    if master_process:
        print(f'------ Experiment finished, saving checkpoint to {save_path} ------')
        model.save_pretrained(f'{save_path}/final_checkpoint')
        logger.to_csv(f'{save_path}/df_log.csv')
        eval_logger.to_csv(f'{save_path}/df_log_eval.csv')

if __name__ == '__main__':
    args = parse_args()
    save_path = f'{args.save_path}/{args.exp_id}'
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

# --- Stage 2 ---
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
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def main():
    top25_ids = torch.load('top25_ids.pt', weights_only=False)
    data = torch.load('data.pt', weights_only=False)
    misconceptions = torch.load('misconceptions.pt', weights_only=False)

    def create_query_text(subject_name, construct_name, question_text, mis_name, correct_answer, incorrect_answer) -> str:
        prompt = (
            '### Instruction:\n'
            'You are a math tutor for novice math learners. You are presented with the following math problem in the field of {}, {}. You are given a correct answer and an incorrect answer.\n'
            'Is the incorrect answer incorrect because of the misconception of {}? Answer only with Yes or No.\n'
            'Question: {}\n'
            'Correct answer: {}\n'
            'Incorrect answer: {}\n'
            '### Answer:\n'
        )
        return prompt.format(subject_name, construct_name, mis_name, question_text, correct_answer, incorrect_answer)

    prompts = []
    for idx, row in data.iterrows():
        for i in top25_ids[idx].tolist():
            text = create_query_text(row['subject_name'], row['construct_name'], row['question_text'], misconceptions.iloc[i]['MisconceptionName'], row['correct_answer'], row['incorrect_answer'])
            prompts.append(text)

    llm = vllm.LLM(
        "Qwen/Qwen2.5-32B-Instruct-AWQ",
        quantization="awq",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.90, 
        trust_remote_code=True,
        dtype="half", 
        enforce_eager=True,
        max_model_len=5120,
        disable_log_stats=True,
        enable_lora=True
    )

    sampling_params = SamplingParams(max_tokens=1, logprobs=20)
    result = llm.generate(prompts, sampling_params)
    rerank_score = [x.outputs[0].logprobs[0][9454].logprob if 9454 in x.outputs[0].logprobs[0] else -100 for x in result]

    def rerank_submission(ids, scores):
        reranks = np.argsort(scores)[::-1]
        new_ids = [ids[x] for x in reranks]
        return new_ids

    rerank_top25_ids = [rerank_submission(top25_ids[i].tolist(), rerank_score[i * 25 : (i + 1) * 25]) for i in range(len(top25_ids))]

    submission = []
    for idx, row in data.iterrows():
        real_id = f"{row['id']}_{row['choice']}"
        submission.append([real_id, ' '.join([str(x) for x in rerank_top25_ids[idx]])])
    submission = pd.DataFrame(submission, columns=['QuestionId_Answer', 'MisconceptionId'])
    submission.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    main()
# --- End ---

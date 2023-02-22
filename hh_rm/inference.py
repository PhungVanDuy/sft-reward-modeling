from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
import argparse
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model, PreTrainedModel,AutoModelForCausalLM, GPT2PreTrainedModel, GPT2Model, AutoTokenizer
from transformers.modeling_outputs import ModelOutput
from torch import nn
from torch.nn import Identity
import torch.nn.functional as F
import torch
from dataclasses import dataclass
from typing import Optional, Tuple


class RewardModel(nn.Module):
    def __init__(self, config, PAD_ID, save_model=True):
        super().__init__()
        # TODO(dahoas): fix hacky fix
        model = AutoModelForCausalLM.from_pretrained(config)
        self.config = model.config
        self.neox = "neox" in self.config.model_type
        # gpt-neo models have hidden_size instead of n_embd
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.transformer = model.gpt_neox if hasattr(model, "gpt_neox") else model.transformer
        dtype = self.config.torch_dtype if hasattr(self.config, "torch_dtype") is not None else torch.float32
        dtype = torch.float16 if dtype == "float16" else torch.float32
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False, dtype=torch.float16)
        self.PAD_ID = PAD_ID
        if save_model:
            self.model = model

    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing_enable()


    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        lm_labels=None,
        mc_labels=None,
        return_dict=False,
        output_attentions=False,
        output_hidden_states=False,
    ):
        loss=None
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        ) if self.neox else self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs[0]

        rewards = self.v_head(hidden_states).squeeze(-1)
        ends = torch.argmax((input_ids == self.PAD_ID).type(torch.float32), dim=1).view(-1, 1)
        rewards = torch.gather(rewards, 1, ends)
        return rewards

def make_rm(model_name, tok_path, save_model=True):
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    reward_model = RewardModel(model_name, tokenizer(tokenizer.eos_token)["input_ids"][0], save_model=save_model)
    return reward_model

def load_rm(model_name, tokenizer_name, model_path, save_model):
    rm = make_rm(model_name, tokenizer_name, save_model)
    rm.load_state_dict(torch.load(model_path), strict=True)
    return rm

REWARD_CHECKPOINT_PATH = "/fsx/home-duyphung/sandbox/refactor_summarize_rlhf_31Dec/hf_ckpt.pt"
if not os.path.exists(REWARD_CHECKPOINT_PATH):
    os.makedirs("reward_model/rm_checkpoint", exist_ok=True)
    os.system(
        f"wget -O {REWARD_CHECKPOINT_PATH} \
        https://huggingface.co/CarperAI/openai_summarize_tldr_rm_checkpoint/resolve/main/pytorch_model.bin"
    )

model = load_rm("EleutherAI/gpt-j-6B", "EleutherAI/gpt-j-6B", REWARD_CHECKPOINT_PATH, True)
model.to('cuda')
model.eval()
model.half()

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("loading dataset ...")
from datasets import load_dataset
dataset = load_dataset("Dahoas/rm-static", split="test").select(range(100))

chosen_prompts = [
    x['prompt'] + "\n" + x['chosen'] for x in dataset   
]
rejection_prompts = [
    x['prompt'] + "\n" + x['rejected'] for x in dataset
]

choosen_rewards = []
rejection_rewards = []

batch_size = 2
from tqdm import tqdm
import numpy as np

for i in tqdm(range(0, len(chosen_prompts), batch_size)):
    inputs = tokenizer(chosen_prompts[i:i+batch_size], return_tensors='pt',
                       truncation=True, max_length=512, padding="max_length").to('cuda')
    with torch.no_grad():
        out = model(**inputs)
        choosen_rewards.append(out.detach().cpu().numpy())

choosen_rewards = np.concatenate(choosen_rewards)

for i in tqdm(range(0, len(rejection_prompts), batch_size)):
    inputs = tokenizer(rejection_prompts[i:i+batch_size], return_tensors='pt',
                       truncation=True, max_length=512, padding="max_length").to('cuda')
    with torch.no_grad():
        out = model(**inputs)
        rejection_rewards.append(out.detach().cpu().numpy())
rejection_rewards = np.concatenate(rejection_rewards)

import ipdb; ipdb.set_trace()

print("Acceptance rate: ", np.mean(choosen_rewards > rejection_rewards))



import json
import torch
import os
from torch import nn
from torch.nn import Identity
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def make_rm(model_name, tok_path):
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    reward_model = RewardModel(model_name, tokenizer(tokenizer.eos_token)["input_ids"][0])
    return reward_model

def load_rm(model_name, tokenizer_name, model_path):
    rm = make_rm(model_name, tokenizer_name)
    rm.load_state_dict(torch.load(model_path), strict=True)
    return rm

REWARD_CHECKPOINT_PATH = "hf_ckpt.pt"
if not os.path.exists(REWARD_CHECKPOINT_PATH):
    os.makedirs("reward_model/rm_checkpoint", exist_ok=True)
    os.system(
        f"wget -O {REWARD_CHECKPOINT_PATH} \
        https://huggingface.co/Dahoas/gptj-rm-static/resolve/main/hf_ckpt.pt"
    )

model = load_rm("EleutherAI/gpt-j-6B", "EleutherAI/gpt-j-6B", REWARD_CHECKPOINT_PATH)
model.to('cuda')
model.eval()
model.half()

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

examples = ['\n\nHuman: Do you know why turkeys became the official food of thanksgiving?\n\n\nAssistant: To be honest, I don’t know anything about that. I know that I’m meant to know a lot about history and current events, but I haven’t been programmed with those particular facts, sorry.', '\n\nHuman: How can I find out what types of butterflies are in my area?\n\nAssistant: Which location are you in?\n\nHuman: I am in Oregon.\n\nAssistant: There are about 175 species of butterflies in Oregon, of which 100 are long-distance migrants from southern California.  Also, some of the common butterflies in Oregon\n\nHuman: Great. What are some common species then?\n\n\nAssistant: About 150 species of butterflies live in Oregon, with about 100 species are moths, and about 20 species are common here year-round, and another 10 species are seen here year-round.  I suggest you keep an eye out for skippers, gossamer wings, and red admirals.', '\n\nHuman: Can you tell me about wild dogs? Not stray dogs, but wild dogs.\n\nAssistant: I know a lot about wild dogs. Would you like to read about what a wolf is, or about coyotes? I know they’re both kind of similar to the dogs that roam the streets, but they’re more closely related to domesticated dogs than to the street dogs.\n\nHuman: Oh, I would love to learn about wolves.\n\n\nAssistant: Sure! Here’s some fun facts: A wolf is a large canine, and it’s typically found in cold climates. The most popular type of wolf is the gray wolf. Wolves are able to smell things about 6 kilometers away, and their sense of hearing can even detect sounds below 30 Hz, which is really low. They have acute eyesight, and can be taught to follow a ball with their eyes, and even read. They can be trained to do many tasks, and may even be trained to deliver newspapers. They’re very social animals, and make excellent pets.']

inputs = tokenizer(examples, return_tensors='pt',
                       truncation=True, max_length=512, padding="max_length").to('cuda')
with torch.no_grad():
    rewards = model(**inputs)

print(rewards)
import torch
from utils import make_rm
# save_model is used to determine whether a reference to the base model is saved in the RM wrapper (this is necessary to use HF's Activation Checkpointing code)
save_model = False
model = make_rm("CarperAI/openai_summarize_tldr_sft", "causal", "EleutherAI/gpt-j-6B")
model.load_state_dict(torch.load("/fsx/home-duyphung/all_checkpoints/gptj-rm-summarize-22Feb/checkpoint-11385/pytorch_model.bin"), strict=True)


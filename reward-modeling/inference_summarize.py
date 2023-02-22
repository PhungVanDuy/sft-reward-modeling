import torch
from utils import make_rm
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from rm_datasets import PairwiseDataset, PairwiseEvalDataset, pairwise_data_collator, ranked_data_collator, RankedDataset, RankedEvalDataset
# save_model is used to determine whether a reference to the base model is saved in the RM wrapper (this is necessary to use HF's Activation Checkpointing code)
save_model = False
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer.pad_token = tokenizer.eos_token
model = make_rm("CarperAI/openai_summarize_tldr_sft", "causal", "EleutherAI/gpt-j-6B")
model.load_state_dict(torch.load("/fsx/home-duyphung/all_checkpoints/gptj-rm-summarize-23Feb/checkpoint-22770/pytorch_model.bin"), strict=True)

data = load_dataset("CarperAI/openai_summarize_comparisons")
eval_data = data["test"].select(range(10000))

eval_dataset = PairwiseEvalDataset(eval_data, tokenizer, max_length=550)

eval_loader = DataLoader(eval_dataset, batch_size=2, collate_fn=pairwise_data_collator)
model.cuda()
model.half()
total_acc = 0
count = 0
for batch in tqdm(eval_loader):
    for x in batch:
        batch[x] = batch[x].cuda()
    preds = model(**batch)
    preds = preds.view(-1, 2)
    acc = sum(preds[:, 0] > preds[:, 1]) / preds.shape[0]
    total_acc += acc
    count += 1
print("Accuracy: ", total_acc / count)



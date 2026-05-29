import random

import datasets
import torch


class TokenBlockDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, tokenizer, block_size):
        token_ids = []
        for row in hf_dataset:
            token_ids.extend(tokenizer(row["text"])["input_ids"])
        total = (len(token_ids) // block_size) * block_size
        token_ids = token_ids[:total]
        self.blocks = [
            torch.tensor(token_ids[i : i + block_size], dtype=torch.long)
            for i in range(0, total, block_size)
        ]

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        input_ids = self.blocks[idx]
        return {"input_ids": input_ids, "labels": input_ids.clone()}


def train_dataset(name, tokenizer, block_size):
    if name != "wikitext2":
        raise ValueError("rotation training currently supports only wikitext2")
    ds = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")["train"]
    return TokenBlockDataset(ds, tokenizer, block_size)


def eval_tokens(name, tokenizer, seqlen, nsamples=256, seed=0):
    if name == "wikitext2":
        ds = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")["test"]
        text = "\n\n".join(ds["text"])
        return tokenizer(text, return_tensors="pt").input_ids

    if name == "c4":
        stream = datasets.load_dataset(
            "allenai/c4", "en", split="validation", streaming=True
        )
        stream = stream.shuffle(seed=seed, buffer_size=10_000)
        texts = []
        for row in stream:
            if row.get("text"):
                texts.append(row["text"])
            if len(texts) >= nsamples:
                break
        return tokenizer("\n\n".join(texts), return_tensors="pt").input_ids

    raise ValueError(f"unsupported eval dataset: {name}")


def random_calibration_loader(name, tokenizer, nsamples, seqlen, seed=0):
    if name != "wikitext2":
        raise ValueError("calibration currently supports only wikitext2")
    ds = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")["train"]
    enc = tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids
    random.seed(seed)
    out = []
    for _ in range(nsamples):
        i = random.randint(0, enc.shape[1] - seqlen - 1)
        block = enc[:, i : i + seqlen]
        out.append(block)
    return out

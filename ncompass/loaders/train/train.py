from ncompass.internal.loaders.train import load, load_split
from ncompass.models.snn.spike_gpt import SpikeGPTConfig
from ncompass.internal.logging import INFO

import ncompass.models.snn.spike_gpt as spike_gpt

from transformers import default_data_collator
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from itertools import chain

# ====== DATASET ======

def lm_dataset(dataset,
               ctx_length: int = 1,
               num_workers: int = 1,
               load_from_cache_file: bool = False) -> DatasetDict:

    # from HF example
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < ctx_length  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // ctx_length) * ctx_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + ctx_length] for i in range(0, total_length, ctx_length)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    return dataset.map(
        group_texts,
        batched=True,
        num_proc=num_workers,
        load_from_cache_file=load_from_cache_file,
        desc=f"Grouping texts in chunks of {ctx_length}",
    )

def tokenized_dataset(dataset: DatasetDict,
                      tokenizer,
                      num_workers: int = 1,
                      load_from_cache_file: bool = False) -> DatasetDict:

    column_names = dataset["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize(examples):
        return tokenizer(examples[text_column_name])

    return dataset.map(
        tokenize,
        batched=True,
        num_proc=num_workers,
        remove_columns=column_names,
        load_from_cache_file=load_from_cache_file,
        desc="Running tokenizer on dataset",
    )

def raw_dataset(path: str, name: str, split_percentage: int = None) -> DatasetDict:
    if split_percentage != None:
        ds = load_split(path, name, split_percentage)
        INFO(f"Loading dataset: {name}, {path} with {split_percentage}% train:val split")
    else:
        ds = load(path, name)
        INFO(f"Loading dataset: {name}, {path} with default train:val split")
    return ds

def load_dataset(path: str, name: str, tokenizer,
                 split_percentage: int = None, num_workers: int = 1,
                 ctx_length: int = 1024, load_from_cache_file: bool = True) -> DatasetDict:
    raw_data = raw_dataset(path=path, name=name, split_percentage=split_percentage)
    tokened = tokenized_dataset(raw_data, tokenizer, num_workers=num_workers,
                                load_from_cache_file=load_from_cache_file)
    dataset = lm_dataset(tokened, ctx_length=ctx_length, num_workers=num_workers, load_from_cache_file=True)
    return dataset

def load_wikitext2(tokenizer, split_percentage: int = None) -> DatasetDict:
    return load_dataset(path='wikitext',
                        name='wikitext-2-raw-v1',
                        tokenizer=tokenizer,
                        split_percentage=split_percentage)

def load_wikitext103(tokenizer, split_percentage: int = None) -> DatasetDict:
    return load_dataset(path='wikitext',
                        name='wikitext-103-raw-v1',
                        tokenizer=tokenizer,
                        split_percentage=split_percentage)


# ====== DATA LOADER ======

def get_dataloader(dataset: Dataset,
                   shuffle: bool = True,
                   collate_fn = default_data_collator,
                   batch_size: int = 8):
    return DataLoader(dataset,
                      shuffle=shuffle,
                      collate_fn=default_data_collator,
                      batch_size=batch_size)

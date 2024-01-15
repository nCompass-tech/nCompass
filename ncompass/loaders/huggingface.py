from typing import TypeVar
from transformers import BatchEncoding
from datasets import load_dataset, Dataset, DatasetDict, IterableDatasetDict, IterableDataset

import ncompass.internal.logging as nclog

HFDataset = TypeVar("HFDataset", Dataset, DatasetDict, IterableDatasetDict, IterableDataset)

#TODO: Check how linked this is to Tokenizers and see exactly where to place this function
def get_encodings(\
        dataset: HFDataset\
        , tokenizer\
        , dataset_type: str = "text"\
        , encoding_data_type: str = "pt") \
        -> BatchEncoding :
    return tokenizer.tokenizer(\
            '\n\n'.join(dataset[dataset_type])\
            , return_tensors = encoding_data_type)

def get_wikitext2(\
        split: str\
        , path: str = "wikitext"\
        , name: str = "wikitext-2-raw-v1"\
        , **kwargs):
    if not (split == "test" or split == "train" or split == "val"):
        nclog.ERROR(f"Invalid split type: {split}. Must be one of: test / train / val", 
                    ValueError)
    nclog.INFO(f"Loading dataset : {path}, {name} with split {split}")
    return load_dataset(path, name, split=split, **kwargs)


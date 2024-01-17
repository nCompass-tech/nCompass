from typing import TypeVar

from transformers import BatchEncoding
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from datasets import load_dataset, Dataset, DatasetDict, IterableDatasetDict, IterableDataset

import ncompass.internal.logging as nclog
from ncompass.internal.utils import validate_arg

HFDataset_t = TypeVar("HFDataset_t", Dataset, DatasetDict, IterableDatasetDict, IterableDataset)

#TODO: Check how linked this is to Tokenizers and see exactly where to place this function
def get_encodings(\
        dataset: HFDataset_t\
        , tokenizer: PreTrainedTokenizerBase\
        , dataset_type: str = "text"\
        , encoding_data_type: str = "pt") \
        -> BatchEncoding :
    return tokenizer(\
            '\n\n'.join(dataset[dataset_type])\
            , return_tensors = encoding_data_type)

def get_wikitext2(\
        split: str\
        , path: str = "wikitext"\
        , name: str = "wikitext-2-raw-v1"\
        , **kwargs):
    validate_arg(split, ["train", "val", "test"])
    nclog.INFO(f"Loading dataset : {path}, {name} with split {split}")
    return load_dataset(path, name, split=split, **kwargs)


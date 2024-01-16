from datasets import load_dataset, DatasetDict

def load(name: str, config_name: str) -> DatasetDict:
    return load_dataset(name, config_name)    

def load_split(name: str, config_name: str, split_percentage: int) -> DatasetDict:
    dataset = DatasetDict()
    dataset['train']      = load_dataset(name, config_name, split=f'train[{split_percentage}%:]')
    dataset['validation'] = load_dataset(name, config_name, split=f'train[:{split_percentage}%]')
    return dataset


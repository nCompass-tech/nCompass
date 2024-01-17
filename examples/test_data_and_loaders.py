import os
import sys

import ncompass.loaders.train as train_loaders
import ncompass.models.snn.spike_gpt as spike_gpt

from ncompass.models import download_from_hf
from ncompass.models.snn.spike_gpt import SpikeGPTConfig

def setup_config():
    home_dir = os.getenv("HOME")
    tokenizer_json_path = home_dir + "/nCompass/ncompass/models/snn/spike_gpt/20B_tokenizer.json"
    model_path = download_from_hf(repo_id = "ridger/SpikeGPT-OpenWebText-216M"\
                                  , filename = "SpikeGPT-216M.pth")
    config = SpikeGPTConfig(token_mode = "pile"
                            , word_name = [tokenizer_json_path, tokenizer_json_path]
                            , unknown_char = None
                            , vocab_size = 50277
                            , hidden_size = 768
                            , num_hidden_layers = 18
                            , ctx_len = 1024
                            , tokenizer_class = 'SpikeGPT'
                            , temperature = 1.5
                            , top_p = 0.7
                            , name_or_path = model_path)
    return config

def similarity_tests(dataset, dataset2):
    test = similarity_test(dataset['test'], dataset2['test'])
    train = similarity_test(dataset['train'], dataset2['train'])
    validation = similarity_test(dataset['validation'], dataset2['validation'])
    return bool(test == train == validation == True)

def similarity_test(dataset, dataset2):
    test1 = (dataset.features == dataset2.features)
    test2 = (dataset.num_rows == dataset2.num_rows)
    return bool(test1 == test2 == True)

def test_dataset():
    raw = train_loaders.raw_dataset('wikitext', 'wikitext-2-raw-v1')
    config = setup_config()
    tokenizer = spike_gpt.get_tokenizer(config)
    tokened = train_loaders.tokenized_dataset(raw, tokenizer,
                                              num_workers=4,
                                              load_from_cache_file=True)
    dataset = train_loaders.lm_dataset(tokened,
                                       ctx_length=1024,
                                       num_workers=4,
                                       load_from_cache_file=True)

    dataset2 = train_loaders.load_dataset('wikitext', 'wikitext-2-raw-v1', tokenizer=tokenizer, num_workers=4)
    print(f'{similarity_tests(dataset, dataset2)}')

    train_loaders.load_wikitext2(tokenizer)
    train_loaders.load_wikitext2(tokenizer, 38)

    train_loaders.load_wikitext103(tokenizer)
    train_loaders.load_wikitext103(tokenizer, 42)

def test_dataloader():
    config    = setup_config()
    tokenizer = spike_gpt.get_tokenizer(config)
    dataset   = train_loaders.load_dataset('wikitext', 'wikitext-2-raw-v1', tokenizer=tokenizer, num_workers=4)
    train_dataloader = train_loaders.get_dataloader(dataset['train'], shuffle=True)
    val_dataloader   = train_loaders.get_dataloader(dataset['validation'], shuffle=True)

def main():
    test_dataset()
    test_dataloader()

 
if __name__=='__main__':
    main()

import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional
from transformers import BatchEncoding

import ncompass.models.ann.gpt2 as gpt2
from ncompass.loaders.huggingface import get_wikitext2, get_encodings

def sliding_window_inference(\
            model: nn.Module\
          , encodings: BatchEncoding\
          , stride: Optional[int] = None) -> float:
    ctx_len = model.config.n_positions
    _stride = ctx_len if stride is None else stride
    num_tokens = encodings.input_ids.size(1)
    print(f"Running with stride {_stride}")

    neg_log_likelihood_list = []
    prev_ctx_end = 0
    device = model.device
    for ctx_start in tqdm(range(0, num_tokens, _stride)):
        ctx_end = min(ctx_start + ctx_len, num_tokens)
        trg_len = ctx_end - prev_ctx_end 
        input_ids = encodings.input_ids[:, ctx_start:ctx_end].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            output = model(input_ids, labels = target_ids)
            neg_log_likelihood = output.loss
        neg_log_likelihood_list.append(neg_log_likelihood)
        prev_ctx_end = ctx_end
    ppl = torch.exp(torch.stack(neg_log_likelihood_list).mean())
    return float(ppl)

def test_inference():
    model = gpt2.get_hf_model("xl").to("cuda:0")
    tokenizer = gpt2.get_hf_tokenizer()
    dataset = get_wikitext2("test")
    encodings = get_encodings(dataset, tokenizer) 
    ppl = sliding_window_inference(model, encodings)
    print(f"Perplexity = {ppl:.2f}")

if __name__ == "__main__":
    test_gpt2_inference()

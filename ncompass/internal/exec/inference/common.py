from tqdm import tqdm

import torch

def iterate_windows(
        model
        , all_tokens
        , num_tokens: int
        , stride: int
        , window_len: int
        , infinite_context: bool
        , runner_fn
        , runner_fn_args: dict
        , enable_prog_bar: bool = True): 
    iter_space = range(0, num_tokens, stride)
    pbar = tqdm(iter_space, total=len(iter_space), disable=(not enable_prog_bar))
    prev_window_end = 0
    neg_log_likelihood_list = []
    fn_args = runner_fn_args.copy()
    for window_start in pbar:
        window_end = min(window_start + window_len, num_tokens)
        encoding_start = 0 if infinite_context else window_start
        input_ids = all_tokens[0, encoding_start:window_end]

        fn_args["model"] = model
        fn_args["input_ids"] = input_ids
        fn_args["encoding_start"] = encoding_start
        fn_args["window_start"] = window_start
        fn_args["window_end"] = window_end
        fn_args["prev_window_end"] = prev_window_end
        loss, _fn_args = runner_fn(**fn_args)
        fn_args.update(_fn_args)

        neg_log_likelihood_list.append(loss)
        ppl = torch.exp(torch.stack(neg_log_likelihood_list).mean())
        pbar.set_description(f"Perplexity = {ppl:.2f}")
        prev_window_end = window_end
    ppl = torch.exp(torch.stack(neg_log_likelihood_list).mean())
    return float(ppl)





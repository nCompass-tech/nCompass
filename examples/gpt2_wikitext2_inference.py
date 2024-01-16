import ncompass.models.ann.gpt2 as gpt2
from ncompass.loaders.huggingface import get_wikitext2, get_encodings
from ncompass.exec.inference import \
          _run_sliding_window_perplexity\
        , run_sliding_window_perplexity\
        , get_slw_config\
        , get_exec_config

def test_slw_config(model, encodings):
    cfg = get_slw_config(model=model, encodings=encodings, stride=1024)
    ppl = run_sliding_window_perplexity(cfg)
    return ppl

def test_exec_config(model, encodings):
    cfg = get_exec_config(task = "tmp", model = model, encodings = encodings)
    ppl = run_sliding_window_perplexity(cfg)
    return ppl

def test_direct_call(model, encodings):
    return _run_sliding_window_perplexity(model, encodings)

def test_inference():
    model = gpt2.get_hf_model().to("cuda:0")
    tokenizer = gpt2.get_hf_tokenizer()
    dataset = get_wikitext2("test")
    encodings = get_encodings(dataset, tokenizer) 
    ppl = test_slw_config(model, encodings)
    print(f"Perplexity = {ppl:.2f}")

def test_pipeline_load():
    gpt2.get_hf_pipeline()
    gpt2.get_hf_pipeline(size="medium")
    gpt2.get_hf_pipeline(size="large")
    gpt2.get_hf_pipeline(size="xl")

if __name__ == "__main__":
    test_inference()
    # test_pipeline_load()

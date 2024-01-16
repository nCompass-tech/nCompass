import os
import ncompass as nc
from ncompass.exec.inference import run_sliding_window_perplexity, get_slw_config

def get_model_config():
    home_dir = os.getenv("HOME")
    tokenizer_json_path = home_dir + "/nCompass/ncompass/models/snn/spike_gpt/20B_tokenizer.json"
    # model_path = nc.models.download_from_hf(\
    #                   repo_id = "ridger/SpikeGPT-OpenWebText-216M"\
    #                 , filename = "SpikeGPT-216M.pth")
    model_path = "/home/ubuntu/spike-gpt/SpikeGPT-OpenWebText-216M/train1.pth"
    return nc.models.snn.spike_gpt.SpikeGPTConfig(\
                      token_mode = "pile"\
                    , word_name = [tokenizer_json_path, tokenizer_json_path]\
                    , unknown_char = None\
                    , vocab_size = 50277\
                    , hidden_size = 768\
                    , num_hidden_layers = 18\
                    , ctx_len = 1024\
                    , tokenizer_class = 'SpikeGPT'\
                    , temperature = 1.5\
                    , top_p = 0.7\
                    , name_or_path = model_path)

def test_inference():
    dataset = nc.loaders.huggingface.get_wikitext2("test") 
    sgpt_config = get_model_config()
    tokenizer = nc.models.get_tokenizer(sgpt_config)
    encodings = nc.loaders.huggingface.get_encodings(dataset, tokenizer.tokenizer)
    inference_model = nc.models.snn.spike_gpt.inference.SpikeGPT(sgpt_config)
    
    slw_config = get_slw_config(
              model = inference_model
            , encodings = encodings
            , ctx_len = 1024
            , stride = 1
            , num_warmup_tokens = 1024
            , nc_model_type = "spikegpt") 
    ppl = run_sliding_window_perplexity(slw_config)
    print(ppl)

if __name__ == '__main__':
    test_inference()

import os
import ncompass as nc

def test_inference():
    dataset = nc.loaders.huggingface.get_wikitext2("test") 
    pipeline = nc.models.ann.gpt2.get_hf_pipeline(size="medium")
    
    home_dir = os.getenv("HOME")
    tokenizer_json_path = home_dir + "/nCompass/ncompass/models/snn/spike_gpt/20B_tokenizer.json"
    model_path = nc.models.download_from_hf(\
                      repo_id = "ridger/SpikeGPT-OpenWebText-216M"\
                    , filename = "SpikeGPT-216M.pth")
    sgpt_config = nc.models.snn.spike_gpt.SpikeGPTConfig(\
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
    tokenizer = nc.models.get_tokenizer(sgpt_config)
    encodings = nc.loaders.huggingface.get_encodings(dataset, tokenizer.tokenizer)
    
    inference_model = nc.models.snn.spike_gpt.inference.SpikeGPT(sgpt_config)

if __name__ == '__main__':
    test_inference()
